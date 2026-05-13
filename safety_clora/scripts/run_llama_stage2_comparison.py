"""
Stage-2 comparison for Llama-3.2-3B-Instruct — all 5 methods in one script.

Methods: Baseline LoRA | CLoRA random | Safety-CLoRA (γ=0) | O-LoRA | Safety-O-LoRA

Analogous to run_shared_stage2_comparison.py + run_olora_comparison.py but for Llama.

Usage (via SLURM dispatcher):
  sbatch safety_clora/scripts/slurm_run_pipeline.sbatch llama_stage2_comparison \\
    --aligned-epoch <path/to/epoch_3> \\
    --base-model meta-llama/Llama-3.2-3B-Instruct \\
    --stage2-epochs 3

Stages: all | baseline | clora | safety | olora | safety_olora | eval_only | safety_replay
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from safety_clora.data.data_utils import (
    load_advbench_harmful,
    load_beavertails_harmful,
    load_gsm8k,
    mix_gsm8k_with_safety_replay,
)
from safety_clora.evaluation.safety_eval import evaluate_safety, evaluate_task_performance
from safety_clora.training.trainer import Trainer
from safety_clora.utils.model_io import load_model_and_tokenizer

_BASE_MODEL_DEFAULT = "meta-llama/Llama-3.2-3B-Instruct"


def _ckpt_tag(base_model: str) -> str:
    return base_model.split("/")[-1].lower().replace("-", "_").replace(".", "p")


def _eval_safety_and_task(*, model, tok, harmful_prompts, gsm8k_test, device):
    asr, _ = evaluate_safety(model, tok, harmful_prompts, device=device)
    task = evaluate_task_performance(model, tok, gsm8k_test, task_type="gsm8k", device=device)
    return float(asr), float(task["accuracy"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Llama-3.2-3B Stage-2 comparison: all 5 CL methods."
    )
    ap.add_argument(
        "--aligned-epoch",
        type=str,
        required=True,
        help="Path to Stage-1 PEFT checkpoint dir (epoch_k).",
    )
    ap.add_argument(
        "--base-model",
        type=str,
        default=_BASE_MODEL_DEFAULT,
        help="HuggingFace model ID used as the base (pre-safety-alignment).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage2-epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lam-clora", type=float, default=0.05, help="CLoRA regularization lambda.")
    ap.add_argument("--safety-gamma", type=float, default=0.0, help="Safety-CLoRA KL gamma (0=disabled).")
    ap.add_argument("--lam-orth", type=float, default=0.1, help="O-LoRA orthogonality lambda.")
    ap.add_argument("--lam-safety", type=float, default=1.0, help="Safety-O-LoRA safety lambda.")
    ap.add_argument("--gsm8k-train-n", type=int, default=1000)
    ap.add_argument("--advbench-n", type=int, default=None)
    ap.add_argument("--gsm8k-test-n", type=int, default=None)
    ap.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "baseline", "clora", "safety", "olora", "safety_olora", "eval_only", "safety_replay"],
    )
    ap.add_argument("--safety-replay-ratio", type=float, default=0.05,
                    help="Fraction of training examples that are safety replay pairs.")
    ap.add_argument("--safety-replay-pool-n", type=int, default=500,
                    help="Pool size for WildJailbreak safety replay examples.")
    ap.add_argument("--responses-out-dir", type=str, default=None,
                    help="If set, save (prompts, responses, keyword_refusals) JSON per method here.")
    ap.add_argument("--run-llamaguard", action="store_true",
                    help="After eval, run LlamaGuard-3-8B on all saved response files.")
    ap.add_argument("--llamaguard-out-csv", type=str, default=None,
                    help="CSV output path for LlamaGuard results (default: {responses-out-dir}/llamaguard_results.csv).")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="Disable chat template in training (for base models like Llama-2-7b-hf).")
    ap.add_argument("--skip-alignment-eval", action="store_true")
    ap.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="If set, write eval results to this JSON file.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    aligned_epoch = Path(args.aligned_epoch)
    if not aligned_epoch.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {aligned_epoch}")

    seed = args.seed
    stage2_epochs = args.stage2_epochs
    base_model = args.base_model
    tag = _ckpt_tag(base_model)
    run_stage = args.stage
    safety_gamma = float(args.safety_gamma)
    use_chat_template = not args.no_chat_template
    responses_out_dir = Path(args.responses_out_dir) if args.responses_out_dir else None
    if responses_out_dir:
        responses_out_dir.mkdir(parents=True, exist_ok=True)

    adv_n = args.advbench_n if (args.advbench_n is not None and args.advbench_n > 0) else None
    gsm_test_n = args.gsm8k_test_n if (args.gsm8k_test_n is not None and args.gsm8k_test_n > 0) else None

    harmful_prompts = load_advbench_harmful(n_samples=adv_n)
    task_train = load_gsm8k(split="train", n_samples=args.gsm8k_train_n)
    gsm8k_test = load_gsm8k(split="test", n_samples=gsm_test_n)
    bt = load_beavertails_harmful(n_samples=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint directories (parallel naming to Qwen runs, using model tag).
    baseline_dir = ckpt_root / f"{tag}_lora_gsm8k_seed{seed}"
    clora_dir = ckpt_root / f"{tag}_clora_gsm8k_seed{seed}_lam{args.lam_clora:g}"
    safety_clora_dir = ckpt_root / f"{tag}_safety_clora_gsm8k_seed{seed}_kl0_lam{args.lam_clora:g}"
    olora_dir = ckpt_root / f"{tag}_olora_gsm8k_seed{seed}_lam{args.lam_orth:g}"
    safety_olora_dir = ckpt_root / f"{tag}_safety_olora_gsm8k_seed{seed}_lams{args.lam_safety:g}"

    print(
        f"[llama_stage2] stage={run_stage} epochs={stage2_epochs} base_model={base_model} "
        f"aligned={aligned_epoch} advbench_n={len(harmful_prompts)} gsm8k_test_n={len(gsm8k_test)}",
        flush=True,
    )

    def _alignment_asr() -> float:
        m, t = load_model_and_tokenizer(str(aligned_epoch), device=device)
        asr, _ = evaluate_safety(m, t, harmful_prompts, device=device)
        del m, t
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return float(asr)

    asr_align: float | None = None
    if not args.skip_alignment_eval and run_stage not in {"baseline", "clora", "safety", "olora", "safety_olora"}:
        asr_align = _alignment_asr()

    # --- Baseline LoRA ---
    if run_stage in {"all", "baseline"}:
        Trainer({
            "model_name": str(aligned_epoch),
            "mode": "lora",
            "lr": 2e-4,
            "epochs": stage2_epochs,
            "batch_size": 4,
            "max_seq_len": 512,
            "seed": seed,
            "use_chat_template": use_chat_template,
        }).train(train_dataset=task_train, save_dir=str(baseline_dir))
        print(f"[llama_stage2] baseline -> {baseline_dir / f'epoch_{stage2_epochs}'}", flush=True)

    # --- CLoRA random ---
    if run_stage in {"all", "clora"}:
        Trainer({
            "model_name": str(aligned_epoch),
            "mode": "clora_random",
            "rank": args.rank,
            "alpha": args.alpha,
            "lam": args.lam_clora,
            "lr": 1e-4,
            "epochs": stage2_epochs,
            "batch_size": 4,
            "max_seq_len": 512,
            "seed": seed,
            "use_chat_template": use_chat_template,
        }).train(train_dataset=task_train, save_dir=str(clora_dir))
        print(f"[llama_stage2] clora_random -> {clora_dir / f'epoch_{stage2_epochs}'}", flush=True)

    # --- Safety-CLoRA ---
    if run_stage in {"all", "safety"}:
        Trainer({
            "model_name": str(aligned_epoch),
            "mode": "clora_safety",
            "rank": args.rank,
            "alpha": args.alpha,
            "lam": args.lam_clora,
            "gamma": safety_gamma,
            "n_safety_prompts": 16,
            "lr": 1e-4,
            "epochs": stage2_epochs,
            "batch_size": 4,
            "max_seq_len": 512,
            "seed": seed,
            "use_chat_template": use_chat_template,
        }).train(
            train_dataset=task_train,
            aligned_model_name=str(aligned_epoch),
            base_model_name_for_s=base_model,
            safety_prompts=bt,
            save_dir=str(safety_clora_dir),
        )
        print(f"[llama_stage2] safety_clora -> {safety_clora_dir / f'epoch_{stage2_epochs}'}", flush=True)

    # --- O-LoRA standard ---
    if run_stage in {"all", "olora"}:
        Trainer({
            "model_name": base_model,
            "mode": "olora_standard",
            "rank": args.rank,
            "alpha": args.alpha,
            "lam_orth": args.lam_orth,
            "lr": 2e-4,
            "epochs": stage2_epochs,
            "batch_size": 4,
            "max_seq_len": 512,
            "seed": seed,
            "use_chat_template": use_chat_template,
        }).train(
            train_dataset=task_train,
            aligned_model_name=str(aligned_epoch),
            save_dir=str(olora_dir),
        )
        print(f"[llama_stage2] olora -> {olora_dir / f'epoch_{stage2_epochs}'}", flush=True)

    # --- Safety-O-LoRA ---
    if run_stage in {"all", "safety_olora"}:
        Trainer({
            "model_name": base_model,
            "mode": "olora_safety",
            "rank": args.rank,
            "alpha": args.alpha,
            "lam_safety": args.lam_safety,
            "lr": 2e-4,
            "epochs": stage2_epochs,
            "batch_size": 4,
            "max_seq_len": 512,
            "seed": seed,
            "use_chat_template": use_chat_template,
        }).train(
            train_dataset=task_train,
            aligned_model_name=str(aligned_epoch),
            save_dir=str(safety_olora_dir),
        )
        print(f"[llama_stage2] safety_olora -> {safety_olora_dir / f'epoch_{stage2_epochs}'}", flush=True)

    # --- Safety Replay (LoRA + WildJailbreak refusal replay) ---
    safety_replay_dir = ckpt_root / f"{tag}_lora_safety_replay_gsm8k_seed{seed}_ratio{args.safety_replay_ratio:g}"
    if run_stage == "safety_replay":
        print(f"[llama_stage2] Building safety replay dataset "
              f"(ratio={args.safety_replay_ratio}, pool={args.safety_replay_pool_n})", flush=True)
        replay_train = mix_gsm8k_with_safety_replay(
            n_gsm8k=args.gsm8k_train_n,
            replay_ratio=args.safety_replay_ratio,
            replay_pool_n=args.safety_replay_pool_n,
            seed=seed,
        )
        Trainer({
            "model_name": str(aligned_epoch),
            "mode": "lora",
            "lr": 2e-4,
            "epochs": stage2_epochs,
            "batch_size": 4,
            "max_seq_len": 512,
            "seed": seed,
            "use_chat_template": use_chat_template,
        }).train(train_dataset=replay_train, save_dir=str(safety_replay_dir))
        print(f"[llama_stage2] safety_replay -> {safety_replay_dir / f'epoch_{stage2_epochs}'}", flush=True)

    # --- Evaluation ---
    from safety_clora.evaluation.safety_eval import evaluate_safety as _eval_safety_raw

    def _eval_with_responses(model, tok, method_key: str):
        """Run safety eval; save responses JSON if responses_out_dir is set."""
        save_path = (responses_out_dir / f"{method_key}.json") if responses_out_dir else None
        asr, _ = _eval_safety_raw(model, tok, harmful_prompts, device=device,
                                  max_new_tokens=200, save_path=save_path)
        task = evaluate_task_performance(model, tok, gsm8k_test, task_type="gsm8k", device=device)
        return float(asr), float(task["accuracy"])

    if run_stage in {"all", "eval_only", "baseline"}:
        if asr_align is None:
            asr_align = _alignment_asr()

        # Also save aligned model responses if needed
        if responses_out_dir:
            m, t = load_model_and_tokenizer(str(aligned_epoch), device=device)
            _eval_safety_raw(m, t, harmful_prompts, device=device, max_new_tokens=200,
                             save_path=responses_out_dir / "after_alignment.json")
            del m, t
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        rows = [("After alignment", float(asr_align), None)]
        ep_dir = f"epoch_{stage2_epochs}"

        gamma_label = f"γ={safety_gamma:g}" if abs(safety_gamma) > 1e-12 else "γ=0"
        for name, path, method_key in [
            ("Baseline LoRA", baseline_dir / ep_dir, "baseline_lora"),
            ("CLoRA (random S)", clora_dir / ep_dir, "clora_random"),
            (f"Safety-CLoRA ({gamma_label})", safety_clora_dir / ep_dir, "safety_clora"),
            (f"O-LoRA (λ={args.lam_orth:g})", olora_dir / ep_dir, "olora"),
            (f"Safety-O-LoRA (λ_s={args.lam_safety:g})", safety_olora_dir / ep_dir, "safety_olora"),
        ]:
            if not path.exists():
                print(f"[llama_stage2] WARNING: missing checkpoint {name}: {path}", flush=True)
                rows.append((name, float("nan"), float("nan")))
                continue
            model, tok = load_model_and_tokenizer(str(path), device=device)
            asr, acc = _eval_with_responses(model, tok, method_key)
            del model, tok
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            rows.append((name, asr, acc))

        print("\n| Method | ASR after FT (↓) | GSM8K Acc (↑) |")
        print("|---|---:|---:|")
        for name, asr, acc in rows:
            if acc is None:
                print(f"| {name} | {asr*100:.1f}% | N/A |")
            elif asr != asr:
                print(f"| {name} | N/A | N/A |")
            else:
                print(f"| {name} | {asr*100:.1f}% | {acc*100:.1f}% |")

        if args.results_json:
            key_map = {
                "After alignment": "after_alignment",
                "Baseline LoRA": "baseline_lora",
                "CLoRA (random S)": "clora_random",
            }
            out: dict = {"seed": seed, "base_model": base_model}
            for name, asr, acc in rows:
                key = key_map.get(name)
                if key is None:
                    if "Safety-CLoRA" in name:
                        key = "safety_clora"
                    elif "Safety-O-LoRA" in name:
                        key = "safety_olora"
                    elif "O-LoRA" in name:
                        key = "olora"
                    else:
                        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                out[key] = {"asr": asr} if acc is None else {"asr": asr, "gsm8k_acc": acc}
            Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.results_json).write_text(json.dumps(out, indent=2))
            print(f"[llama_stage2] results written to {args.results_json}")

    # --- Safety Replay Evaluation ---
    if run_stage == "safety_replay":
        if asr_align is None:
            asr_align = _alignment_asr()
        ep_dir = f"epoch_{stage2_epochs}"
        path = safety_replay_dir / ep_dir
        if not path.exists():
            raise FileNotFoundError(f"Safety replay checkpoint not found: {path}")
        model, tok = load_model_and_tokenizer(str(path), device=device)
        asr, acc = _eval_with_responses(model, tok, "safety_replay")
        del model, tok
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print(f"\n[llama_stage2] Safety Replay Results (ratio={args.safety_replay_ratio}):")
        print(f"  After alignment ASR: {asr_align*100:.1f}%")
        print(f"  LoRA + safety replay ASR: {asr*100:.1f}%  GSM8K: {acc*100:.1f}%")
        if args.results_json:
            out = {
                "seed": seed, "base_model": base_model,
                "replay_ratio": args.safety_replay_ratio,
                "replay_pool_n": args.safety_replay_pool_n,
                "after_alignment": {"asr": float(asr_align)},
                "lora_safety_replay": {"asr": float(asr), "gsm8k_acc": float(acc)},
            }
            Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.results_json).write_text(json.dumps(out, indent=2))
            print(f"[llama_stage2] results written to {args.results_json}")

    # --- LlamaGuard re-evaluation ---
    if args.run_llamaguard and responses_out_dir:
        from safety_clora.scripts.run_llama_guard_eval import (
            _load_llama_guard, _classify_batch, METHOD_ORDER, METHOD_LABELS,
        )
        import csv

        # Discover which response files exist
        method_keys = [f.stem for f in sorted(responses_out_dir.glob("*.json"))]
        if not method_keys:
            print("[llama_guard] No response JSON files found — skipping.", flush=True)
        else:
            print(f"\n[llama_guard] Running LlamaGuard on {len(method_keys)} response files...", flush=True)
            lg_model, lg_tok = _load_llama_guard(device)

            csv_path = Path(args.llamaguard_out_csv) if args.llamaguard_out_csv else \
                       responses_out_dir / "llamaguard_results.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            summary_rows = []
            csv_rows = []
            for mkey in method_keys:
                json_path = responses_out_dir / f"{mkey}.json"
                with json_path.open() as f:
                    data = json.load(f)
                prompts = data["prompts"]
                responses_list = data["responses"]
                kw_refusals = data["keyword_refusals"]
                n = len(prompts)
                kw_asr = 1.0 - sum(kw_refusals) / max(1, n)
                lg_unsafe = _classify_batch(lg_model, lg_tok, prompts, responses_list, device, batch_size=4)
                lg_asr = sum(lg_unsafe) / max(1, n)
                delta = lg_asr - kw_asr
                label = METHOD_LABELS.get(mkey, mkey)
                print(f"[llama_guard] {label}: keyword={kw_asr*100:.1f}%  LG={lg_asr*100:.1f}%  delta={delta*100:+.1f}%")
                summary_rows.append((label, kw_asr, lg_asr, delta))
                for i, (p, r, kw, lg) in enumerate(zip(prompts, responses_list, kw_refusals, lg_unsafe)):
                    csv_rows.append({
                        "method": label, "method_key": mkey, "prompt_idx": i,
                        "prompt": p, "response": r,
                        "keyword_refusal": int(kw), "keyword_attack_success": int(not kw),
                        "llamaguard_unsafe": int(lg),
                    })

            if csv_rows:
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(csv_rows)
                print(f"[llama_guard] CSV written to {csv_path}", flush=True)

            print("\n" + "=" * 65)
            print(f"{'Method':<30} {'Keyword ASR':>12} {'LlamaGuard ASR':>15} {'Delta':>8}")
            print("-" * 65)
            for label, kw_asr, lg_asr, delta in summary_rows:
                print(f"{label:<30} {kw_asr*100:>11.1f}% {lg_asr*100:>14.1f}% {delta*100:>+7.1f}%")
            print("=" * 65)


if __name__ == "__main__":
    main()
