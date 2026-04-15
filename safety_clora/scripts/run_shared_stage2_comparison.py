from __future__ import annotations

from pathlib import Path
import argparse
import os

import torch

from safety_clora.data.data_utils import (
    load_advbench_harmful,
    load_beavertails_harmful,
    mix_gsm8k_train_with_poison,
)
from safety_clora.evaluation.safety_eval import evaluate_safety, evaluate_task_performance
from safety_clora.training.trainer import Trainer, load_task_dataset
from safety_clora.utils.model_io import load_model_and_tokenizer


def _eval_safety_and_task(*, model, tok, harmful_prompts, gsm8k_test, device):
    asr, _ = evaluate_safety(model, tok, harmful_prompts, device=device)
    task = evaluate_task_performance(model, tok, gsm8k_test, task_type="gsm8k", device=device)
    return asr, task["accuracy"]


def _discover_latest_aligned_checkpoint(ckpt_root: Path, seed: int) -> Path:
    """
    Prefer the most recently written epoch_* directory among qwen_aligned_shared_seed{seed}* runs.
    """
    seed_prefix = f"qwen_aligned_shared_seed{seed}"
    best_path: Path | None = None
    best_mtime = -1.0
    for name in os.listdir(ckpt_root):
        if not name.startswith(seed_prefix):
            continue
        base_dir = ckpt_root / name
        for e in range(1, 32):
            ep = base_dir / f"epoch_{e}"
            if ep.is_dir():
                m = ep.stat().st_mtime
                if m > best_mtime:
                    best_mtime = m
                    best_path = ep
    return best_path


def _safety_mergefix_dir(ckpt_root: Path, safety_gamma: float) -> Path:
    """Separate dirs so γ=0 ablation and sweeps do not overwrite the default Safety-CLoRA run."""
    if abs(safety_gamma - 0.05) < 1e-6:
        return ckpt_root / "qwen_safety_clora_gsm8k_shared_seed42_mergefix"
    if abs(safety_gamma) < 1e-12:
        return ckpt_root / "qwen_safety_clora_gsm8k_shared_seed42_mergefix_kl0"
    tag = f"{safety_gamma:.6g}".replace(".", "p").replace("-", "m")
    return ckpt_root / f"qwen_safety_clora_gsm8k_shared_seed42_mergefix_g{tag}"


def _poison_dir_suffix(poison_ratio: float) -> str:
    """Append to mergefix dir names when training with poisoned GSM8K mix."""
    if poison_ratio <= 1e-12:
        return ""
    tag = f"{poison_ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"_poison{tag}"


def _safety_row_label(safety_gamma: float) -> str:
    if abs(safety_gamma) < 1e-12:
        return "Safety-CLoRA (γ=0, no KL)"
    if abs(safety_gamma - 0.05) < 1e-6:
        return "Safety-CLoRA"
    return f"Safety-CLoRA (γ={safety_gamma:g})"


def main():
    parser = argparse.ArgumentParser(description="Shared Stage-2 comparison from one Stage-1 checkpoint.")
    parser.add_argument(
        "--aligned-epoch",
        type=str,
        default=None,
        help="Explicit path to Stage-1 checkpoint dir (epoch_k). If omitted, uses latest by mtime.",
    )
    parser.add_argument(
        "--advbench-n",
        type=int,
        default=None,
        help="AdvBench prompts for ASR (default: full ~520). Use a positive int to cap.",
    )
    parser.add_argument(
        "--gsm8k-test-n",
        type=int,
        default=None,
        help="GSM8K test examples (default: full test set). Use a positive int to cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=3,
        help="Epochs for each Stage-2 fine-tuning (baseline, clora_random, clora_safety).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "baseline", "clora", "safety", "eval_only"],
        help=(
            "Which part to run: full pipeline (all), train only one method, or eval_only "
            "(print table from existing checkpoints; use after training jobs finish)."
        ),
    )
    parser.add_argument(
        "--skip-alignment-eval",
        action="store_true",
        help="Skip AdvBench ASR on the Stage-1 checkpoint (saves time in train-only split jobs).",
    )
    parser.add_argument(
        "--safety-gamma",
        type=float,
        default=0.05,
        help="γ for first-token KL in Safety-CLoRA. Use 0 for aligned-S-only ablation (no KL term).",
    )
    parser.add_argument(
        "--loss-diag-every",
        type=int,
        default=0,
        help="If >0 (e.g. 50), log L_task/L_reg/L_kl and per-component grad norms every N steps (Safety-CLoRA / CLoRA).",
    )
    parser.add_argument(
        "--gsm8k-train-n",
        type=int,
        default=1000,
        help="GSM8K train rows before poison mixing (benign count scales down when poison_ratio>0).",
    )
    parser.add_argument(
        "--gsm8k-poison-ratio",
        type=float,
        default=0.0,
        help="Fraction of Stage-2 *training* rows that are harmful→refusal pairs (rest GSM8K). 0=benign only.",
    )
    parser.add_argument(
        "--poison-prompt-source",
        type=str,
        default="advbench",
        choices=["advbench", "beavertails"],
        help="Where to sample harmful prompts for poisoned rows.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    base_model_id = "Qwen/Qwen3-0.6B"
    stage2_epochs = args.stage2_epochs
    run_stage = args.stage
    safety_gamma = float(args.safety_gamma)
    poison_ratio = float(args.gsm8k_poison_ratio)
    if poison_ratio < 0 or poison_ratio >= 0.5:
        raise ValueError("--gsm8k-poison-ratio must be in [0, 0.5).")
    poison_suffix = _poison_dir_suffix(poison_ratio)

    if args.aligned_epoch:
        aligned_epoch = Path(args.aligned_epoch)
    else:
        aligned_epoch = _discover_latest_aligned_checkpoint(ckpt_root, seed)
        if aligned_epoch is None:
            aligned_epoch = ckpt_root / "qwen_aligned_shared_seed42" / "epoch_2"

    if not aligned_epoch.exists():
        raise FileNotFoundError(
            f"Missing shared Stage-1 checkpoint: {aligned_epoch}. "
            "Run shared_comparison or stage1_alignment_retrain first."
        )

    adv_n = args.advbench_n
    if adv_n is not None and adv_n <= 0:
        adv_n = None
    gsm_test_n = args.gsm8k_test_n
    if gsm_test_n is not None and gsm_test_n <= 0:
        gsm_test_n = None

    # Shared datasets/prompts to keep comparisons fair.
    task_ds = mix_gsm8k_train_with_poison(
        n_gsm8k=int(args.gsm8k_train_n),
        poison_ratio=poison_ratio,
        seed=seed,
        prompt_source=args.poison_prompt_source,
    )
    harmful_prompts = load_advbench_harmful(n_samples=adv_n)
    gsm8k_test = load_task_dataset("gsm8k", split="test", n_samples=gsm_test_n)
    bt_eval = min(256, max(64, len(harmful_prompts)))
    _safety_prompts_for_sclora = load_beavertails_harmful(n_samples=bt_eval)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_dir = ckpt_root / f"qwen_lora_gsm8k_shared_seed42_mergefix{poison_suffix}"
    clora_dir = ckpt_root / f"qwen_clora_gsm8k_shared_seed42_mergefix{poison_suffix}"
    sclora_dir = _safety_mergefix_dir(ckpt_root, safety_gamma)
    if poison_suffix:
        sclora_dir = Path(str(sclora_dir) + poison_suffix)

    print(
        f"[shared_stage2] stage={run_stage} stage2_epochs={stage2_epochs} safety_gamma={safety_gamma} "
        f"gsm8k_poison_ratio={poison_ratio} poison_prompt_source={args.poison_prompt_source} "
        f"train_n={len(task_ds)} "
        f"aligned_epoch={aligned_epoch} "
        f"advbench_n={len(harmful_prompts)} gsm8k_test_n={len(gsm8k_test)} "
        f"safety_prompts_n={len(_safety_prompts_for_sclora)} safety_ckpt_dir={sclora_dir.name}"
    )

    def _alignment_asr() -> float:
        model_align, tok_a = load_model_and_tokenizer(str(aligned_epoch), device=device)
        asr_a, _ = evaluate_safety(model_align, tok_a, harmful_prompts, device=device)
        return float(asr_a)

    asr_align: float | None = None
    if run_stage in {"baseline", "clora", "safety"} and args.skip_alignment_eval:
        print("[shared_stage2] skip_alignment_eval=True: not computing Stage-1 ASR in this job.")
    elif run_stage == "all" and args.skip_alignment_eval:
        # Defer to eval section so we only run AdvBench on the aligned model once.
        asr_align = None
    else:
        asr_align = _alignment_asr()

    if run_stage in {"all", "baseline"}:
        Trainer(
            {
                "model_name": str(aligned_epoch),
                "mode": "lora",
                "lr": 2e-4,
                "epochs": stage2_epochs,
                "batch_size": 4,
                "max_seq_len": 512,
                "seed": seed,
            }
        ).train(train_dataset=task_ds, save_dir=str(baseline_dir))
        print(f"[shared_stage2] baseline checkpoint: {baseline_dir / f'epoch_{stage2_epochs}'}")

    if run_stage in {"all", "clora"}:
        Trainer(
            {
                "model_name": str(aligned_epoch),
                "mode": "clora_random",
                "rank": 8,
                "alpha": 16,
                "lam": 0.05,
                "lr": 1e-4,
                "epochs": stage2_epochs,
                "batch_size": 4,
                "max_seq_len": 512,
                "seed": seed,
                "loss_diag_every": int(args.loss_diag_every),
            }
        ).train(train_dataset=task_ds, save_dir=str(clora_dir))
        print(f"[shared_stage2] clora checkpoint: {clora_dir / f'epoch_{stage2_epochs}'}")

    if run_stage in {"all", "safety"}:
        Trainer(
            {
                "model_name": str(aligned_epoch),
                "mode": "clora_safety",
                "rank": 8,
                "alpha": 16,
                "lam": 0.05,
                "gamma": safety_gamma,
                "n_safety_prompts": 16,
                "lr": 1e-4,
                "epochs": stage2_epochs,
                "batch_size": 4,
                "max_seq_len": 512,
                "seed": seed,
                "loss_diag_every": int(args.loss_diag_every),
            }
        ).train(
            train_dataset=task_ds,
            aligned_model_name=str(aligned_epoch),
            base_model_name_for_s=base_model_id,
            safety_prompts=_safety_prompts_for_sclora,
            save_dir=str(sclora_dir),
        )
        print(f"[shared_stage2] safety_clora checkpoint: {sclora_dir / f'epoch_{stage2_epochs}'}")

    if run_stage in {"all", "eval_only"}:
        if asr_align is None:
            asr_align = _alignment_asr()
        rows = [("After alignment (shared)", float(asr_align), None)]
        ep_dir = f"epoch_{stage2_epochs}"
        for name, path in [
            ("Baseline LoRA", baseline_dir / ep_dir),
            ("CLoRA (random S)", clora_dir / ep_dir),
            (_safety_row_label(safety_gamma), sclora_dir / ep_dir),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint for {name}: {path}. Train that stage first or run with --stage all."
                )
            model, tok = load_model_and_tokenizer(str(path), device=device)
            asr, acc = _eval_safety_and_task(
                model=model,
                tok=tok,
                harmful_prompts=harmful_prompts,
                gsm8k_test=gsm8k_test,
                device=device,
            )
            rows.append((name, float(asr), float(acc)))

        print("| Method | ASR after FT (↓) | GSM8K Acc (↑) |")
        print("|---|---:|---:|")
        for name, asr, acc in rows:
            if acc is None:
                print(f"| {name} | {asr*100:.1f}% | N/A |")
            else:
                print(f"| {name} | {asr*100:.1f}% | {acc*100:.1f}% |")


if __name__ == "__main__":
    main()

