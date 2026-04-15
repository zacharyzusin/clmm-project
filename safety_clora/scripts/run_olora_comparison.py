"""
O-LoRA and Safety-O-LoRA Stage-2 comparison.

Analogous to run_shared_stage2_comparison.py but for the O-LoRA family.
Takes a pinned Stage-1 PEFT checkpoint and trains three Stage-2 variants:
  - baseline LoRA  (reuses existing checkpoint if available)
  - O-LoRA standard  (uniform orthogonality lambda)
  - Safety-O-LoRA    (asymmetric lambda_safety > lambda_cap)

Usage (via SLURM dispatcher):
  sbatch slurm_run_pipeline.sbatch olora_comparison \\
    --aligned-epoch <path/to/epoch_3> \\
    --base-model Qwen/Qwen3-0.6B \\
    --stage2-epochs 3 \\
    --lam-orth 0.1 \\
    --lam-safety 1.0

Or run directly:
  python -m safety_clora.scripts.run_olora_comparison [args]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from safety_clora.data.data_utils import load_advbench_harmful, load_gsm8k
from safety_clora.evaluation.safety_eval import evaluate_safety, evaluate_task_performance
from safety_clora.scripts.run_shared_stage2_comparison import _discover_latest_aligned_checkpoint
from safety_clora.training.trainer import Trainer
from safety_clora.utils.model_io import load_model_and_tokenizer


def _eval_safety_and_task(*, model, tok, harmful_prompts, gsm8k_test, device):
    asr, _ = evaluate_safety(model, tok, harmful_prompts, device=device)
    task = evaluate_task_performance(model, tok, gsm8k_test, task_type="gsm8k", device=device)
    return float(asr), float(task["accuracy"])


def main() -> None:
    ap = argparse.ArgumentParser(description="O-LoRA / Safety-O-LoRA Stage-2 comparison.")
    ap.add_argument(
        "--aligned-epoch",
        type=str,
        default=None,
        help="Path to Stage-1 PEFT checkpoint dir (epoch_k). Auto-discovered if omitted.",
    )
    ap.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID used as the base for O-LoRA (NOT the aligned checkpoint).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage2-epochs", type=int, default=3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument(
        "--lam-orth",
        type=float,
        default=0.1,
        help="Orthogonality lambda for O-LoRA standard (uniform across all prev adapters).",
    )
    ap.add_argument(
        "--lam-safety",
        type=float,
        default=1.0,
        help="Orthogonality lambda against the safety adapter for Safety-O-LoRA (should be > lam-orth).",
    )
    ap.add_argument(
        "--gsm8k-train-n",
        type=int,
        default=1000,
        help="Number of GSM8K training examples.",
    )
    ap.add_argument("--advbench-n", type=int, default=None, help="Cap on AdvBench prompts (default: all ~520).")
    ap.add_argument("--gsm8k-test-n", type=int, default=None, help="Cap on GSM8K test examples (default: all).")
    ap.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "baseline", "olora", "safety_olora", "eval_only"],
        help="Which part to run.",
    )
    ap.add_argument(
        "--skip-alignment-eval",
        action="store_true",
        help="Skip ASR eval on Stage-1 checkpoint (saves time in split jobs).",
    )
    ap.add_argument("--loss-diag-every", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    stage2_epochs = args.stage2_epochs

    # --- Stage-1 checkpoint ---
    if args.aligned_epoch:
        aligned_epoch = Path(args.aligned_epoch)
    else:
        aligned_epoch = _discover_latest_aligned_checkpoint(ckpt_root, seed)
        if aligned_epoch is None:
            raise FileNotFoundError(
                "Could not auto-discover Stage-1 checkpoint. Use --aligned-epoch."
            )

    if not aligned_epoch.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {aligned_epoch}")

    # Verify it's a PEFT checkpoint (required for O-LoRA adapter extraction).
    ckpt_type_file = aligned_epoch / "CHECKPOINT_TYPE"
    if ckpt_type_file.exists():
        ckpt_type = ckpt_type_file.read_text(encoding="utf-8").strip()
        if ckpt_type != "peft_lora_adapter":
            raise RuntimeError(
                f"O-LoRA requires a peft_lora_adapter checkpoint, but {aligned_epoch} has "
                f"CHECKPOINT_TYPE={ckpt_type!r}. Provide a Stage-1 checkpoint saved with mode='lora'."
            )

    # --- Shared datasets ---
    adv_n = args.advbench_n if (args.advbench_n is not None and args.advbench_n > 0) else None
    gsm_test_n = args.gsm8k_test_n if (args.gsm8k_test_n is not None and args.gsm8k_test_n > 0) else None

    harmful_prompts = load_advbench_harmful(n_samples=adv_n)
    task_train = load_gsm8k(split="train", n_samples=args.gsm8k_train_n)
    gsm8k_test = load_gsm8k(split="test", n_samples=gsm_test_n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Checkpoint directories ---
    # Baseline LoRA reuses the same dir as run_shared_stage2_comparison.py so we don't retrain.
    baseline_dir = ckpt_root / "qwen_lora_gsm8k_shared_seed42_mergefix"
    olora_dir = ckpt_root / f"qwen_olora_gsm8k_seed{seed}_lam{args.lam_orth:g}"
    safety_olora_dir = ckpt_root / f"qwen_safety_olora_gsm8k_seed{seed}_lams{args.lam_safety:g}"

    run_stage = args.stage

    print(
        f"[olora_comparison] stage={run_stage} stage2_epochs={stage2_epochs} "
        f"lam_orth={args.lam_orth} lam_safety={args.lam_safety} "
        f"aligned_epoch={aligned_epoch} "
        f"advbench_n={len(harmful_prompts)} gsm8k_test_n={len(gsm8k_test)}",
        flush=True,
    )

    # --- Stage-1 ASR ---
    def _alignment_asr() -> float:
        m, t = load_model_and_tokenizer(str(aligned_epoch), device=device)
        asr, _ = evaluate_safety(m, t, harmful_prompts, device=device)
        del m, t
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return float(asr)

    asr_align: float | None = None
    if not args.skip_alignment_eval and run_stage not in {"baseline", "olora", "safety_olora"}:
        asr_align = _alignment_asr()

    # --- Baseline LoRA ---
    # Re-uses existing checkpoint from run_shared_stage2_comparison.py if it exists,
    # otherwise trains from the Stage-1 checkpoint (using it as base, merged).
    if run_stage in {"all", "baseline"}:
        if not (baseline_dir / f"epoch_{stage2_epochs}").exists():
            print("[olora_comparison] baseline checkpoint missing — training baseline LoRA", flush=True)
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
            ).train(train_dataset=task_train, save_dir=str(baseline_dir))
        else:
            print(f"[olora_comparison] reusing existing baseline checkpoint: {baseline_dir}", flush=True)

    # --- O-LoRA standard ---
    if run_stage in {"all", "olora"}:
        Trainer(
            {
                "model_name": args.base_model,      # Qwen3-0.6B base weights
                "mode": "olora_standard",
                "rank": args.rank,
                "alpha": args.alpha,
                "lam_orth": args.lam_orth,
                "lr": 2e-4,
                "epochs": stage2_epochs,
                "batch_size": 4,
                "max_seq_len": 512,
                "seed": seed,
                "loss_diag_every": args.loss_diag_every,
            }
        ).train(
            train_dataset=task_train,
            aligned_model_name=str(aligned_epoch),  # Stage-1 PEFT checkpoint for adapter extraction
            save_dir=str(olora_dir),
        )
        print(f"[olora_comparison] O-LoRA checkpoint: {olora_dir / f'epoch_{stage2_epochs}'}", flush=True)

    # --- Safety-O-LoRA ---
    if run_stage in {"all", "safety_olora"}:
        Trainer(
            {
                "model_name": args.base_model,
                "mode": "olora_safety",
                "rank": args.rank,
                "alpha": args.alpha,
                "lam_safety": args.lam_safety,
                "lr": 2e-4,
                "epochs": stage2_epochs,
                "batch_size": 4,
                "max_seq_len": 512,
                "seed": seed,
                "loss_diag_every": args.loss_diag_every,
            }
        ).train(
            train_dataset=task_train,
            aligned_model_name=str(aligned_epoch),
            save_dir=str(safety_olora_dir),
        )
        print(
            f"[olora_comparison] Safety-O-LoRA checkpoint: {safety_olora_dir / f'epoch_{stage2_epochs}'}",
            flush=True,
        )

    # --- Evaluation ---
    if run_stage in {"all", "eval_only"}:
        if asr_align is None:
            asr_align = _alignment_asr()

        rows = [("After alignment", float(asr_align), None)]
        ep_dir = f"epoch_{stage2_epochs}"

        for name, path in [
            ("Baseline LoRA", baseline_dir / ep_dir),
            (f"O-LoRA (λ={args.lam_orth:g})", olora_dir / ep_dir),
            (f"Safety-O-LoRA (λ_s={args.lam_safety:g})", safety_olora_dir / ep_dir),
        ]:
            if not path.exists():
                print(f"[olora_comparison] WARNING: checkpoint missing for {name}: {path}", flush=True)
                rows.append((name, float("nan"), float("nan")))
                continue
            model, tok = load_model_and_tokenizer(str(path), device=device)
            asr, acc = _eval_safety_and_task(
                model=model,
                tok=tok,
                harmful_prompts=harmful_prompts,
                gsm8k_test=gsm8k_test,
                device=device,
            )
            del model, tok
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rows.append((name, asr, acc))

        print("\n| Method | ASR after FT (↓) | GSM8K Acc (↑) |")
        print("|---|---:|---:|")
        for name, asr, acc in rows:
            if acc is None:
                print(f"| {name} | {asr*100:.1f}% | N/A |")
            elif asr != asr:  # nan
                print(f"| {name} | N/A | N/A |")
            else:
                print(f"| {name} | {asr*100:.1f}% | {acc*100:.1f}% |")


if __name__ == "__main__":
    main()
