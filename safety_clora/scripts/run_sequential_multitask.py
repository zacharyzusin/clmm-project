"""
Sequential multitask evaluation: T1 (aligned) → T2 → T3 → T4, re-evaluating
AdvBench + all task benchmarks after each capability stage.

Default task order: gsm8k → sst2 → mbpp
Reorder with --task-order, e.g. --task-order gsm8k mbpp sst2

Checkpoint naming: seq_{method}_seed{seed}_t{pos}_{task}
  - Default order preserves existing checkpoint names for warm-start / cache hits.
  - Reordered runs write to new directories (e.g. t3_mbpp vs t3_sst2), so no collision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from safety_clora.data.data_utils import load_advbench_harmful, load_beavertails_harmful, load_agnews
from safety_clora.evaluation.safety_eval import evaluate_safety, evaluate_task_performance
from safety_clora.scripts.run_shared_stage2_comparison import _discover_latest_aligned_checkpoint
from safety_clora.training.trainer import Trainer, load_task_dataset
from safety_clora.utils.model_io import load_model_and_tokenizer

# ---------------------------------------------------------------------------
# Task registry: canonical names → loader kwargs + eval task_type
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "gsm8k":   dict(load_split="train", eval_type="gsm8k"),
    "sst2":    dict(load_split="train", eval_type="sst2"),
    "mbpp":    dict(load_split="train", eval_type="mbpp"),
    "agnews":  dict(load_split="train", eval_type="agnews"),
}

VALID_TASKS = list(TASK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_suite(*, model, tok, harmful_prompts, test_datasets, device):
    asr, _ = evaluate_safety(model, tok, harmful_prompts, device=device)
    results = {"asr": float(asr)}
    for task, ds in test_datasets.items():
        kw = dict(max_new_tokens=256) if task == "mbpp" else {}
        r = evaluate_task_performance(model, tok, ds, task_type=task, device=device, **kw)
        results[f"{task}_acc"] = float(r["accuracy"])
    return results


def _fmt_row(label, r, task_order):
    accs = " | ".join(f"{r.get(f'{t}_acc', 0)*100:.1f}%" for t in task_order)
    return f"| {label} | {r['asr']*100:.1f}% | {accs} |"


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _run_stage(
    *,
    name: str,
    model_init: str,
    train_ds,
    save_dir: Path,
    args,
    aligned: Path,
    bt,
    cap_adapters: list,
    dry_run: bool,
) -> Path:
    final_ep = save_dir / f"epoch_{args.epochs_per_stage}"

    if final_ep.exists():
        print(f"[sequential] {name} checkpoint exists, skipping training -> {final_ep}", flush=True)
        return final_ep

    if dry_run:
        print(f"[dry-run] would train {name} -> {save_dir}", flush=True)
        return final_ep  # path doesn't exist; caller handles dry-run

    save_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "lora":
        cfg = dict(
            model_name=model_init, mode="lora",
            lr=2e-4, epochs=args.epochs_per_stage,
            batch_size=4, max_seq_len=512, seed=args.seed,
        )
        Trainer(cfg).train(train_dataset=train_ds, save_dir=str(save_dir))

    elif args.method == "clora_random":
        cfg = dict(
            model_name=model_init, mode="clora_random",
            rank=8, alpha=16, lam=args.lam,
            lr=1e-4, epochs=args.epochs_per_stage,
            batch_size=4, max_seq_len=512, seed=args.seed,
        )
        Trainer(cfg).train(train_dataset=train_ds, save_dir=str(save_dir))

    elif args.method == "clora_safety":
        cfg = dict(
            model_name=model_init, mode="clora_safety",
            rank=8, alpha=16, lam=args.lam,
            gamma=float(args.safety_gamma),
            n_safety_prompts=16,
            lr=1e-4, epochs=args.epochs_per_stage,
            batch_size=4, max_seq_len=512, seed=args.seed,
        )
        Trainer(cfg).train(
            train_dataset=train_ds,
            aligned_model_name=str(aligned),
            base_model_name_for_s=args.base_model,
            safety_prompts=bt,
            save_dir=str(save_dir),
        )

    else:
        # olora_standard or olora_safety: always start from clean base model;
        # safety adapter extracted from Stage-1 PEFT checkpoint;
        # prior capability adapters passed explicitly.
        cfg = dict(
            model_name=args.base_model, mode=args.method,
            rank=8, alpha=16,
            lam_orth=args.lam_orth, lam_safety=args.lam_safety,
            lr=2e-4, epochs=args.epochs_per_stage,
            batch_size=4, max_seq_len=512, seed=args.seed,
        )
        Trainer(cfg).train(
            train_dataset=train_ds,
            aligned_model_name=str(aligned),
            extra_prev_adapters=list(cap_adapters),
            save_dir=str(save_dir),
        )

    print(f"[sequential] stage {name} -> {final_ep}", flush=True)
    return final_ep


def _load_cap_adapters(ep_path: Path) -> dict:
    pt_file = ep_path / "olora_adapters.pt"
    data = torch.load(str(pt_file), map_location="cpu", weights_only=False)
    return {name: (d["A"], d["B"]) for name, d in data.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sequential multitask safety evaluation with configurable task order."
    )
    ap.add_argument("--aligned-epoch", type=str, default=None,
                    help="Stage-1 (T1) checkpoint directory.")
    ap.add_argument(
        "--method",
        choices=["lora", "clora_random", "clora_safety", "olora_standard", "olora_safety"],
        required=True,
    )
    ap.add_argument(
        "--task-order",
        nargs="+",
        choices=VALID_TASKS,
        default=["gsm8k", "sst2", "mbpp"],
        metavar="TASK",
        help=(
            "Ordered list of capability tasks. Default: gsm8k sst2 mbpp. "
            "Example reorderings: --task-order gsm8k mbpp sst2  "
            "                     --task-order sst2 gsm8k mbpp"
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs-per-stage", type=int, default=3)
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--safety-gamma", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--lam-orth", type=float, default=0.1)
    ap.add_argument("--lam-safety", type=float, default=1.0)
    ap.add_argument("--gsm8k-train-n", type=int, default=1000)
    ap.add_argument("--sst2-train-n", type=int, default=6000)
    ap.add_argument("--mbpp-train-n", type=int, default=200)
    ap.add_argument("--agnews-train-n", type=int, default=1000)
    ap.add_argument("--gsm8k-test-n", type=int, default=None)
    ap.add_argument("--sst2-test-n", type=int, default=None)
    ap.add_argument("--mbpp-test-n", type=int, default=100)
    ap.add_argument("--agnews-test-n", type=int, default=500)
    ap.add_argument("--advbench-n", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without running any training.")
    args = ap.parse_args()

    # Validate: no duplicate tasks.
    if len(args.task_order) != len(set(args.task_order)):
        ap.error("--task-order contains duplicate tasks.")

    root = Path(__file__).resolve().parents[1]
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    if args.aligned_epoch:
        aligned = Path(args.aligned_epoch)
    else:
        aligned = _discover_latest_aligned_checkpoint(ckpt_root, args.seed)
        if aligned is None:
            raise FileNotFoundError("Provide --aligned-epoch.")

    # Print configuration summary.
    order_str = " → ".join(f"T{i+2}:{t}" for i, t in enumerate(args.task_order))
    print(f"[sequential] method={args.method}  order=T1:aligned → {order_str}", flush=True)

    # Dry run: print the stage plan and exit without loading any data.
    if args.dry_run:
        print("[sequential] DRY RUN — no training will occur.", flush=True)
        tag_dr = f"{args.method}_seed{args.seed}"
        print("\n[dry-run] Stage sequence:")
        for pos, task in enumerate(args.task_order, start=2):
            ckpt_dir = ckpt_root / f"seq_{tag_dr}_t{pos}_{task}"
            exists = (ckpt_dir / f"epoch_{args.epochs_per_stage}").exists()
            print(f"  T{pos}: {task:8s}  ->  {ckpt_dir.name}  [{'EXISTS — will skip' if exists else 'would train'}]")
        return

    # Load evaluation datasets.
    adv_n = args.advbench_n if (args.advbench_n is None or args.advbench_n > 0) else None
    harmful = load_advbench_harmful(n_samples=adv_n)

    # Test datasets for all tasks in the order (used for eval after every stage).
    test_n_map = {"gsm8k": args.gsm8k_test_n, "sst2": args.sst2_test_n,
                  "mbpp": args.mbpp_test_n, "agnews": args.agnews_test_n}
    test_split_map = {"gsm8k": "test", "sst2": "validation", "mbpp": "test", "agnews": "test"}
    test_datasets = {
        t: load_task_dataset(t, split=test_split_map[t], n_samples=test_n_map[t])
        for t in args.task_order
    }

    # Train datasets (only load the ones we need, in order).
    train_n_map = {"gsm8k": args.gsm8k_train_n, "sst2": args.sst2_train_n,
                   "mbpp": args.mbpp_train_n, "agnews": args.agnews_train_n}
    train_datasets = {
        t: load_task_dataset(t, split="train", n_samples=train_n_map[t])
        for t in args.task_order
    }

    bt = load_beavertails_harmful(n_samples=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = f"{args.method}_seed{args.seed}"
    is_olora = args.method in {"olora_standard", "olora_safety"}
    cap_adapters_list: list = []

    # Eval T1 (aligned, before any capability training).
    m, t = load_model_and_tokenizer(str(aligned), device=device)
    row0 = _eval_suite(model=m, tok=t, harmful_prompts=harmful,
                       test_datasets=test_datasets, device=device)
    del m, t
    torch.cuda.empty_cache()

    rows = [("T1 aligned (before chain)", row0)]
    ep = str(aligned)

    for pos, task in enumerate(args.task_order, start=2):
        stage_name = f"T{pos}_{task.upper()}"
        ckpt_dir = ckpt_root / f"seq_{tag}_t{pos}_{task}"

        ep_path = _run_stage(
            name=stage_name,
            model_init=ep,
            train_ds=train_datasets[task],
            save_dir=ckpt_dir,
            args=args,
            aligned=aligned,
            bt=bt,
            cap_adapters=list(cap_adapters_list),
            dry_run=args.dry_run,
        )

        if is_olora:
            cap_adapters_list.append(_load_cap_adapters(ep_path))

        m, t = load_model_and_tokenizer(str(ep_path), device=device)
        row = _eval_suite(model=m, tok=t, harmful_prompts=harmful,
                          test_datasets=test_datasets, device=device)
        del m, t
        torch.cuda.empty_cache()

        rows.append((f"After {stage_name}", row))
        ep = str(ep_path)

    # Print results table.
    header_tasks = " | ".join(t.upper() for t in args.task_order)
    print(f"\n| Stage | ASR (↓) | {header_tasks} |")
    sep_cols = "|".join("---:" for _ in args.task_order)
    print(f"|---|---:|{sep_cols}|")
    for label, r in rows:
        print(_fmt_row(label, r, args.task_order))


if __name__ == "__main__":
    main()
