"""
Subspace overlap analysis: how much does each capability task's adapter
subspace overlap with the safety adapter subspace, per layer?

For O-LoRA methods: loads A matrices directly from olora_adapters.pt.
For CLoRA/LoRA methods: computes ΔW = merged_weights − base_weights and
  extracts the top-rank left singular vectors as the effective A subspace.

Metric: mean absolute cosine similarity between all rank-direction pairs,
  i.e.  mean|C|  where  C = A_safety_norm.T @ A_task_norm,
  A_norm has unit-norm columns.  Range [0, 1].

Output: CSV + formatted console table.

Usage (no GPU needed):
  python -m safety_clora.scripts.run_subspace_analysis \
    --ckpt-root safety_clora/checkpoints \
    --base-model-cache ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<hash> \
    --out-csv results/subspace_overlap.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import safetensors.torch as st

# ---------------------------------------------------------------------------
# Overlap metric
# ---------------------------------------------------------------------------

def subspace_overlap(A1: torch.Tensor, A2: torch.Tensor) -> float:
    """
    Mean absolute cosine similarity between column pairs of A1 and A2.

    Args:
        A1: (out_features, rank1)
        A2: (out_features, rank2)
    Returns:
        Scalar in [0, 1]. Higher = more overlap.
    """
    A1 = A1.float()
    A2 = A2.float()

    # Normalize columns to unit norm.
    A1_norm = A1 / (A1.norm(dim=0, keepdim=True) + 1e-8)
    A2_norm = A2 / (A2.norm(dim=0, keepdim=True) + 1e-8)

    # C[i, j] = cosine similarity between col i of A1 and col j of A2.
    C = A1_norm.t() @ A2_norm  # (rank1, rank2)
    return float(C.abs().mean().item())


# ---------------------------------------------------------------------------
# Adapter loading helpers
# ---------------------------------------------------------------------------

# Layer name pattern used in all checkpoints: model.layers.{i}.self_attn.{proj}
PROJ_TYPES = ("q_proj", "v_proj")
N_LAYERS = 28


def _layer_names() -> list[str]:
    names = []
    for i in range(N_LAYERS):
        for proj in PROJ_TYPES:
            names.append(f"model.layers.{i}.self_attn.{proj}")
    return names


def load_safety_adapter(peft_ckpt_dir: Path, rank: int = 8) -> Dict[str, torch.Tensor]:
    """
    Load safety adapter A matrices from a PEFT checkpoint directory.

    PEFT naming convention (opposite of intuition):
      lora_A['default'].weight: (rank, in_features)  <- our B matrix
      lora_B['default'].weight: (out_features, rank)  <- our A matrix

    Returns: {layer_name: A (out, rank)}
    """
    adapter_file = peft_ckpt_dir / "adapter" / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Safety adapter not found: {adapter_file}")

    raw = st.load_file(str(adapter_file))
    # PEFT keys look like:
    #   base_model.model.model.layers.{i}.self_attn.{proj}.lora_B.weight -> our A (out, rank)
    adapters: Dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if "lora_B" not in key:
            continue
        # Strip down to canonical layer name.
        # e.g. base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
        #   -> model.layers.0.self_attn.q_proj
        for proj in PROJ_TYPES:
            if f".{proj}.lora_B" in key:
                # Extract layer index.
                parts = key.split(".")
                try:
                    li = parts.index("layers")
                    layer_idx = parts[li + 1]
                except (ValueError, IndexError):
                    continue
                layer_name = f"model.layers.{layer_idx}.self_attn.{proj}"
                adapters[layer_name] = tensor.clone()  # (out, rank)
                break
    if not adapters:
        raise ValueError(f"No lora_B keys found in {adapter_file}")
    return adapters


def load_olora_adapters(olora_pt: Path) -> Dict[str, torch.Tensor]:
    """
    Load A matrices from an olora_adapters.pt file.

    File format: {layer_name: {'A': (out, rank), 'B': (rank, in)}}
    Returns: {layer_name: A (out, rank)}
    """
    data = torch.load(str(olora_pt), map_location="cpu", weights_only=False)
    return {name: entry["A"].clone() for name, entry in data.items()}


def load_delta_w_adapters(
    merged_ckpt_dir: Path,
    base_model_safetensors: Path,
    rank: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Approximate adapter A matrices for merged checkpoints (CLoRA / LoRA) by:
      ΔW = W_merged − W_base
      A_approx = top-rank left singular vectors of ΔW  (shape: out × rank)

    This recovers the column space of the true A matrix (up to rotation) because
    ΔW = A @ B  ⟹  col_span(ΔW) = col_span(A).

    Returns: {layer_name: A_approx (out, rank)}
    """
    merged_file = merged_ckpt_dir / "model.safetensors"
    if not merged_file.exists():
        raise FileNotFoundError(f"Merged weights not found: {merged_file}")

    merged = st.load_file(str(merged_file))
    base = st.load_file(str(base_model_safetensors))

    adapters: Dict[str, torch.Tensor] = {}
    for layer_name in _layer_names():
        key = f"{layer_name}.weight"
        if key not in merged or key not in base:
            continue
        dW = merged[key].float() - base[key].float()  # (out, in)
        if dW.norm() < 1e-9:
            # No update (shouldn't happen for trained adapters, but guard).
            out_dim = dW.shape[0]
            A_approx = torch.zeros(out_dim, rank)
        else:
            U, _, _ = torch.linalg.svd(dW, full_matrices=False)
            A_approx = U[:, :rank].clone()  # (out, rank)
        adapters[layer_name] = A_approx
    return adapters


def load_peft_adapters(peft_ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    """
    Load A matrices from a PEFT adapter checkpoint (baseline LoRA sequential stages).
    Same convention as load_safety_adapter — lora_B.weight is our A.
    """
    return load_safety_adapter(peft_ckpt_dir)


# ---------------------------------------------------------------------------
# Per-method adapter loading
# ---------------------------------------------------------------------------

AdapterSet = Dict[str, Dict[str, torch.Tensor]]  # {task_name: {layer_name: A}}

TASK_NAMES = ["gsm8k", "sst2", "mbpp"]
TASK_TAGS  = ["t2_gsm8k", "t3_sst2", "t4_mbpp"]


def load_adapters_olora(
    ckpt_root: Path, method_tag: str, seq_prefix: str = "seq_"
) -> AdapterSet:
    """Load O-LoRA adapters for all three capability stages."""
    adapters = {}
    for task, stage_tag in zip(TASK_NAMES, TASK_TAGS):
        pt = ckpt_root / f"{seq_prefix}{method_tag}_seed42_{stage_tag}" / "epoch_3" / "olora_adapters.pt"
        if not pt.exists():
            print(f"  [WARN] missing {pt}")
            continue
        adapters[task] = load_olora_adapters(pt)
    return adapters


def load_adapters_delta_w(
    ckpt_root: Path, method_tag: str, base_model_safetensors: Path, rank: int = 8,
    seq_prefix: str = "seq_",
) -> AdapterSet:
    """Load CLoRA/LoRA adapters via ΔW SVD for all three capability stages."""
    adapters = {}
    for task, stage_tag in zip(TASK_NAMES, TASK_TAGS):
        ckpt_dir = ckpt_root / f"{seq_prefix}{method_tag}_seed42_{stage_tag}" / "epoch_3"
        if not ckpt_dir.exists():
            print(f"  [WARN] missing {ckpt_dir}")
            continue
        adapters[task] = load_delta_w_adapters(ckpt_dir, base_model_safetensors, rank)
    return adapters


def load_adapters_peft(ckpt_root: Path, method_tag: str, seq_prefix: str = "seq_") -> AdapterSet:
    """Load baseline LoRA adapters (PEFT format) for all three stages."""
    adapters = {}
    for task, stage_tag in zip(TASK_NAMES, TASK_TAGS):
        ckpt_dir = ckpt_root / f"{seq_prefix}{method_tag}_seed42_{stage_tag}" / "epoch_3"
        if not ckpt_dir.exists():
            print(f"  [WARN] missing {ckpt_dir}")
            continue
        adapters[task] = load_peft_adapters(ckpt_dir)
    return adapters


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

Row = Tuple[str, int, str, str, float]  # method, layer_idx, proj, task, overlap


def compute_overlaps(
    method: str,
    safety_adapters: Dict[str, torch.Tensor],
    task_adapters: AdapterSet,
) -> list[Row]:
    rows = []
    for layer_name in _layer_names():
        layer_idx_str, proj = _parse_layer_name(layer_name)
        if layer_idx_str is None:
            continue
        layer_idx = int(layer_idx_str)

        A_safety = safety_adapters.get(layer_name)
        if A_safety is None:
            continue

        for task in TASK_NAMES:
            if task not in task_adapters:
                continue
            A_task = task_adapters[task].get(layer_name)
            if A_task is None:
                continue
            overlap = subspace_overlap(A_safety, A_task)
            rows.append((method, layer_idx, proj, task, overlap))
    return rows


def _parse_layer_name(name: str) -> Tuple[Optional[str], str]:
    # model.layers.{i}.self_attn.{proj}
    parts = name.split(".")
    try:
        li = parts.index("layers")
        return parts[li + 1], parts[-1]
    except (ValueError, IndexError):
        return None, ""


def print_summary_table(rows: list[Row]) -> None:
    """Print mean overlap per (method, task) averaged over all layers and projections."""
    from collections import defaultdict
    sums: dict = defaultdict(float)
    counts: dict = defaultdict(int)
    for method, layer_idx, proj, task, overlap in rows:
        key = (method, task)
        sums[key] += overlap
        counts[key] += 1

    methods = sorted({r[0] for r in rows})
    print("\n=== Mean Subspace Overlap (safety vs task) — averaged over all layers & projections ===")
    header = f"{'Method':<25} {'GSM8K':>8} {'SST-2':>8} {'MBPP':>8}"
    print(header)
    print("-" * len(header))
    for m in methods:
        vals = [f"{sums[(m, t)] / max(1, counts[(m, t)]):.4f}" if counts[(m, t)] else "  N/A  "
                for t in TASK_NAMES]
        print(f"{m:<25} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

    print("\n=== Per-layer overlap: method=* task=sst2 (layers 0-27, q_proj) ===")
    print(f"{'Layer':>6} " + " ".join(f"{m[:15]:>16}" for m in methods))
    layer_sst2: dict = defaultdict(dict)
    for method, layer_idx, proj, task, overlap in rows:
        if task == "sst2" and proj == "q_proj":
            layer_sst2[layer_idx][method] = overlap
    for li in range(N_LAYERS):
        vals = [f"{layer_sst2[li].get(m, float('nan')):.4f}" for m in methods]
        print(f"{li:>6} " + " ".join(f"{v:>16}" for v in vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt-root",
        type=str,
        default=None,
        help="Path to safety_clora/checkpoints/. Defaults to <script_dir>/../checkpoints.",
    )
    ap.add_argument(
        "--base-model-cache",
        type=str,
        default=None,
        help="Path to Qwen3-0.6B HF snapshot dir (containing model.safetensors).",
    )
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank (default 8).")
    ap.add_argument(
        "--out-csv",
        type=str,
        default="results/subspace_overlap.csv",
        help="Output CSV file path (relative to CWD or absolute).",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["olora_standard", "olora_safety", "clora_random", "clora_safety", "lora"],
        help="Methods to include.",
    )
    ap.add_argument(
        "--safety-ckpt-dir",
        type=str,
        default=None,
        help="Override path to the Stage-1 safety adapter epoch dir. "
             "Defaults to <ckpt-root>/qwen_aligned_shared_seed42_.../epoch_3.",
    )
    ap.add_argument(
        "--seq-prefix",
        type=str,
        default="seq_",
        help="Prefix used in sequential checkpoint directory names (default: 'seq_'). "
             "Use 'seq_llama_3p2_3b_instruct_' for Llama-3.2-3B runs.",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else repo_root / "checkpoints"
    seq_prefix = args.seq_prefix

    # Locate base model safetensors.
    if args.base_model_cache:
        base_st = Path(args.base_model_cache) / "model.safetensors"
    else:
        # Default: look in the user's HF hub cache.
        import os
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
        candidates = list(hf_home.glob("models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors"))
        if not candidates:
            raise FileNotFoundError(
                "Cannot find Qwen3-0.6B model.safetensors in HF cache. "
                "Pass --base-model-cache <path_to_snapshot_dir>."
            )
        base_st = candidates[0]

    print(f"Base model safetensors: {base_st}")
    print(f"Checkpoint root:        {ckpt_root}")
    print(f"Seq prefix:             {seq_prefix}")

    # Safety adapter (Stage 1 PEFT checkpoint).
    if args.safety_ckpt_dir:
        safety_ckpt = Path(args.safety_ckpt_dir)
    else:
        safety_ckpt = ckpt_root / "qwen_aligned_shared_seed42_saferlhf_chosen_refusal_n1500_ep3" / "epoch_3"
    print(f"\nLoading safety adapter from {safety_ckpt} ...")
    safety_adapters = load_safety_adapter(safety_ckpt, rank=args.rank)
    print(f"  Loaded {len(safety_adapters)} layers.")

    all_rows: list[Row] = []

    for method in args.methods:
        print(f"\nLoading task adapters: {method}")
        if method in ("olora_standard", "olora_safety"):
            task_adapters = load_adapters_olora(ckpt_root, method, seq_prefix=seq_prefix)
        elif method == "lora":
            task_adapters = load_adapters_peft(ckpt_root, method, seq_prefix=seq_prefix)
        elif method in ("clora_random", "clora_safety"):
            task_adapters = load_adapters_delta_w(ckpt_root, method, base_st, rank=args.rank, seq_prefix=seq_prefix)
        else:
            print(f"  Unknown method {method}, skipping.")
            continue

        for task, adapters in task_adapters.items():
            print(f"  {task}: {len(adapters)} layers loaded.")

        rows = compute_overlaps(method, safety_adapters, task_adapters)
        all_rows.extend(rows)
        print(f"  Computed {len(rows)} overlap values.")

    # Write CSV.
    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "layer_idx", "proj", "task", "overlap"])
        writer.writerows(all_rows)
    print(f"\nCSV written to {out_csv}  ({len(all_rows)} rows)")

    # Print summary.
    print_summary_table(all_rows)


if __name__ == "__main__":
    main()
