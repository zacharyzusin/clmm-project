"""
Aggregate multi-seed results and print mean ± std tables.

Usage:
  # 2-task table (reads stage2_shared + stage2_olora JSON files)
  python -m safety_clora.scripts.compute_stats --table 2task \
      results/seeds/stage2_shared_seed*.json results/seeds/stage2_olora_seed*.json

  # Sequential table (reads sequential JSON files for one method)
  python -m safety_clora.scripts.compute_stats --table sequential \
      results/seeds/sequential_safety_clora_seed*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _load(paths: List[str]) -> List[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def _mean_std(vals: List[float]) -> str:
    if not vals:
        return "N/A"
    if len(vals) == 1:
        return f"{vals[0]*100:.1f}%"
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    std = var ** 0.5
    return f"{mu*100:.1f} ± {std*100:.1f}%"


def table_2task(data: List[dict]) -> None:
    # Collect per-method lists of (asr, gsm8k_acc) across seeds.
    method_order = ["after_alignment", "baseline_lora", "clora_random", "safety_clora", "olora", "safety_olora"]
    display = {
        "after_alignment": "After alignment",
        "baseline_lora": "Baseline LoRA",
        "clora_random": "CLoRA (random S)",
        "safety_clora": "Safety-CLoRA (γ=0)",
        "olora": "O-LoRA (λ=0.1)",
        "safety_olora": "Safety-O-LoRA (λ_s=1.0)",
    }
    seeds_seen = sorted({d["seed"] for d in data})
    print(f"Seeds: {seeds_seen}  (n={len(seeds_seen)})\n")

    asr_by_method: Dict[str, List[float]] = defaultdict(list)
    acc_by_method: Dict[str, List[float]] = defaultdict(list)

    for d in data:
        for key in method_order:
            if key in d:
                entry = d[key]
                if "asr" in entry and entry["asr"] == entry["asr"]:  # not nan
                    asr_by_method[key].append(entry["asr"])
                if "gsm8k_acc" in entry and entry["gsm8k_acc"] == entry["gsm8k_acc"]:
                    acc_by_method[key].append(entry["gsm8k_acc"])

    print("| Method | ASR after FT ↓ | GSM8K Acc ↑ |")
    print("|---|---:|---:|")
    for key in method_order:
        if key not in asr_by_method:
            continue
        name = display.get(key, key)
        asr_str = _mean_std(asr_by_method[key])
        acc_str = _mean_std(acc_by_method[key]) if acc_by_method[key] else "—"
        print(f"| {name} | {asr_str} | {acc_str} |")


def table_sequential(data: List[dict]) -> None:
    if not data:
        print("No data.")
        return
    method = data[0]["method"]
    task_order = data[0]["task_order"]
    seeds_seen = sorted({d["seed"] for d in data})
    print(f"Method: {method}  |  Order: {' → '.join(task_order)}  |  Seeds: {seeds_seen}  (n={len(seeds_seen)})\n")

    # Collect per-stage ASR lists.
    stage_labels = [s["label"] for s in data[0]["stages"]]
    asr_by_stage: Dict[str, List[float]] = defaultdict(list)

    for d in data:
        for stage in d["stages"]:
            asr_by_stage[stage["label"]].append(stage["asr"])

    task_header = " | ".join(t.upper() for t in task_order)
    print(f"| Stage | ASR ↓ | {task_header} |")
    print(f"|---|---:|{'|'.join('---:' for _ in task_order)}|")

    for label in stage_labels:
        asr_str = _mean_std(asr_by_stage[label])
        # Task accs: collect per-task across seeds for this stage.
        acc_parts = []
        for task in task_order:
            vals = []
            for d in data:
                for s in d["stages"]:
                    if s["label"] == label and f"{task}_acc" in s:
                        vals.append(s[f"{task}_acc"])
            acc_parts.append(_mean_std(vals) if vals else "—")
        print(f"| {label} | {asr_str} | {' | '.join(acc_parts)} |")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate multi-seed results.")
    ap.add_argument(
        "--table",
        choices=["2task", "sequential"],
        required=True,
        help="Which table to produce.",
    )
    ap.add_argument("files", nargs="+", help="JSON result files (glob-expanded by shell).")
    args = ap.parse_args()

    missing = [f for f in args.files if not Path(f).exists()]
    if missing:
        print(f"ERROR: missing files: {missing}", file=sys.stderr)
        sys.exit(1)

    data = _load(args.files)

    if args.table == "2task":
        table_2task(data)
    else:
        table_sequential(data)


if __name__ == "__main__":
    main()
