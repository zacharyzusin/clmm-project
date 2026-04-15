"""
Generate and save model responses for all 2-task final checkpoints.

Runs inference on the full AdvBench harmful behaviors dataset (~520 prompts)
for each of the five methods plus the Stage-1 aligned checkpoint, and saves
(prompt, response, keyword_refusal) triples to JSON files under results/responses/.

This is a CPU/GPU inference-only job — no LlamaGuard loaded here.
Run before run_llama_guard_eval.py.

Usage:
  python -m safety_clora.scripts.run_generate_responses \\
    --ckpt-root safety_clora/checkpoints \\
    --out-dir results/responses
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from safety_clora.data.data_utils import load_advbench_harmful
from safety_clora.evaluation.safety_eval import evaluate_safety
from safety_clora.utils.model_io import load_model_and_tokenizer

# ---------------------------------------------------------------------------
# Canonical 2-task checkpoint mapping  (relative to ckpt_root)
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINTS = {
    "after_alignment":   "qwen_aligned_shared_seed42_saferlhf_chosen_refusal_n1500_ep3/epoch_3",
    "baseline_lora":     "qwen_lora_gsm8k_shared_seed42_mergefix/epoch_3",
    "clora_random":      "qwen_clora_gsm8k_shared_seed42_mergefix/epoch_3",
    "safety_clora_kl0":  "qwen_safety_clora_gsm8k_shared_seed42_mergefix_kl0/epoch_3",
    "olora_lam0.1":      "qwen_olora_gsm8k_seed42_lam0.1/epoch_3",
    "safety_olora_lams1": "qwen_safety_olora_gsm8k_seed42_lams1/epoch_3",
}

# Human-readable labels for the table
METHOD_LABELS = {
    "after_alignment":    "After alignment",
    "baseline_lora":      "Baseline LoRA",
    "clora_random":       "CLoRA (random S)",
    "safety_clora_kl0":   "Safety-CLoRA (γ=0)",
    "olora_lam0.1":       "O-LoRA (λ=0.1)",
    "safety_olora_lams1": "Safety-O-LoRA (λ_s=1.0)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate and save AdvBench responses for all checkpoints.")
    ap.add_argument(
        "--ckpt-root",
        type=str,
        default=None,
        help="Root directory containing checkpoints (default: safety_clora/checkpoints relative to repo root).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="results/responses",
        help="Directory to write per-method JSON files (default: results/responses).",
    )
    ap.add_argument(
        "--advbench-n",
        type=int,
        default=None,
        help="Number of AdvBench prompts (default: all ~520).",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS.keys()),
        help="Subset of methods to run (default: all six).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max tokens to generate per response (default 200; LlamaGuard needs full response).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    script_pkg = Path(__file__).resolve().parents[1]

    if args.ckpt_root:
        ckpt_root = Path(args.ckpt_root)
    else:
        ckpt_root = script_pkg / "checkpoints"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    harmful_prompts = load_advbench_harmful(n_samples=args.advbench_n)
    print(f"[generate_responses] {len(harmful_prompts)} AdvBench prompts | device={device}", flush=True)

    summary_rows = []

    for method_key in args.methods:
        if method_key not in DEFAULT_CHECKPOINTS:
            print(f"[generate_responses] Unknown method '{method_key}', skipping.", flush=True)
            continue

        rel_path = DEFAULT_CHECKPOINTS[method_key]
        ckpt_path = ckpt_root / rel_path
        out_json = out_dir / f"{method_key}.json"

        label = METHOD_LABELS.get(method_key, method_key)

        if not ckpt_path.exists():
            print(f"[generate_responses] WARNING: checkpoint not found for {label}: {ckpt_path}", flush=True)
            continue

        if out_json.exists():
            print(f"[generate_responses] {label}: responses already saved at {out_json}, skipping.", flush=True)
            with out_json.open() as f:
                data = json.load(f)
            n_refusals = sum(data["keyword_refusals"])
            kw_asr = 1.0 - n_refusals / max(1, len(data["responses"]))
            summary_rows.append((label, kw_asr, len(data["responses"])))
            continue

        print(f"[generate_responses] Loading {label} from {ckpt_path} ...", flush=True)
        model, tok = load_model_and_tokenizer(str(ckpt_path), device=device)
        if torch.cuda.is_available():
            model = model.to(torch.bfloat16)

        asr, _ = evaluate_safety(
            model,
            tok,
            harmful_prompts,
            device=device,
            max_new_tokens=args.max_new_tokens,
            save_path=out_json,
        )

        del model, tok
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"[generate_responses] {label}: keyword ASR={asr*100:.1f}%  saved → {out_json}", flush=True)
        summary_rows.append((label, asr, len(harmful_prompts)))

    # Print summary table
    print("\n--- Keyword ASR Summary ---")
    print(f"{'Method':<30} {'Keyword ASR':>12} {'N prompts':>10}")
    print("-" * 55)
    for label, asr, n in summary_rows:
        print(f"{label:<30} {asr*100:>11.1f}% {n:>10}")
    print(f"\nResponse JSONs written to: {out_dir}")


if __name__ == "__main__":
    main()
