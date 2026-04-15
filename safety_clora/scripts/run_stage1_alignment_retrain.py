from __future__ import annotations

from pathlib import Path
from typing import Sequence

import argparse
import torch

from safety_clora.data.data_utils import load_advbench_harmful, load_beavertails_harmful
from safety_clora.evaluation.safety_eval import evaluate_safety
from safety_clora.data.data_utils import load_alignment_sft_dataset
from safety_clora.training.trainer import Trainer
from safety_clora.utils.model_io import load_model_and_tokenizer


def _print_generations(*, model, tokenizer, prompts: Sequence[str], device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        for p in prompts:
            eval_prompt = f"### Instruction:\n{p}\n\n### Response:\n"
            toks = tokenizer(eval_prompt, return_tensors="pt").to(device)
            in_len = toks["input_ids"].shape[1]
            out_ids = model.generate(**toks, max_new_tokens=80, do_sample=False)
            gen_ids = out_ids[0, in_len:]
            resp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            print("----- prompt -----")
            print(p)
            print("----- response (first 200 chars) -----")
            print(resp[:200])


def main():
    parser = argparse.ArgumentParser(description="Stage-1 alignment SFT with automatic data-source fallback.")
    parser.add_argument("--align-n", type=int, default=1500, help="Target number of alignment SFT examples")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-alignment-examples",
        type=int,
        default=500,
        help="If high-quality SafeRLHF subsets are too small, walk down to broader sources.",
    )
    parser.add_argument(
        "--eval-advbench-n",
        type=int,
        default=64,
        help="AdvBench prompts for ASR (use 0 or -1 for full set ~520).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    base_model_id = "Qwen/Qwen3-0.6B"

    align_n = args.align_n
    epochs = args.epochs
    min_align = args.min_alignment_examples

    # Prefer stronger contrastive refusal-ish SafeRLHF, then strict explicit-refusal,
    # then broader chosen/safer responses, then synthetic refusals.
    align_ds = None
    source = None
    for candidate in ("saferlhf_chosen_refusal", "saferlhf_contrast_refusalish", "saferlhf_chosen"):
        try:
            ds = load_alignment_sft_dataset(source=candidate, n_samples=align_n, split="train")
        except Exception as e:
            print(f"[Stage-1 retrain] skip {candidate}: {e}")
            continue
        if len(ds) >= min_align:
            align_ds = ds
            source = candidate
            break
        print(
            f"[Stage-1 retrain] {candidate} only produced n={len(ds)} (< min_alignment_examples={min_align}); "
            "trying next source."
        )

    if align_ds is None:
        source = "synthetic_refusal"
        align_ds = load_alignment_sft_dataset(source=source, n_samples=align_n, split="train")
        print(f"[Stage-1 retrain] using fallback alignment_source={source} n={len(align_ds)}")

    aligned_dir = ckpt_root / f"qwen_aligned_shared_seed{seed}_{source}_n{align_n}_ep{epochs}"
    print(f"[Stage-1 retrain] alignment_source={source} n={len(align_ds)} epochs={epochs}")
    adv_n = args.eval_advbench_n
    if adv_n is not None and adv_n <= 0:
        adv_n = None
    adv_harmful_prompts = load_advbench_harmful(n_samples=adv_n)
    bt_n = min(256, max(64, len(adv_harmful_prompts)))
    beavertails_harmful_prompts = load_beavertails_harmful(n_samples=bt_n)

    # Train Stage 1 alignment with the current Trainer logic.
    Trainer(
        {
            "model_name": base_model_id,
            "mode": "lora",
            "lr": 2e-4,
            "epochs": epochs,
            "batch_size": 4,
            "max_seq_len": 512,
            "seed": seed,
            "use_chat_template": True,
        }
    ).train(train_dataset=align_ds, save_dir=str(aligned_dir))

    aligned_epoch = aligned_dir / f"epoch_{epochs}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model_and_tokenizer(str(aligned_epoch), device=device)

    asr_adv, _ = evaluate_safety(model, tok, adv_harmful_prompts, device=device)
    asr_bt, _ = evaluate_safety(model, tok, beavertails_harmful_prompts, device=device)

    print(f"[Stage-1 retrain] aligned_epoch={aligned_epoch}")
    print(f"[Stage-1 retrain] ASR on AdvBench harmful: {asr_adv*100:.2f}% (n={len(adv_harmful_prompts)})")
    print(f"[Stage-1 retrain] ASR on BeaverTails harmful: {asr_bt*100:.2f}% (n={len(beavertails_harmful_prompts)})")

    # Quick manual sanity check on a few obviously harmful prompts.
    sanity_prompts = adv_harmful_prompts[:5]
    print("[Stage-1 retrain] Generation sanity-check on first 5 AdvBench harmful prompts")
    _print_generations(model=model, tokenizer=tok, prompts=sanity_prompts, device=device)


if __name__ == "__main__":
    main()

