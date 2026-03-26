from __future__ import annotations

from pathlib import Path

import torch

from safety_clora.data.data_utils import load_beavertails_harmful, load_advbench_harmful
from safety_clora.evaluation.safety_eval import evaluate_safety, evaluate_task_performance
from safety_clora.training.trainer import Trainer, load_alignment_dataset, load_task_dataset
from safety_clora.utils.model_io import load_model_and_tokenizer


def main():
    root = Path(__file__).resolve().parents[1]
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    base_model_id = "Qwen/Qwen3-0.6B"

    # Stage 1: Alignment SFT
    cfg_align = {
        "model_name": base_model_id,
        "mode": "lora",
        "lr": 2e-4,
        "epochs": 2,
        "batch_size": 4,
        "max_seq_len": 512,
    }
    t1 = Trainer(cfg_align)
    align_ds = load_alignment_dataset(n_samples=500)
    aligned_dir = ckpt_root / "qwen_aligned"
    t1.train(train_dataset=align_ds, save_dir=str(aligned_dir))

    aligned_epoch = aligned_dir / "epoch_2"

    # Stage 2: Capability FT with Safety-CLoRA
    cfg_sclora = {
        "model_name": str(aligned_epoch),
        "mode": "clora_safety",
        "rank": 8,
        "alpha": 16,
        "lam": 0.5,
        "gamma": 0.1,
        "n_safety_prompts": 16,
        "lr": 2e-4,
        "epochs": 3,
        "batch_size": 4,
        "max_seq_len": 512,
    }
    t2 = Trainer(cfg_sclora)
    task_ds = load_task_dataset("gsm8k", split="train", n_samples=1000)
    safety_prompts = load_beavertails_harmful(n_samples=64)
    sclora_dir = ckpt_root / "qwen_safety_clora_gsm8k"
    t2.train(
        train_dataset=task_ds,
        aligned_model_name=str(aligned_epoch),
        base_model_name_for_s=base_model_id,
        safety_prompts=safety_prompts,
        save_dir=str(sclora_dir),
    )

    # Stage 3: Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    harmful_prompts = load_advbench_harmful(n_samples=64)
    gsm8k_test = load_task_dataset("gsm8k", split="test", n_samples=200)

    rows = []
    for name, path in [
        ("After alignment", aligned_epoch),
        ("After Safety-CLoRA FT", sclora_dir / "epoch_3"),
    ]:
        model, tok = load_model_and_tokenizer(str(path), device=device)
        asr, _responses = evaluate_safety(model, tok, harmful_prompts, device=device)
        if name == "After alignment":
            rows.append((name, asr, None))
        else:
            task = evaluate_task_performance(model, tok, gsm8k_test, task_type="gsm8k", device=device)
            rows.append((name, asr, task["accuracy"]))

    print("| Checkpoint | ASR (↓ better) | Task Acc (↑ better) |")
    print("|---|---:|---:|")
    for name, asr, acc in rows:
        acc_str = "N/A" if acc is None else f"{acc*100:.1f}%"
        print(f"| {name} | {asr*100:.1f}% | {acc_str} |")


if __name__ == "__main__":
    main()

