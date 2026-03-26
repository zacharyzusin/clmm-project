from __future__ import annotations

from pathlib import Path

import torch

from safety_clora.data.data_utils import load_advbench_harmful
from safety_clora.evaluation.safety_eval import evaluate_safety, evaluate_task_performance
from safety_clora.training.trainer import Trainer, load_alignment_dataset, load_task_dataset
from safety_clora.utils.model_io import load_model_and_tokenizer


def main():
    root = Path(__file__).resolve().parents[1]
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    cfg_align = {
        "model_name": "Qwen/Qwen3-0.6B",
        "mode": "lora",
        "lr": 2e-4,
        "epochs": 2,
        "batch_size": 4,
        "max_seq_len": 512,
    }

    # Stage 1: Alignment SFT (LoRA baseline mode here; minimal scaffold)
    t1 = Trainer(cfg_align)
    align_ds = load_alignment_dataset(n_samples=500)
    aligned_dir = ckpt_root / "qwen_aligned"
    t1.train(train_dataset=align_ds, save_dir=str(aligned_dir))

    # Stage 2: Capability FT with CLoRA (random S)
    cfg_clora = {
        "model_name": str(aligned_dir / "epoch_2"),
        "mode": "clora_random",
        "rank": 8,
        "alpha": 16,
        "lam": 0.5,
        "lr": 2e-4,
        "epochs": 3,
        "batch_size": 4,
        "max_seq_len": 512,
    }
    t2 = Trainer(cfg_clora)
    task_ds = load_task_dataset("gsm8k", split="train", n_samples=1000)
    clora_dir = ckpt_root / "qwen_clora_gsm8k"
    t2.train(train_dataset=task_ds, save_dir=str(clora_dir))

    # Stage 3: Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    harmful_prompts = load_advbench_harmful(n_samples=64)
    gsm8k_test = load_task_dataset("gsm8k", split="test", n_samples=200)

    rows = []
    for name, path in [
        ("After alignment", aligned_dir / "epoch_2"),
        ("After CLoRA FT", clora_dir / "epoch_3"),
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

