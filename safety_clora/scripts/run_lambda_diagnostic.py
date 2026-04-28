"""
Tasks 1 & 2: 50-step λ diagnostic for CLoRA (random S) and Safety-O-LoRA on Llama.

Runs 50 gradient steps per λ value and reports the ratio
  (λ * L_reg) / L_task
to find the λ range where regularization is meaningful (ratio 0.1–1.0) but not crushing.

Usage:
    python -m safety_clora.scripts.run_lambda_diagnostic --task 1   # CLoRA
    python -m safety_clora.scripts.run_lambda_diagnostic --task 2   # Safety-O-LoRA
    python -m safety_clora.scripts.run_lambda_diagnostic --task all  # both
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from safety_clora.data.data_utils import load_gsm8k
from safety_clora.models.clora import apply_clora_to_model
from safety_clora.models.olora import apply_olora_to_model, extract_peft_lora_adapters
from safety_clora.training.losses import clora_regularization_loss
from safety_clora.models.olora import olora_orth_loss_for_model, OLoRALinear

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Script lives at safety_clora/scripts/run_lambda_diagnostic.py
# parents[0] = safety_clora/scripts, parents[1] = safety_clora, parents[2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO_ROOT / "safety_clora" / "checkpoints"
ALIGNED_CKPT = (
    CKPT_ROOT
    / "llama_3p2_3b_instruct_aligned_seed42_saferlhf_chosen_refusal_n1500_ep3"
    / "epoch_3"
)
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

N_STEPS = 50
BATCH_SIZE = 4
MAX_SEQ_LEN = 512
LR = 1e-4
SEED = 42
# Average losses over the last WINDOW steps (skip warmup where B=0 → reg≈0).
WINDOW = 25

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _make_lm_dataset(tokenizer, ds, max_seq_len: int):
    """Tokenize with chat template (mirrors trainer.py logic)."""
    def _tok(ex):
        prompt = ex["input"]
        answer = ex["output"]
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + (answer or "")
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_len,
                               add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_len,
                             add_special_tokens=False)["input_ids"]
        eos = tokenizer.eos_token_id
        if eos is not None and (len(full_ids) == 0 or full_ids[-1] != eos):
            full_ids = (full_ids + [eos])[:max_seq_len]
        input_ids = full_ids[:max_seq_len]
        labels = ([-100] * min(len(prompt_ids), max_seq_len)) + full_ids[len(prompt_ids):]
        labels = labels[:max_seq_len]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return ds.map(_tok, remove_columns=ds.column_names)


def _pad_collate(batch, pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        attention_mask.append(x["attention_mask"] + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _build_loader(tokenizer, ds, batch_size: int, max_seq_len: int):
    tok_ds = _make_lm_dataset(tokenizer, ds, max_seq_len)
    tok_ds = tok_ds.with_format("torch")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return DataLoader(
        tok_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _pad_collate(b, pad_id),
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_base_and_tokenizer(device: torch.device):
    """Load clean Llama-3.2-3B-Instruct base weights."""
    print(f"[diag] Loading base model: {BASE_MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(ALIGNED_CKPT, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID).to(device)
    return model, tok


def _load_aligned_merged(device: torch.device):
    """Load Stage-1 PEFT checkpoint, merge adapters → clean HF model."""
    print(f"[diag] Loading aligned PEFT checkpoint: {ALIGNED_CKPT}")
    tok = AutoTokenizer.from_pretrained(ALIGNED_CKPT, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_id_file = ALIGNED_CKPT / "BASE_MODEL_ID"
    base_id = base_id_file.read_text().strip()
    base = AutoModelForCausalLM.from_pretrained(base_id).to(device)
    peft_model = PeftModel.from_pretrained(base, ALIGNED_CKPT / "adapter", is_trainable=False)
    merged = peft_model.merge_and_unload()
    return merged, tok


# ---------------------------------------------------------------------------
# 50-step training loop
# ---------------------------------------------------------------------------

def _run_50_steps(model, loader, opt, device, *, collect_reg_fn=None):
    """
    Run exactly N_STEPS gradient steps.

    collect_reg_fn: callable(model) -> (raw_reg_loss, label_dict)
        Should return the raw regularization loss (BEFORE λ scaling) and any
        extra per-component scalars to track (as {str: float}).

    Returns list of dicts: one per step with keys 'l_task', 'l_reg_raw', extras...
    """
    model.train()
    it = iter(loader)
    records = []
    for step in range(N_STEPS):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        l_task = out.loss

        row = {"l_task": l_task.item()}

        if collect_reg_fn is not None:
            reg_raw, extras = collect_reg_fn(model)
            row["l_reg_raw"] = reg_raw.item() if torch.is_tensor(reg_raw) else float(reg_raw)
            row.update({k: v.item() if torch.is_tensor(v) else float(v) for k, v in extras.items()})

        # Backward through task loss only; reg is tracked but NOT added here —
        # we want to observe the natural task-loss magnitude independently.
        # The ratio check uses the scaled reg that WOULD be added.
        l_task.backward()
        opt.step()

        records.append(row)

    return records


def _window_avg(records: list, key: str) -> float:
    vals = [r[key] for r in records if key in r]
    window = vals[-WINDOW:] if len(vals) >= WINDOW else vals
    return sum(window) / max(1, len(window))


# ---------------------------------------------------------------------------
# Task 1: CLoRA λ sweep
# ---------------------------------------------------------------------------

CLORA_LAMBDAS = [0.0001, 0.001, 0.01, 0.1, 0.5]


def task1_clora(device: torch.device):
    print("\n" + "=" * 60)
    print("TASK 1: CLoRA (random S) λ sweep — Llama-3.2-3B-Instruct")
    print(f"  N_STEPS={N_STEPS}, avg over last {WINDOW} steps")
    print("=" * 60)

    # Load data once.
    ds_raw = load_gsm8k("train", n_samples=1000)
    # Load aligned merged model and tokenizer once; we'll re-initialize CLoRA for each λ.
    aligned_model, tok = _load_aligned_merged(device)
    loader = _build_loader(tok, ds_raw, BATCH_SIZE, MAX_SEQ_LEN)

    # Clone state dict so we can reset for each λ.
    import copy
    base_state = copy.deepcopy(aligned_model.state_dict())

    rows = []
    for lam in CLORA_LAMBDAS:
        print(f"\n[diag] λ={lam} — applying CLoRA random S ...")
        # Reset base model weights.
        aligned_model.load_state_dict(base_state)

        # Apply CLoRA random S to the reset model.
        model, _ = apply_clora_to_model(
            aligned_model, rank=8, alpha=16, lam=lam, mode="random"
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=LR)

        def _reg_fn(m, _lam=lam):
            raw = clora_regularization_loss(m)  # averaged across modules, no λ
            return raw, {}

        records = _run_50_steps(model, loader, opt, device, collect_reg_fn=_reg_fn)

        l_task = _window_avg(records, "l_task")
        l_reg_raw = _window_avg(records, "l_reg_raw")
        l_scaled = lam * l_reg_raw
        ratio = l_scaled / max(l_task, 1e-9)
        rows.append((lam, l_task, l_reg_raw, l_scaled, ratio))
        print(f"  λ={lam:<8}  L_task={l_task:.4f}  L_reg={l_reg_raw:.4f}  "
              f"λ*L_reg={l_scaled:.4f}  ratio={ratio:.4f}")

    print("\n" + "-" * 65)
    print(f"{'λ':<10} {'L_task':>8} {'L_reg':>10} {'λ*L_reg':>10} {'ratio':>8}")
    print("-" * 65)
    for lam, l_task, l_reg_raw, l_scaled, ratio in rows:
        flag = "  ← TARGET" if 0.1 <= ratio <= 1.0 else ("  ← TOO HIGH" if ratio > 1.0 else "  ← too low")
        print(f"{lam:<10} {l_task:>8.4f} {l_reg_raw:>10.4f} {l_scaled:>10.4f} {ratio:>8.4f}{flag}")
    print("-" * 65)
    print("Target: ratio in [0.10, 1.00]")


# ---------------------------------------------------------------------------
# Task 2: Safety-O-LoRA λ_safety sweep
# ---------------------------------------------------------------------------

OLORA_LAM_SAFETY_VALS = [0.001, 0.01, 0.1, 0.5, 1.0]
LAM_CAP_FIXED = 0.01  # reduced from prior 0.1


def task2_olora(device: torch.device):
    print("\n" + "=" * 60)
    print("TASK 2: Safety-O-LoRA λ_safety sweep — Llama-3.2-3B-Instruct")
    print(f"  λ_cap={LAM_CAP_FIXED} (fixed), N_STEPS={N_STEPS}, avg over last {WINDOW} steps")
    print("=" * 60)

    # Load data once.
    ds_raw = load_gsm8k("train", n_samples=1000)

    # Load Stage-1 PEFT model to extract safety adapters.
    print(f"[diag] Loading PEFT model for safety adapter extraction: {ALIGNED_CKPT}")
    base_id_file = ALIGNED_CKPT / "BASE_MODEL_ID"
    base_id = base_id_file.read_text().strip()
    tok = AutoTokenizer.from_pretrained(ALIGNED_CKPT, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_for_extract = AutoModelForCausalLM.from_pretrained(base_id).to(device)
    peft_model = PeftModel.from_pretrained(
        base_for_extract, ALIGNED_CKPT / "adapter", is_trainable=False
    )
    safety_adapters = extract_peft_lora_adapters(peft_model)
    print(f"[diag] Extracted {len(safety_adapters)} safety adapters.")
    del peft_model, base_for_extract

    loader = _build_loader(tok, ds_raw, BATCH_SIZE, MAX_SEQ_LEN)

    rows = []
    for lam_safety in OLORA_LAM_SAFETY_VALS:
        print(f"\n[diag] λ_safety={lam_safety}, λ_cap={LAM_CAP_FIXED} — applying Safety-O-LoRA ...")
        # Reload clean base for each run (O-LoRA starts from base, not merged aligned).
        base_model = AutoModelForCausalLM.from_pretrained(base_id).to(device)

        # Build prev_adapters_by_name: safety adapter = adapter index 0.
        prev_adapters_by_name = {
            name: [(A, B)] for name, (A, B) in safety_adapters.items()
        }
        model, olora_mods = apply_olora_to_model(
            base_model, rank=8, alpha=16,
            prev_adapters_by_name=prev_adapters_by_name,
        )

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=LR)

        # lam_list per layer: [lam_safety] (only one prior = safety adapter).
        lam_list_by_name = {
            name: [lam_safety]
            for name, _ in olora_mods
        }

        def _reg_fn(m, _lam_list_by_name=lam_list_by_name, _lam_s=lam_safety):
            # Compute raw orth loss (without λ) per layer, then average.
            raw = _raw_orth_loss_for_model(m)
            scaled = _lam_s * raw
            return scaled, {"l_orth_safety_raw": raw}

        records = _run_50_steps(model, loader, opt, device, collect_reg_fn=_reg_fn)

        l_task = _window_avg(records, "l_task")
        # l_reg_raw here is actually the λ-scaled reg (we pass scaled as raw to keep interface consistent)
        l_orth_safety_raw = _window_avg(records, "l_orth_safety_raw")
        l_scaled = lam_safety * l_orth_safety_raw
        ratio = l_scaled / max(l_task, 1e-9)
        rows.append((lam_safety, l_task, l_orth_safety_raw, l_scaled, ratio))
        print(f"  λ_s={lam_safety:<7}  L_task={l_task:.4f}  L_orth_raw={l_orth_safety_raw:.4f}  "
              f"λ_s*L_orth={l_scaled:.4f}  ratio={ratio:.4f}")

        del base_model, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "-" * 72)
    print(f"{'λ_safety':<12} {'L_task':>8} {'L_orth_raw':>12} {'λ_s*L_orth':>12} {'ratio':>8}")
    print("-" * 72)
    for lam_safety, l_task, l_orth_raw, l_scaled, ratio in rows:
        flag = "  ← TARGET" if 0.1 <= ratio <= 1.0 else ("  ← TOO HIGH" if ratio > 1.0 else "  ← too low")
        print(f"{lam_safety:<12} {l_task:>8.4f} {l_orth_raw:>12.4f} {l_scaled:>12.4f} {ratio:>8.4f}{flag}")
    print("-" * 72)
    print("Target: ratio in [0.10, 1.00]")


def _raw_orth_loss_for_model(model) -> torch.Tensor:
    """Average ||A_safety^T A_curr||_F^2 across modules, WITHOUT λ scaling."""
    total = None
    n = 0
    for _, module in model.named_modules():
        if not isinstance(module, OLoRALinear) or module.n_prev == 0:
            continue
        A_curr = module.A.to(torch.float32)
        A_safety = module.A_prev_0
        gram = A_safety.t() @ A_curr
        loss = gram.pow(2).sum()
        total = loss if total is None else total + loss
        n += 1
    if total is None:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["1", "2", "all"],
        default="all",
        help="Which diagnostic to run: 1=CLoRA, 2=Safety-O-LoRA, all=both",
    )
    args = parser.parse_args()

    # Set seeds for reproducibility.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device={device}")

    if args.task in ("1", "all"):
        task1_clora(device)

    if args.task in ("2", "all"):
        task2_olora(device)

    print("\n[diag] Done.")


if __name__ == "__main__":
    main()
