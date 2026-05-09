"""
Step-3 λ diagnostic for Llama-2-7b-hf (WildJailbreak aligned, plain text).

Runs 50 gradient steps per λ and reports ratio = (λ * L_reg) / L_task.
Target: ratio in [0.10, 1.00].  λ_target = value closest to 0.50.

λ ranges (per plan):
  CLoRA random S  : {0.01, 0.05, 0.1, 0.5, 1.0}
  Safety-O-LoRA   : λ_safety ∈ {0.1, 0.5, 1.0, 2.0},  λ_cap=0.01 (fixed)

No chat template — Llama-2-7b-hf is a base model, plain text throughout.

Usage:
    python -m safety_clora.scripts.run_lambda_diagnostic_llama2 --task all
    python -m safety_clora.scripts.run_lambda_diagnostic_llama2 --task 1
    python -m safety_clora.scripts.run_lambda_diagnostic_llama2 --task 2
    # Smoke-test (1 step, 1 λ each):
    python -m safety_clora.scripts.run_lambda_diagnostic_llama2 --task all --smoke-test
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from safety_clora.data.data_utils import load_gsm8k
from safety_clora.models.clora import apply_clora_to_model
from safety_clora.models.olora import apply_olora_to_model, extract_peft_lora_adapters, OLoRALinear
from safety_clora.training.losses import clora_regularization_loss
from safety_clora.training.trainer import _make_lm_dataset, _make_collate_fn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_CKPT = (
    REPO_ROOT / "safety_clora" / "checkpoints"
    / "llama2_7b_aligned_wildjailbreak_ep3" / "epoch_3"
)

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

N_STEPS = 50
BATCH_SIZE = 1
MAX_SEQ_LEN = 128
LR = 1e-4
SEED = 42
WINDOW = 25  # average over last WINDOW steps to skip warmup

CLORA_LAMBDAS = [0.01, 0.05, 0.1, 0.5, 1.0]
OLORA_LAM_SAFETY_VALS = [0.1, 0.5, 1.0, 2.0]
LAM_CAP_FIXED = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_base_model_id() -> str:
    return (ALIGNED_CKPT / "BASE_MODEL_ID").read_text().strip()


def _load_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(str(ALIGNED_CKPT), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_aligned_merged(base_model_id: str, device: torch.device):
    """Load Stage-1 PEFT checkpoint and merge adapters into base weights."""
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16
    ).to(device)
    peft_model = PeftModel.from_pretrained(
        base, str(ALIGNED_CKPT / "adapter"), is_trainable=False
    )
    return peft_model.merge_and_unload()


def _load_base(base_model_id: str, device: torch.device):
    return AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16
    ).to(device)


def _build_loader(tokenizer, ds_raw, batch_size: int, max_seq_len: int) -> DataLoader:
    tok_ds = _make_lm_dataset(tokenizer, ds_raw, max_seq_len, use_chat_template=False)
    collate = _make_collate_fn(tokenizer)
    return DataLoader(
        tok_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, drop_last=True,
    )


def _run_steps(model, loader, opt, device, *, collect_reg_fn=None, n_steps: int):
    model.train()
    it = iter(loader)
    records = []
    for _ in range(n_steps):
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
        row = {"l_task": out.loss.item()}
        if collect_reg_fn is not None:
            reg_raw, extras = collect_reg_fn(model)
            row["l_reg_raw"] = float(reg_raw.item() if torch.is_tensor(reg_raw) else reg_raw)
            row.update({k: float(v.item() if torch.is_tensor(v) else v) for k, v in extras.items()})
        out.loss.backward()
        opt.step()
        records.append(row)
    return records


def _run_steps_no_grad(model, loader, device, *, collect_reg_fn=None, n_steps: int):
    """Forward-only variant: no optimizer, no backward — for memory-constrained diagnostics."""
    model.eval()
    it = iter(loader)
    records = []
    with torch.inference_mode():
        for _ in range(n_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            row = {"l_task": out.loss.item()}
            if collect_reg_fn is not None:
                reg_raw, extras = collect_reg_fn(model)
                row["l_reg_raw"] = float(reg_raw.item() if torch.is_tensor(reg_raw) else reg_raw)
                row.update({k: float(v.item() if torch.is_tensor(v) else v) for k, v in extras.items()})
            records.append(row)
    return records


def _window_avg(records: list, key: str, window: int) -> float:
    vals = [r[key] for r in records if key in r]
    subset = vals[-window:] if len(vals) >= window else vals
    return sum(subset) / max(1, len(subset))


def _raw_orth_loss_for_model(model) -> torch.Tensor:
    """Average ||A_safety^T A_curr||_F^2 across OLoRA modules (no λ scaling)."""
    total = None
    n = 0
    for _, module in model.named_modules():
        if not isinstance(module, OLoRALinear) or module.n_prev == 0:
            continue
        A_curr = module.A.to(torch.float32)
        A_safety = module.A_prev_0
        loss = (A_safety.t() @ A_curr).pow(2).sum()
        total = loss if total is None else total + loss
        n += 1
    if total is None:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Task 1: CLoRA λ sweep
# ---------------------------------------------------------------------------

def task1_clora(device: torch.device, n_steps: int):
    print("\n" + "=" * 65)
    print("TASK 1: CLoRA (random S) λ sweep — Llama-2-7b-hf (plain text)")
    print(f"  N_STEPS={n_steps}, avg over last {min(WINDOW, n_steps)} steps")
    print("=" * 65)

    base_model_id = _read_base_model_id()
    tokenizer = _load_tokenizer()
    ds_raw = load_gsm8k("train", n_samples=1000)
    loader = _build_loader(tokenizer, ds_raw, BATCH_SIZE, MAX_SEQ_LEN)

    rows = []
    for lam in CLORA_LAMBDAS:
        print(f"\n[diag] λ={lam} — loading merged aligned model + applying CLoRA random S ...")
        aligned_model = _load_aligned_merged(base_model_id, device)
        model, _ = apply_clora_to_model(aligned_model, rank=8, alpha=16, lam=lam, mode="random")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

        def _reg_fn(m, _lam=lam):
            return clora_regularization_loss(m), {}

        records = _run_steps(model, loader, opt, device, collect_reg_fn=_reg_fn, n_steps=n_steps)
        w = min(WINDOW, n_steps)
        l_task = _window_avg(records, "l_task", w)
        l_reg_raw = _window_avg(records, "l_reg_raw", w)
        l_scaled = lam * l_reg_raw
        ratio = l_scaled / max(l_task, 1e-9)
        rows.append((lam, l_task, l_reg_raw, l_scaled, ratio))
        print(f"  λ={lam:<8} L_task={l_task:.4f}  L_reg={l_reg_raw:.4f}  "
              f"λ*L_reg={l_scaled:.4f}  ratio={ratio:.4f}")

        del model, aligned_model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _print_table1(rows)


def _print_table1(rows):
    print("\n" + "-" * 68)
    print(f"{'λ':<10} {'L_task':>8} {'L_reg':>10} {'λ*L_reg':>10} {'ratio':>8}")
    print("-" * 68)
    for lam, l_task, l_reg_raw, l_scaled, ratio in rows:
        flag = "  ← TARGET" if 0.1 <= ratio <= 1.0 else ("  ← TOO HIGH" if ratio > 1.0 else "  ← too low")
        print(f"{lam:<10} {l_task:>8.4f} {l_reg_raw:>10.4f} {l_scaled:>10.4f} {ratio:>8.4f}{flag}")
    print("-" * 68)
    print("Target range: ratio ∈ [0.10, 1.00]  |  λ_target: ratio closest to 0.50")
    in_range = [(lam, ratio) for lam, _, _, _, ratio in rows if 0.1 <= ratio <= 1.0]
    if in_range:
        best = min(in_range, key=lambda x: abs(x[1] - 0.5))
        print(f"  CLoRA λ_target = {best[0]}  (ratio={best[1]:.4f})")
    else:
        print("  No λ in target range — check values above/below.")


# ---------------------------------------------------------------------------
# Task 2: Safety-O-LoRA λ_safety sweep
# ---------------------------------------------------------------------------

def task2_olora(device: torch.device, n_steps: int):
    print("\n" + "=" * 65)
    print("TASK 2: Safety-O-LoRA λ_safety sweep — Llama-2-7b-hf (plain text)")
    print(f"  λ_cap={LAM_CAP_FIXED} (fixed), N_STEPS={n_steps}, avg over last {min(WINDOW, n_steps)} steps")
    print("=" * 65)

    base_model_id = _read_base_model_id()
    tokenizer = _load_tokenizer()
    ds_raw = load_gsm8k("train", n_samples=1000)
    loader = _build_loader(tokenizer, ds_raw, BATCH_SIZE, MAX_SEQ_LEN)

    # Extract safety adapters once from PEFT checkpoint.
    print(f"[diag] Loading PEFT checkpoint to extract safety adapters: {ALIGNED_CKPT}")
    base_for_extract = _load_base(base_model_id, device)
    peft_model = PeftModel.from_pretrained(
        base_for_extract, str(ALIGNED_CKPT / "adapter"), is_trainable=False
    )
    safety_adapters = extract_peft_lora_adapters(peft_model)
    print(f"[diag] Extracted {len(safety_adapters)} safety adapters.")
    del peft_model, base_for_extract
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    for lam_safety in OLORA_LAM_SAFETY_VALS:
        print(f"\n[diag] λ_safety={lam_safety}, λ_cap={LAM_CAP_FIXED} — loading clean base ...")
        base_model = _load_base(base_model_id, device)
        prev_adapters_by_name = {name: [(A, B)] for name, (A, B) in safety_adapters.items()}
        model, olora_mods = apply_olora_to_model(
            base_model, rank=8, alpha=16,
            prev_adapters_by_name=prev_adapters_by_name,
        )

        def _reg_fn(m, _ls=lam_safety):
            raw = _raw_orth_loss_for_model(m)
            return _ls * raw, {"l_orth_raw": raw}

        records = _run_steps_no_grad(model, loader, device, collect_reg_fn=_reg_fn, n_steps=n_steps)
        w = min(WINDOW, n_steps)
        l_task = _window_avg(records, "l_task", w)
        l_orth_raw = _window_avg(records, "l_orth_raw", w)
        l_scaled = lam_safety * l_orth_raw
        ratio = l_scaled / max(l_task, 1e-9)
        rows.append((lam_safety, l_task, l_orth_raw, l_scaled, ratio))
        print(f"  λ_s={lam_safety:<6} L_task={l_task:.4f}  L_orth_raw={l_orth_raw:.4f}  "
              f"λ_s*L_orth={l_scaled:.4f}  ratio={ratio:.4f}")

        del base_model, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _print_table2(rows)


def _print_table2(rows):
    print("\n" + "-" * 72)
    print(f"{'λ_safety':<12} {'L_task':>8} {'L_orth_raw':>12} {'λ_s*L_orth':>12} {'ratio':>8}")
    print("-" * 72)
    for lam_safety, l_task, l_orth_raw, l_scaled, ratio in rows:
        flag = "  ← TARGET" if 0.1 <= ratio <= 1.0 else ("  ← TOO HIGH" if ratio > 1.0 else "  ← too low")
        print(f"{lam_safety:<12} {l_task:>8.4f} {l_orth_raw:>12.4f} {l_scaled:>12.4f} {ratio:>8.4f}{flag}")
    print("-" * 72)
    print("Target range: ratio ∈ [0.10, 1.00]  |  λ_target: ratio closest to 0.50")
    in_range = [(lam, ratio) for lam, _, _, _, ratio in rows if 0.1 <= ratio <= 1.0]
    if in_range:
        best = min(in_range, key=lambda x: abs(x[1] - 0.5))
        print(f"  Safety-O-LoRA λ_target = {best[0]}  (ratio={best[1]:.4f})")
    else:
        print("  No λ in target range — check values above/below.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1", "2", "all"], default="all")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 1 step with 1 λ per task (fast sanity check before full submit)"
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device={device}  smoke_test={args.smoke_test}")
    print(f"[diag] checkpoint: {ALIGNED_CKPT}")

    if args.smoke_test:
        global CLORA_LAMBDAS, OLORA_LAM_SAFETY_VALS
        CLORA_LAMBDAS = CLORA_LAMBDAS[:1]
        OLORA_LAM_SAFETY_VALS = OLORA_LAM_SAFETY_VALS[:1]

    n_steps = 1 if args.smoke_test else N_STEPS

    if args.task in ("1", "all"):
        task1_clora(device, n_steps)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.task in ("2", "all"):
        task2_olora(device, n_steps)

    print("\n[diag] Done.")


if __name__ == "__main__":
    main()
