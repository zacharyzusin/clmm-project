"""
Task 3: Full 2-task λ sweep on Llama-3.2-3B-Instruct.

One job per (method, λ) pair. Trains Stage 2 (align → GSM8K, 3 epochs, seed 42)
and records ASR, GSM8K accuracy, and the final-epoch (λ * L_reg) / L_task ratio.

Usage:
  python -m safety_clora.scripts.run_llama_lambdasweep \\
    --method clora_random --lam 0.1

  python -m safety_clora.scripts.run_llama_lambdasweep \\
    --method olora_safety --lam-safety 0.5 --lam-cap 0.01

Methods: clora_random | clora_safety | olora_standard | olora_safety
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

from safety_clora.data.data_utils import load_advbench_harmful, load_beavertails_harmful, load_gsm8k
from safety_clora.evaluation.safety_eval import evaluate_safety, evaluate_task_performance
from safety_clora.models.clora import apply_clora_to_model, merge_clora_to_base_linear
from safety_clora.models.olora import (
    apply_olora_to_model,
    extract_peft_lora_adapters,
    merge_olora_to_base_linear,
    olora_orth_loss_for_model,
    OLoRALinear,
)
from safety_clora.training.losses import clora_regularization_loss
from safety_clora.training.trainer import (
    _make_lm_dataset,
    _make_collate_fn,
    _maybe_merge_peft,
    _freeze_all_but_clora,
    _freeze_all_but_olora,
    _save_checkpoint,
    _set_seed,
)
from safety_clora.utils.model_io import load_model_and_tokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent          # clmm-project/
_CKPT_ROOT = _REPO_ROOT / "safety_clora" / "checkpoints"
_RESULTS_DIR = _REPO_ROOT / "results"

_ALIGNED_CKPT = str(
    _CKPT_ROOT
    / "llama_3p2_3b_instruct_aligned_seed42_saferlhf_chosen_refusal_n1500_ep3"
    / "epoch_3"
)
_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# ---------------------------------------------------------------------------
# Hyper-parameters (fixed)
# ---------------------------------------------------------------------------

RANK = 8
ALPHA = 16
LR_CLORA = 1e-4
LR_OLORA = 2e-4
LR_LORA = 2e-4
EPOCHS = 3
BATCH_SIZE = 4
MAX_SEQ_LEN = 512
SEED = 42
N_TRAIN = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lam_tag(lam: float) -> str:
    """Compact string safe for filenames: 0.1 → '0p1', 0.05 → '0p05'."""
    return f"{lam:g}".replace(".", "p")


def _extract_safety_adapters(aligned_ckpt: str, device: torch.device) -> Dict:
    model, _ = load_model_and_tokenizer(aligned_ckpt, device=device, trainable=False)
    if not isinstance(model, PeftModel):
        raise RuntimeError(f"Expected PeftModel at {aligned_ckpt}, got {type(model).__name__}")
    adapters = extract_peft_lora_adapters(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return adapters


# ---------------------------------------------------------------------------
# Core training loop — returns final_ratio
# ---------------------------------------------------------------------------

def train_and_eval(
    *,
    method: str,
    lam: float,
    lam_cap: float,
    device: torch.device,
    save_dir: Path,
) -> Dict:
    """
    Trains one (method, λ) run for EPOCHS epochs on GSM8K.

    Returns dict with keys: asr, gsm8k_acc, final_ratio, lam, method.
    final_ratio = mean(λ * L_reg / L_task) over the last epoch's steps.
    """
    _set_seed(SEED)

    # ---- Data ----
    task_train = load_gsm8k("train", n_samples=N_TRAIN)
    harmful_prompts = load_advbench_harmful(n_samples=None)
    gsm8k_test = load_gsm8k("test", n_samples=None)

    # ---- Model + tokenizer ----
    if method in ("clora_random", "clora_safety"):
        # CLoRA: start from merged aligned checkpoint
        model, tokenizer = load_model_and_tokenizer(_ALIGNED_CKPT, device=device, trainable=True)
        if device.type == "cuda":
            model = model.to(torch.bfloat16)
        model.gradient_checkpointing_enable()
        model = _maybe_merge_peft(model)

        base_model_ref = None
        aligned_model_ref = None
        if method == "clora_safety":
            # Need base + aligned references for S-matrix construction
            aligned_model_ref, _ = load_model_and_tokenizer(_ALIGNED_CKPT, device=device, trainable=False)
            aligned_model_ref = _maybe_merge_peft(aligned_model_ref)
            aligned_model_ref.eval()
            for p in aligned_model_ref.parameters():
                p.requires_grad_(False)
            base_model_ref = AutoModelForCausalLM.from_pretrained(
                _BASE_MODEL, torch_dtype=torch.float32
            ).to(device)
            base_model_ref.eval()
            for p in base_model_ref.parameters():
                p.requires_grad_(False)

        model, _ = apply_clora_to_model(
            model, rank=RANK, alpha=ALPHA, lam=lam,
            mode="safety" if method == "clora_safety" else "random",
            base_model=base_model_ref,
            aligned_model=aligned_model_ref,
        )
        _freeze_all_but_clora(model)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR_CLORA
        )

    elif method in ("olora_standard", "olora_safety"):
        # O-LoRA: start from clean base, inject safety adapters as frozen priors
        model, tokenizer = load_model_and_tokenizer(_BASE_MODEL, device=device, trainable=True)
        if device.type == "cuda":
            model = model.to(torch.bfloat16)
        model.gradient_checkpointing_enable()

        safety_adapters = _extract_safety_adapters(_ALIGNED_CKPT, device)
        print(f"[sweep] extracted {len(safety_adapters)} safety adapter layers", flush=True)

        prev_adapters_by_name = {name: [(A, B)] for name, (A, B) in safety_adapters.items()}

        # Build lam_list: for olora_standard → [lam], for olora_safety → [lam_safety].
        # lam_cap is used only when cap adapters exist (sequential); in 2-task there are none.
        effective_lam = lam  # lam is lam_safety for safety, lam_orth for standard
        olora_lam_list = {name: [effective_lam] for name in safety_adapters}

        model, _ = apply_olora_to_model(
            model, rank=RANK, alpha=ALPHA,
            prev_adapters_by_name=prev_adapters_by_name,
        )
        _freeze_all_but_olora(model)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR_OLORA
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    # ---- Tokenize ----
    tokenized = _make_lm_dataset(tokenizer, task_train, max_seq_len=MAX_SEQ_LEN, use_chat_template=True)
    dl = DataLoader(
        tokenized, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=_make_collate_fn(tokenizer),
    )

    # ---- Train ----
    # Track last-epoch ratio to measure whether regularization stayed active.
    last_epoch_task_losses: List[float] = []
    last_epoch_reg_penalties: List[float] = []  # λ * L_reg (already scaled)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        track_this_epoch = (epoch == EPOCHS)
        if track_this_epoch:
            last_epoch_task_losses.clear()
            last_epoch_reg_penalties.clear()

        pbar = tqdm(dl, desc=f"epoch {epoch}/{EPOCHS} [{method} λ={lam:g}]", leave=True)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            task_loss = out.loss

            if method in ("clora_random", "clora_safety"):
                raw_reg = clora_regularization_loss(model)
                penalty = lam * raw_reg
                total_loss = task_loss + penalty
            elif method in ("olora_standard", "olora_safety"):
                # olora_orth_loss_for_model returns λ-scaled loss (λ already in lam_list).
                penalty = olora_orth_loss_for_model(model, olora_lam_list)
                total_loss = task_loss + penalty

            total_loss.backward()
            optimizer.step()

            if track_this_epoch:
                last_epoch_task_losses.append(float(task_loss.detach()))
                last_epoch_reg_penalties.append(float(penalty.detach()))

            pbar.set_postfix(
                task=f"{float(task_loss.detach()):.4f}",
                penalty=f"{float(penalty.detach()):.4f}",
            )

        # Save checkpoint each epoch; merge only at final epoch.
        ckpt_dir = save_dir / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if epoch == EPOCHS:
            if method in ("clora_random", "clora_safety"):
                model = merge_clora_to_base_linear(model)
            elif method in ("olora_standard", "olora_safety"):
                from safety_clora.training.trainer import _save_olora_adapters
                _save_olora_adapters(model, ckpt_dir)
                model = merge_olora_to_base_linear(model)
        _save_checkpoint(model, tokenizer, ckpt_dir)

    # ---- Final ratio ----
    mean_task = sum(last_epoch_task_losses) / max(1, len(last_epoch_task_losses))
    mean_penalty = sum(last_epoch_reg_penalties) / max(1, len(last_epoch_reg_penalties))
    final_ratio = mean_penalty / max(mean_task, 1e-9)

    print(
        f"[sweep] final-epoch avg: L_task={mean_task:.4f}  λ*L_reg={mean_penalty:.4f}"
        f"  ratio={final_ratio:.4f}",
        flush=True,
    )

    # ---- Clean up reference models before eval ----
    if method == "clora_safety":
        del aligned_model_ref, base_model_ref
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Evaluate ----
    final_ckpt = save_dir / f"epoch_{EPOCHS}"
    eval_model, eval_tok = load_model_and_tokenizer(str(final_ckpt), device=device)
    asr, _ = evaluate_safety(eval_model, eval_tok, harmful_prompts, device=device)
    task_perf = evaluate_task_performance(eval_model, eval_tok, gsm8k_test, task_type="gsm8k", device=device)
    gsm8k_acc = float(task_perf["accuracy"])
    del eval_model, eval_tok
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "method": method,
        "lam": lam,
        "lam_cap": lam_cap,
        "asr": float(asr),
        "gsm8k_acc": gsm8k_acc,
        "final_ratio": final_ratio,
        "final_epoch_mean_task_loss": mean_task,
        "final_epoch_mean_penalty": mean_penalty,
        "seed": SEED,
        "base_model": _BASE_MODEL,
        "epochs": EPOCHS,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        required=True,
        choices=["clora_random", "clora_safety", "olora_standard", "olora_safety"],
    )
    ap.add_argument(
        "--lam",
        type=float,
        default=None,
        help="λ for CLoRA / λ_orth for O-LoRA standard / λ_safety for Safety-O-LoRA.",
    )
    ap.add_argument(
        "--lam-safety",
        type=float,
        default=None,
        help="Alias for --lam when running Safety-O-LoRA (either flag works).",
    )
    ap.add_argument(
        "--lam-cap",
        type=float,
        default=0.01,
        help="λ_cap for Safety-O-LoRA capability adapters (fixed; default 0.01).",
    )
    args = ap.parse_args()

    # Resolve lam — accept either --lam or --lam-safety.
    lam = args.lam if args.lam is not None else args.lam_safety
    if lam is None:
        ap.error("--lam (or --lam-safety for Safety-O-LoRA) is required")

    lam_cap = float(args.lam_cap)
    method = args.method

    lam_tag = _lam_tag(lam)
    run_name = f"llama_lambdasweep_{method}_lam{lam_tag}_seed{SEED}"
    save_dir = _CKPT_ROOT / run_name
    results_path = _RESULTS_DIR / f"llama_2task_lambdasweep_{method}_{lam_tag}_seed{SEED}.json"

    print(f"[sweep] method={method}  lam={lam}  lam_cap={lam_cap}", flush=True)
    print(f"[sweep] checkpoint dir: {save_dir}", flush=True)
    print(f"[sweep] results path:   {results_path}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sweep] device={device}", flush=True)

    results = train_and_eval(
        method=method,
        lam=lam,
        lam_cap=lam_cap,
        device=device,
        save_dir=save_dir,
    )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[sweep] results written to {results_path}", flush=True)

    print(
        f"\n[sweep] DONE — method={method}  λ={lam}"
        f"  ASR={results['asr']*100:.1f}%"
        f"  GSM8K={results['gsm8k_acc']*100:.1f}%"
        f"  final_ratio={results['final_ratio']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
