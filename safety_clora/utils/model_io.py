from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device, *, trainable: bool = False):
    """
    Loads either:
      - a full Transformers model checkpoint, or
      - a PEFT adapter checkpoint saved under {ckpt}/adapter with a base model at {ckpt}.

    Returns: (model, tokenizer)
    """
    ckpt = Path(checkpoint_path)

    # HF Hub ID (not a local dir)
    if not ckpt.exists():
        tok = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
        return model, tok

    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    marker = ckpt / "CHECKPOINT_TYPE"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == "peft_lora_adapter":
        base_id_file = ckpt / "BASE_MODEL_ID"
        if not base_id_file.exists():
            raise FileNotFoundError(f"Missing BASE_MODEL_ID in {ckpt} for PEFT adapter checkpoint.")
        base_id = base_id_file.read_text(encoding="utf-8").strip()
        base = AutoModelForCausalLM.from_pretrained(base_id).to(device)
        model = PeftModel.from_pretrained(base, ckpt / "adapter", is_trainable=trainable).to(device)
        return model, tok

    model = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
    return model, tok

