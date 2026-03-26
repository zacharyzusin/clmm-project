from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from safety_clora.models.clora import CLoRALinear


def clora_regularization_loss(model: nn.Module) -> torch.Tensor:
    reg = None
    n = 0
    for m in model.modules():
        if isinstance(m, CLoRALinear):
            loss = m.clora_reg_loss()
            reg = loss if reg is None else (reg + loss)
            n += 1
    if reg is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    # Normalize by number of CLoRA modules so lambda has stable meaning.
    return reg / max(1, n)


@torch.no_grad()
def _cache_aligned_first_token_logits(
    aligned_model: nn.Module,
    tokenizer,
    safety_prompts: Sequence[str],
    device: torch.device,
    n_prompts: int,
    max_seq_len: int = 512,
) -> torch.Tensor:
    aligned_model.eval()
    prompts = list(safety_prompts)[:n_prompts]
    toks = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    ).to(device)
    out = aligned_model(**toks)
    logits = out.logits  # (b, seq, vocab)
    last_pos = toks["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(logits.size(0), device=device)
    first_tok_logits = logits[batch_idx, last_pos, :]  # (b, vocab)
    return first_tok_logits.detach().to(torch.float32).contiguous()


def first_token_kl_loss(
    model: nn.Module,
    aligned_model: nn.Module,
    safety_prompts: Sequence[str],
    tokenizer,
    device: torch.device,
    n_prompts: int = 16,
    cached_aligned_logits: Optional[torch.Tensor] = None,
    max_seq_len: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (kl_loss, cached_aligned_logits).
    KL(aligned || current) at the position that predicts the *first generated token*.
    """
    model.train()
    if cached_aligned_logits is None:
        cached_aligned_logits = _cache_aligned_first_token_logits(
            aligned_model=aligned_model,
            tokenizer=tokenizer,
            safety_prompts=safety_prompts,
            device=device,
            n_prompts=n_prompts,
            max_seq_len=max_seq_len,
        )

    prompts = list(safety_prompts)[: cached_aligned_logits.size(0)]
    toks = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    ).to(device)
    out = model(**toks)
    logits = out.logits.to(torch.float32)  # (b, seq, vocab)

    last_pos = toks["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(logits.size(0), device=device)
    cur_first_tok_logits = logits[batch_idx, last_pos, :]  # (b, vocab)

    ref_logp = F.log_softmax(cached_aligned_logits, dim=-1)
    cur_logp = F.log_softmax(cur_first_tok_logits, dim=-1)
    ref_p = ref_logp.exp()
    kl = (ref_p * (ref_logp - cur_logp)).sum(dim=-1).mean()
    return kl, cached_aligned_logits

