"""
O-LoRA and Safety-O-LoRA implementations.

O-LoRA (Wang et al., EMNLP 2023, arxiv 2310.14152):
  For each new task t, trains a fresh LoRA adapter {A_t, B_t} while keeping all
  prior tasks' adapters frozen. An orthogonality penalty pushes A_t away from
  the column spaces of A_1 ... A_{t-1}:

    L_O-LoRA = L_task + lambda * sum_{i<t} ||A_i^T A_t||_F^2

Safety-O-LoRA (our modification):
  Treats the safety alignment adapter as a permanently privileged task:
  1. Asymmetric lambda: lambda_safety >> lambda_cap for other tasks
  2. Never-merge rule: safety adapter is kept as a separate add-on, never
     folded into base weights, so its subspace remains cleanly separated.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Re-use the parent-module utility from clora.py.
from safety_clora.models.clora import _get_parent_module


class OLoRALinear(nn.Module):
    """
    O-LoRA wrapped Linear layer.

    Forward:
        y = W_base(x)
          + sum_i  scaling_prev_i * (x B_prev_i^T A_prev_i^T)   [frozen]
          + scaling_curr            * (x B_curr^T  A_curr^T)     [trainable]

    A matrices: (out_features, rank)
    B matrices: (rank, in_features)
    Same convention as CLoRALinear so extraction utilities are compatible.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        prev_adapters: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        prev_scalings: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"OLoRALinear expects nn.Linear, got {type(base_layer)}")

        self.base = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = self.alpha / max(1, self.rank)

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        device = self.base.weight.device
        dtype = self.base.weight.dtype

        # Trainable current-task adapter.
        self.A = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(self.rank, self.in_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

        # Frozen previous-task adapters stored as named buffers.
        prev_adapters = prev_adapters or []
        default_sc = self.scaling
        prev_scalings = list(prev_scalings) if prev_scalings else [default_sc] * len(prev_adapters)

        self.n_prev = len(prev_adapters)
        for i, (A_i, B_i) in enumerate(prev_adapters):
            self.register_buffer(
                f"A_prev_{i}",
                A_i.detach().to(device=device, dtype=torch.float32).contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"B_prev_{i}",
                B_i.detach().to(device=device, dtype=torch.float32).contiguous(),
                persistent=True,
            )
        self.prev_scalings: List[float] = prev_scalings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_prev(self):
        """Yield (A_i, B_i, scaling_i) for each frozen previous adapter."""
        for i in range(self.n_prev):
            yield (
                getattr(self, f"A_prev_{i}"),
                getattr(self, f"B_prev_{i}"),
                self.prev_scalings[i],
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        x_f = x.to(torch.float32)
        for A_i, B_i, sc in self._iter_prev():
            delta = (x_f @ B_i.t()) @ A_i.t()
            out = out + (sc * delta).to(out.dtype)
        curr = (x @ self.B.t()) @ self.A.t()
        return out + self.scaling * curr

    # ------------------------------------------------------------------
    # Orthogonality loss
    # ------------------------------------------------------------------

    def olora_orth_loss(self, lam_list: Sequence[float]) -> torch.Tensor:
        """
        sum_i  lam_i * ||A_i^T @ A_curr||_F^2

        lam_list: one value per frozen prev adapter.
        """
        A_curr = self.A.to(torch.float32)
        loss = torch.zeros(1, device=self.A.device, dtype=torch.float32).squeeze()
        for i, (A_i, _, _) in enumerate(self._iter_prev()):
            lam = float(lam_list[i]) if i < len(lam_list) else float(lam_list[-1])
            gram = A_i.t() @ A_curr  # (rank, rank)
            loss = loss + lam * gram.pow(2).sum()
        return loss


# ---------------------------------------------------------------------------
# PEFT adapter extraction
# ---------------------------------------------------------------------------

def _peft_to_base_name(peft_name: str) -> str:
    """Strip 'base_model.model.' prefix added by PEFT's LoraModel wrapper."""
    prefix = "base_model.model."
    return peft_name[len(prefix):] if peft_name.startswith(prefix) else peft_name


def extract_peft_lora_adapters(
    peft_model: nn.Module,
    target_suffixes: Tuple[str, ...] = ("q_proj", "v_proj"),
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract LoRA adapter weights from a PeftModel **before** merging.

    Returns:
        {base_layer_name: (A, B)}
        A: (out_features, rank)  — our convention, matches CLoRALinear
        B: (rank, in_features)

    PEFT storage:
        lora_A['default'].weight: (rank, in_features)   <- our B
        lora_B['default'].weight: (out_features, rank)  <- our A
    """
    adapters: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for full_name, module in peft_model.named_modules():
        if not any(full_name.endswith(suf) for suf in target_suffixes):
            continue
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        try:
            # lora_B['default'].weight => (out, rank) == our A
            # lora_A['default'].weight => (rank, in)  == our B
            A = module.lora_B["default"].weight.detach().clone()  # (out, rank)
            B = module.lora_A["default"].weight.detach().clone()  # (rank, in)
        except (KeyError, AttributeError):
            continue
        base_name = _peft_to_base_name(full_name)
        adapters[base_name] = (A, B)
    return adapters


# ---------------------------------------------------------------------------
# Apply O-LoRA to a model
# ---------------------------------------------------------------------------

def apply_olora_to_model(
    model: nn.Module,
    rank: int,
    alpha: int,
    prev_adapters_by_name: Optional[Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    prev_scalings_by_name: Optional[Dict[str, List[float]]] = None,
    target_suffixes: Tuple[str, ...] = ("q_proj", "v_proj"),
) -> Tuple[nn.Module, List[Tuple[str, OLoRALinear]]]:
    """
    Replace target Linear layers with OLoRALinear modules.

    prev_adapters_by_name:  {layer_name: [(A_0, B_0), (A_1, B_1), ...]}
    prev_scalings_by_name:  {layer_name: [sc_0, sc_1, ...]}
      If None, each prev adapter uses alpha/rank scaling.

    Returns (model, [(layer_name, OLoRALinear), ...])
    """
    prev_adapters_by_name = prev_adapters_by_name or {}
    prev_scalings_by_name = prev_scalings_by_name or {}

    olora_mods: List[Tuple[str, OLoRALinear]] = []

    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(suf) for suf in target_suffixes):
            continue

        parent, child_name = _get_parent_module(model, full_name)
        if parent is None:
            continue

        prev_adapters = prev_adapters_by_name.get(full_name, [])
        prev_scalings = prev_scalings_by_name.get(full_name) or None

        wrapped = OLoRALinear(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            prev_adapters=prev_adapters,
            prev_scalings=prev_scalings,
        )
        setattr(parent, child_name, wrapped)
        olora_mods.append((full_name, wrapped))

    return model, olora_mods


# ---------------------------------------------------------------------------
# Aggregate orthogonality loss across all OLoRALinear modules
# ---------------------------------------------------------------------------

def olora_orth_loss_for_model(
    model: nn.Module,
    lam_list_by_name: Optional[Dict[str, List[float]]] = None,
) -> torch.Tensor:
    """
    Normalized sum of per-layer orthogonality losses.

    lam_list_by_name: {layer_name: [lam_0, lam_1, ...]}
      If None, falls back to lam=0.1 for all prev adapters.
    """
    lam_list_by_name = lam_list_by_name or {}
    total: Optional[torch.Tensor] = None
    n = 0
    for full_name, module in model.named_modules():
        if not isinstance(module, OLoRALinear) or module.n_prev == 0:
            continue
        lam_list = lam_list_by_name.get(full_name, [0.1])
        loss = module.olora_orth_loss(lam_list)
        total = loss if total is None else total + loss
        n += 1
    if total is None:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Merge adapters into base weights (for checkpoint saving)
# ---------------------------------------------------------------------------

def merge_olora_to_base_linear(
    model: nn.Module,
    skip_prev_indices: Optional[Sequence[int]] = None,
) -> nn.Module:
    """
    Merge OLoRALinear adapters back into the underlying base Linear layer.

    skip_prev_indices: if provided, those prev adapter indices are NOT merged
      (used to implement the Safety-O-LoRA never-merge rule for adapter 0 = safety).
      If None, ALL adapters (prev + current) are merged.

    After merging, each OLoRALinear is replaced with its base nn.Linear so the
    resulting model can be saved/reloaded by HuggingFace without custom module keys.
    """
    skip = set(skip_prev_indices) if skip_prev_indices is not None else set()

    for full_name, module in list(model.named_modules()):
        if not isinstance(module, OLoRALinear):
            continue
        parent, child_name = _get_parent_module(model, full_name)
        if parent is None:
            continue

        W = module.base.weight.data.float()

        for i, (A_i, B_i, sc) in enumerate(module._iter_prev()):
            if i in skip:
                continue
            W = W + sc * (A_i.float() @ B_i.float())

        # Current adapter
        W = W + module.scaling * (module.A.float() @ module.B.float())
        module.base.weight.data = W.to(module.base.weight.dtype)

        setattr(parent, child_name, module.base)

    return model
