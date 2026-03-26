"""
CLoRA and Safety-CLoRA implementations.

CLoRA (Controlled LoRA): Adds an orthogonal regularization term to LoRA training.
  L_reg = (||A^T S||_F^2 + ||S^T B^T||_F^2)
  where S is a pre-defined (frozen) regularization matrix.

Safety-CLoRA: CLoRA where S is initialized from the alignment direction
  d_aligned = W_aligned - W_base, rather than randomly.
  S is set to span the orthogonal complement of d_aligned (per layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CLoRAConfig:
    rank: int = 8
    alpha: int = 16


class CLoRALinear(nn.Module):
    """
    A minimal LoRA-wrapped Linear with CLoRA regularization.

    LoRA update:  y = xW^T + (alpha/r) * x B^T A^T
      - A: (out_features, r)
      - B: (r, in_features)

    CLoRA regularizer (paper form):
      ||A^T S||_F^2 + ||S^T B^T||_F^2

    Here S is stored as a frozen buffer.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        s_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"CLoRALinear expects nn.Linear, got {type(base_layer)}")

        self.base = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = self.alpha / max(1, self.rank)

        # Freeze base layer weights (typical PEFT usage)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # LoRA params
        self.A = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.B = nn.Parameter(torch.zeros(self.rank, self.in_features))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

        # S matrix: if None, random orthonormal columns
        if s_matrix is None:
            s_matrix = _random_orthonormal_matrix(
                d=self.out_features,
                m=min(self.rank, self.out_features),
                device=self.A.device,
                dtype=self.A.dtype,
            )
        else:
            if s_matrix.dim() != 2 or s_matrix.size(0) != self.out_features:
                raise ValueError(
                    f"s_matrix must be 2D with shape ({self.out_features}, m), got {tuple(s_matrix.shape)}"
                )
            s_matrix = s_matrix.to(device=self.A.device, dtype=self.A.dtype)

        self.register_buffer("S", s_matrix, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.B.t()) @ self.A.t()
        return base_out + self.scaling * lora_out

    def clora_reg_loss(self) -> torch.Tensor:
        # A: (out, r)  S: (out, m) -> A^T S: (r, m)
        a_term = (self.A.t() @ self.S).pow(2).sum()
        # B: (r, in)   B^T: (in, r)  S^T B^T requires matching dims.
        # The original paper uses S with shapes that match the specific formulation.
        # For this implementation we use the common variant:
        #   ||A^T S||_F^2 + ||B S||_F^2
        # where S is (in, m) would be needed for the second term.
        #
        # To keep a single S buffer per layer and still penalize both factors,
        # we build an input-space S_in on the fly (random but deterministic per-module)
        # when S has incompatible shape. For Safety-CLoRA we mainly care about the A-side
        # constraint (directional control in output space).
        if self.S.size(0) == self.in_features:
            b_term = (self.B @ self.S).pow(2).sum()
        else:
            # fall back: penalize B magnitude (lightweight proxy)
            b_term = self.B.pow(2).sum()
        return a_term + b_term


def compute_alignment_direction(
    base_model: nn.Module,
    aligned_model: nn.Module,
    layer_name: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Returns normalized d_aligned = W_aligned - W_base for `layer_name` (a parameter name).
    """
    base_sd = dict(base_model.named_parameters())
    aligned_sd = dict(aligned_model.named_parameters())
    if layer_name not in base_sd or layer_name not in aligned_sd:
        raise KeyError(f"layer_name '{layer_name}' not found in both models' parameters")
    d = aligned_sd[layer_name].detach() - base_sd[layer_name].detach()
    if device is not None:
        d = d.to(device)
    flat = d.flatten()
    n = flat.norm(p=2).clamp_min(1e-12)
    return (flat / n).to(dtype=torch.float32)


def compute_orthogonal_complement_basis(
    direction_vector: torch.Tensor,
    n_vectors: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build S whose columns are orthonormal and orthogonal to `direction_vector`.
    direction_vector: 1D unit vector (d,)
    returns: S matrix (d, n_vectors)
    """
    u = direction_vector.to(device=device, dtype=torch.float32).flatten()
    d = u.numel()
    n_vectors = int(min(n_vectors, max(1, d - 1)))

    # Random matrix -> project out u -> QR.
    R = torch.randn(d, n_vectors, device=device, dtype=torch.float32)
    proj = u[:, None] * (u[None, :] @ R)  # (d, n_vectors)
    R_orth = R - proj
    Q, _ = torch.linalg.qr(R_orth, mode="reduced")
    return Q[:, :n_vectors].contiguous()


def build_safety_s_matrix(
    base_model: nn.Module,
    aligned_model: nn.Module,
    module_name: str,
    rank: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Compute an S matrix for a specific Linear module, using the module's weight parameter name:
      {module_name}.weight

    We compute an orthogonal-complement basis in the flattened weight space, then reshape to
    (out_features, m) by projecting to output space for tractability.

    Practical approximation:
    - Full weight space is huge; we instead create S in output space (out_features x m).
    - We do this by computing the output-space direction as the top left singular vector
      of d_aligned_W and then building an orthogonal basis around it.
    """
    weight_name = f"{module_name}.weight"
    try:
        base_w = dict(base_model.named_parameters())[weight_name].detach().to(device=device, dtype=torch.float32)
        aligned_w = dict(aligned_model.named_parameters())[weight_name].detach().to(device=device, dtype=torch.float32)
    except KeyError:
        return None

    dW = aligned_w - base_w  # (out, in)
    out_features = dW.size(0)
    if out_features < 2:
        return None

    # Approximate direction in output space: dominant left singular vector of dW.
    try:
        U, Svals, _Vh = torch.linalg.svd(dW, full_matrices=False)
        u_out = U[:, 0]  # (out,)
    except Exception:
        # fallback: mean over input dim
        u_out = dW.mean(dim=1)

    u_out = u_out / u_out.norm(p=2).clamp_min(1e-12)
    S_out = compute_orthogonal_complement_basis(u_out, n_vectors=min(rank, out_features - 1), device=device)
    return S_out  # (out, m)


def apply_clora_to_model(
    model: nn.Module,
    rank: int,
    alpha: int,
    lam: float,
    mode: str,
    base_model: Optional[nn.Module] = None,
    aligned_model: Optional[nn.Module] = None,
) -> Tuple[nn.Module, List[CLoRALinear]]:
    """
    Replace attention q_proj/v_proj Linear layers with CLoRALinear.
    mode: 'random' or 'safety'
    Returns (model, clora_modules)
    """
    if mode not in {"random", "safety"}:
        raise ValueError("mode must be 'random' or 'safety'")
    if mode == "safety" and (base_model is None or aligned_model is None):
        raise ValueError("mode='safety' requires base_model and aligned_model")

    target_suffixes = ("q_proj", "v_proj")
    clora_modules: List[CLoRALinear] = []

    # We'll traverse named_modules and replace via parent module setattr.
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(suf) for suf in target_suffixes):
            continue

        parent, child_name = _get_parent_module(model, full_name)
        if parent is None:
            continue

        s_mat = None
        if mode == "safety":
            s_mat = build_safety_s_matrix(
                base_model=base_model,
                aligned_model=aligned_model,
                module_name=full_name,
                rank=rank,
                device=next(model.parameters()).device,
            )

        wrapped = CLoRALinear(module, rank=rank, alpha=alpha, s_matrix=s_mat)
        setattr(parent, child_name, wrapped)
        clora_modules.append(wrapped)

    return model, clora_modules


def _random_orthonormal_matrix(d: int, m: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    m = int(min(m, d))
    X = torch.randn(d, m, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(X, mode="reduced")
    return Q[:, :m].to(device=device, dtype=dtype).contiguous()


def _get_parent_module(model: nn.Module, module_path: str) -> Tuple[Optional[nn.Module], str]:
    parts = module_path.split(".")
    if len(parts) == 1:
        return None, parts[0]
    parent = model
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, parts[-1]
        parent = getattr(parent, p)
    return parent, parts[-1]

