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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


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

    In practice, the two terms operate on different spaces:
      - A is (out, r) so S_out should be (out, m)
      - B is (r, in) so S_in should be (in, m)

    We therefore store S_out and S_in as frozen buffers.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        s_out: Optional[torch.Tensor] = None,
        s_in: Optional[torch.Tensor] = None,
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

        param_device = self.base.weight.device
        param_dtype = self.base.weight.dtype

        # LoRA params
        self.A = nn.Parameter(torch.zeros(self.out_features, self.rank, device=param_device, dtype=param_dtype))
        self.B = nn.Parameter(torch.zeros(self.rank, self.in_features, device=param_device, dtype=param_dtype))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.zeros_(self.B)

        # S_out and S_in: if not provided, initialize random orthonormal columns.
        if s_out is None:
            s_out = _random_orthonormal_matrix(
                d=self.out_features,
                m=min(self.rank, max(1, self.out_features - 1)),
                device=param_device,
                dtype=param_dtype,
            )
        else:
            if s_out.dim() != 2 or s_out.size(0) != self.out_features:
                raise ValueError(
                    f"s_out must be 2D with shape ({self.out_features}, m), got {tuple(s_out.shape)}"
                )
            s_out = s_out.to(device=param_device, dtype=param_dtype)

        if s_in is None:
            s_in = _random_orthonormal_matrix(
                d=self.in_features,
                m=min(self.rank, max(1, self.in_features - 1)),
                device=param_device,
                dtype=param_dtype,
            )
        else:
            if s_in.dim() != 2 or s_in.size(0) != self.in_features:
                raise ValueError(
                    f"s_in must be 2D with shape ({self.in_features}, m), got {tuple(s_in.shape)}"
                )
            s_in = s_in.to(device=param_device, dtype=param_dtype)

        self.register_buffer("S_out", s_out, persistent=True)
        self.register_buffer("S_in", s_in, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.B.t()) @ self.A.t()
        return base_out + self.scaling * lora_out

    def clora_reg_loss(self) -> torch.Tensor:
        # ||A^T S_out||_F^2
        a_term = (self.A.t() @ self.S_out).pow(2).sum()
        # ||S_in^T B^T||_F^2 == ||B S_in||_F^2
        b_term = (self.B @ self.S_in).pow(2).sum()
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


def build_safety_s_matrices(
    base_model: nn.Module,
    aligned_model: nn.Module,
    module_name: str,
    rank: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute (S_out, S_in) for a specific Linear module, using the module's weight parameter name:
      {module_name}.weight

    Practical approximation (tractable):
    - Compute dW = W_aligned - W_base (out, in)
    - Use its dominant left singular vector u_out (out,) as the alignment direction in output space.
    - Use its dominant right singular vector v_in (in,) as the alignment direction in input space.
    - Set:
        S_out = basis of orthogonal complement to u_out  (out, m)
        S_in  = basis of orthogonal complement to v_in   (in, m)

    This matches the intent of "penalize updates orthogonal to the alignment direction"
    without ever constructing bases in the full flattened (out*in)-dimensional space.
    """
    weight_name = f"{module_name}.weight"
    try:
        base_w = dict(base_model.named_parameters())[weight_name].detach().to(device=device, dtype=torch.float32)
        aligned_w = dict(aligned_model.named_parameters())[weight_name].detach().to(device=device, dtype=torch.float32)
    except KeyError:
        return None, None

    dW = aligned_w - base_w  # (out, in)
    out_features = dW.size(0)
    in_features = dW.size(1)
    if out_features < 2 or in_features < 2:
        return None, None

    try:
        U, _Svals, Vh = torch.linalg.svd(dW, full_matrices=False)
        u_out = U[:, 0]  # (out,)
        v_in = Vh[0, :]  # (in,)
    except Exception:
        # fallback: mean over input dim
        u_out = dW.mean(dim=1)
        v_in = dW.mean(dim=0)

    u_out = u_out / u_out.norm(p=2).clamp_min(1e-12)
    v_in = v_in / v_in.norm(p=2).clamp_min(1e-12)

    m_out = min(rank, out_features - 1)
    m_in = min(rank, in_features - 1)
    S_out = compute_orthogonal_complement_basis(u_out, n_vectors=m_out, device=device) if m_out > 0 else None
    S_in = compute_orthogonal_complement_basis(v_in, n_vectors=m_in, device=device) if m_in > 0 else None
    return S_out, S_in


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

        s_out = None
        s_in = None
        if mode == "safety":
            s_out, s_in = build_safety_s_matrices(
                base_model=base_model,
                aligned_model=aligned_model,
                module_name=full_name,
                rank=rank,
                device=next(model.parameters()).device,
            )

        wrapped = CLoRALinear(module, rank=rank, alpha=alpha, s_out=s_out, s_in=s_in)
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

