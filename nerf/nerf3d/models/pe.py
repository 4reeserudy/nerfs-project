# nerf/nerf3d/models/pe.py
from __future__ import annotations
import torch

__all__ = ["positional_encoding"]

def positional_encoding(
    x: torch.Tensor, L: int, include_input: bool = True
) -> torch.Tensor:
    """
    Generic sinusoidal PE for last-dim D (works for 2D, 3D, ...).
    x: (..., D)
    returns: (..., D + 2*D*L) if include_input else (..., 2*D*L)
    """
    if L <= 0:
        return x if include_input else x.new_zeros((*x.shape[:-1], 0))

    # shape helpers
    orig_shape = x.shape
    D = orig_shape[-1]
    x = x.reshape(-1, D)  # (N, D)

    # frequencies: [1, 2, 4, ..., 2^{L-1}] * pi
    freqs = (2.0 ** torch.arange(L, device=x.device, dtype=x.dtype)) * torch.pi  # (L,)

    # (N, D) -> (N, L, D) broadcast
    angles = x.unsqueeze(1) * freqs.view(L, 1)  # (N, L, D)

    # sin/cos, then flatten the (L, D) axes
    sin = torch.sin(angles)  # (N, L, D)
    cos = torch.cos(angles)  # (N, L, D)
    pe = torch.cat([sin, cos], dim=1)  # (N, 2L, D)
    pe = pe.permute(0, 2, 1).reshape(x.shape[0], D * 2 * L)  # (N, 2*D*L)

    out = [x] if include_input else []
    out.append(pe)
    out = torch.cat(out, dim=-1)  # (N, D + 2*D*L) or (N, 2*D*L)
    return out.reshape(*orig_shape[:-1], out.shape[-1])