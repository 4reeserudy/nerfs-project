# nerf/learn2d/pe.py
from __future__ import annotations
import torch

def output_dim(L: int) -> int:
    """For 2D inputs with input retained: D = 2 + 4L."""
    return 2 + 4 * L


def get_frequencies(L: int) -> torch.Tensor:
    """
    Dyadic frequencies f_k = 2^k for k=0..L-1.
    Returns float32 tensor of shape (L,).
    """
    if L <= 0:
        return torch.empty(0, dtype=torch.float32)
    return 2.0 ** torch.arange(L, dtype=torch.float32)


def apply_bands(xy: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    xy:    (N, 2) in [0,1], float32
    freqs: (L,) float32
    Returns: (N, 4L) with per-k ordering:
      [ sin(pi*f_k*x), cos(pi*f_k*x),
        sin(pi*f_k*y), cos(pi*f_k*y) ] for k=0..L-1
    """
    N = xy.shape[0]
    if freqs.numel() == 0:
        return torch.empty((N, 0), dtype=xy.dtype, device=xy.device)

    # Match dtype/device
    freqs = freqs.to(dtype=xy.dtype, device=xy.device)  # (L,)
    x = xy[:, 0:1]  # (N,1)
    y = xy[:, 1:1+1]

    # Angles: (N,L)
    angles_x = x * (torch.pi * freqs.view(1, -1))
    angles_y = y * (torch.pi * freqs.view(1, -1))

    # Trig features (N,L)
    sx = torch.sin(angles_x)
    cx = torch.cos(angles_x)
    sy = torch.sin(angles_y)
    cy = torch.cos(angles_y)

    # Interleave per-k: stack -> (N,L,4) then flatten -> (N,4L)
    trig_feats = torch.stack([sx, cx, sy, cy], dim=2).reshape(N, -1)
    return trig_feats


def fourier_encode(xy: torch.Tensor, L: int) -> torch.Tensor:
    """
    Keep original inputs and append sin/cos bands:
      feats = [x, y, sin(pi*2^0*x), cos(...), sin(pi*2^0*y), cos(...), ..., sin(pi*2^{L-1}*y), cos(...)]
    Returns: (N, 2 + 4L)
    """
    if L < 0:
        raise ValueError("L must be >= 0")
    trig = apply_bands(xy, get_frequencies(L))
    return torch.cat([xy, trig], dim=-1)
