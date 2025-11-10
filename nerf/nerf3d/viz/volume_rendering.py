# nerf/nerf3d/viz/volume_rendering.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import torch


# -----------------------------------------------------------------------------
# Core utilities (no I/O, vectorized over batch B and samples S)
# -----------------------------------------------------------------------------

def compute_deltas(
    t_edges: Optional[torch.Tensor] = None,   # (B, S+1)
    t_centers: Optional[torch.Tensor] = None, # (B, S)
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return sample centers and spacings Δ_i.

    If edges are provided:
        t_mid  = 0.5 * (e[i] + e[i+1])
        deltas = e[i+1] - e[i]

    If only centers are provided:
        build edges by midpointing neighbors and extrapolating the two ends.
    """
    if t_edges is not None:
        # (B, S), (B, S)
        t_mid = 0.5 * (t_edges[:, :-1] + t_edges[:, 1:])
        deltas = t_edges[:, 1:] - t_edges[:, :-1]
    elif t_centers is not None:
        c = t_centers
        # Mid-edges between centers
        mid_edges = 0.5 * (c[:, :-1] + c[:, 1:])               # (B, S-1)
        # Extrapolate left/right edges using first/last gaps
        left_gap  = (c[:, 1] - c[:, 0]).unsqueeze(1)           # (B, 1)
        right_gap = (c[:, -1] - c[:, -2]).unsqueeze(1)         # (B, 1)
        left_edge  = (c[:, 0] - 0.5 * left_gap)                # (B, 1)
        right_edge = (c[:, -1] + 0.5 * right_gap)              # (B, 1)
        edges = torch.cat([left_edge, mid_edges, right_edge], dim=1)  # (B, S+1)

        t_mid = c                                              # centers are already midpoints
        deltas = edges[:, 1:] - edges[:, :-1]                  # (B, S)
    else:
        raise ValueError("compute_deltas: provide either t_edges or t_centers")

    deltas = torch.clamp_min(deltas, eps)
    return t_mid, deltas


def alpha_from_sigma(
    sigma: torch.Tensor,     # (B, S)
    deltas: torch.Tensor,    # (B, S)
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Beer–Lambert opacity per sample:
        α_i = 1 - exp(-σ_i * Δ_i)
    """
    # Ensure positive spacings; avoid negative exponents explosion
    deltas = torch.clamp_min(deltas, eps)
    # Product σ * Δ can be large; clamp upper bound for numerical stability
    tau = torch.clamp(sigma * deltas, min=0.0, max=60.0)  # exp(-60) ~ 8.8e-27
    alpha = 1.0 - torch.exp(-tau)
    return torch.clamp(alpha, 0.0, 1.0)


def transmittance_from_alpha(
    alpha: torch.Tensor,   # (B, S)
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute cumulative transmittance T_i for each sample:

        T_0 = 1
        T_i = Π_{j < i} (1 - α_j)

    Returned shape: (B, S)

    Implementation uses cumulative product for efficiency and stability.
    """
    # (B, S) → (B, S) exclusive cumprod of (1 - α)
    # torch.cumprod is inclusive, so shift by one
    one_minus_a = torch.clamp(1.0 - alpha, eps, 1.0)  # avoid 0
    cum = torch.cumprod(one_minus_a, dim=-1)          # T_0 = (1-α0), T_1 = (1-α0)(1-α1), ...
    
    # Convert to exclusive: shift right and insert 1 at start
    B, S = alpha.shape
    T = torch.ones((B, S), device=alpha.device, dtype=alpha.dtype)
    T[:, 1:] = cum[:, :-1]
    return T


def weights_from_alpha_T(
    alpha: torch.Tensor,   # (B, S)
    T: torch.Tensor        # (B, S) Transmittance at each sample
) -> torch.Tensor:
    """
    Convert opacity α and transmittance T into volume-rendering weights:

        w_i = α_i * T_i

    These weights sum to ≤ 1 per ray (remaining mass goes to background).
    """
    weights = alpha * T
    # Clamp for safety; weights must be non-negative
    return torch.clamp(weights, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Compositing heads
# -----------------------------------------------------------------------------

def composite_rgb(
    colors: torch.Tensor,   # (B, S, 3) in [0,1]
    weights: torch.Tensor,  # (B, S)
    white_bg: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RGB = Σ_i w_i * c_i
    acc = Σ_i w_i
    If white_bg: RGB += (1 - acc)
    """
    w = weights.unsqueeze(-1)                # (B,S,1)
    rgb = (w * colors).sum(dim=1)            # (B,3)
    acc = weights.sum(dim=1)                 # (B,)
    if white_bg:
        rgb = rgb + (1.0 - acc).unsqueeze(-1)
    return rgb.clamp(0.0, 1.0), acc.clamp(0.0, 1.0)


def composite_depth(
    t_mid: torch.Tensor,    # (B, S)
    weights: torch.Tensor,  # (B, S)
) -> torch.Tensor:
    """
    Depth = Σ_i w_i * t_i  (no normalization)
    """
    return (weights * t_mid).sum(dim=1)      # (B,)


# -----------------------------------------------------------------------------
# High-level renderer (math only, no plots)
# -----------------------------------------------------------------------------

def render_volume(
    sigma: torch.Tensor,                    # (B, S)
    colors: torch.Tensor,                   # (B, S, 3) in [0,1]
    t_edges: Optional[torch.Tensor] = None, # (B, S+1)
    t_centers: Optional[torch.Tensor] = None,# (B, S)
    white_bg: bool = False,
    eps: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """
    Discrete volume rendering:
      1) t_mid, Δ = compute_deltas(...)
      2) α = 1 - exp(-σ Δ)
      3) T = Π_{j<i} (1 - α_j)   (exclusive)
      4) w = T * α
      5) rgb = Σ w c   (+ white bg term)
      6) depth = Σ w t
    """
    if (t_edges is None) and (t_centers is None):
        raise ValueError("render_volume: provide t_edges or t_centers")

    # Shapes sanity
    B, S = sigma.shape
    if colors.shape[:2] != (B, S):
        raise ValueError(f"colors shape {colors.shape} must be (B,S,3) with B={B}, S={S}")
    if colors.shape[-1] != 3:
        raise ValueError("colors must have last dim = 3")

    # 1) centers & spacings
    t_mid, deltas = compute_deltas(t_edges=t_edges, t_centers=t_centers, eps=eps)  # (B,S), (B,S)

    # 2) alpha from density
    alpha = alpha_from_sigma(sigma, deltas, eps=eps)  # (B,S)

    # 3) transmittance
    T = transmittance_from_alpha(alpha, eps=eps)      # (B,S)

    # 4) weights
    weights = weights_from_alpha_T(alpha, T)          # (B,S)

    # 5) composite rgb (+ optional white bg)
    rgb, acc = composite_rgb(colors, weights, white_bg=white_bg)  # (B,3), (B,)

    # 6) expected depth
    depth = composite_depth(t_mid, weights)           # (B,)

    return {
        "rgb": rgb,           # (B,3)
        "acc": acc,           # (B,)
        "depth": depth,       # (B,)
        "weights": weights,   # (B,S)
        "t_mid": t_mid,       # (B,S)
        "alpha": alpha,       # (B,S)
        "T": T,               # (B,S)
        "deltas": deltas,     # (B,S)
    }