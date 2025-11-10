# nerf/nerf3d/rays/sampling.py
from __future__ import annotations
from typing import Tuple
import torch

# -----------------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------------
# Given world-space rays r(s) = o_w + s d_w with s ∈ [near, far], choose depth
# samples {s_i} per ray and map them to 3D points. This module:
#   1) builds linear depth grids (edges or centers)
#   2) adds stratified jitter during training
#   3) converts depths → 3D points
#   4) provides small helpers (edges↔centers, merge/sort, optional noise)
#
# Out of scope here: importance (PDF/CDF) resampling and volume rendering.
# We’ll add those later in a separate pass.

# -----------------------------------------------------------------------------
# A. Linear depths (coarse)
# -----------------------------------------------------------------------------

def make_linear_edges(
    near: torch.Tensor,      # (..., 1) or scalar
    far: torch.Tensor,       # (..., 1) or scalar
    n_samples: int,
) -> torch.Tensor:
    """
    Create (n_samples+1) linearly spaced depth edges between near and far.
    near, far can be scalar or broadcastable tensors.
    Returns: (..., n_samples+1)
    """
    # Convert to tensor (if scalar)
    if not isinstance(near, torch.Tensor):
        near = torch.tensor(near, dtype=torch.float32)
    if not isinstance(far, torch.Tensor):
        far = torch.tensor(far, dtype=torch.float32)

    # Produce linspace on the last dim and broadcast
    # shape: (n_samples+1,)
    base = torch.linspace(0.0, 1.0, steps=n_samples + 1, device=near.device, dtype=near.dtype)
    # Shape broadcast: near + (far-near)*base
    edges = near + (far - near) * base
    return edges


def edges_to_centers(edges: torch.Tensor) -> torch.Tensor:
    """
    Convert N+1 edges → N centers by midpoint between successive edges.
    edges: (..., N+1)
    Returns: (..., N)
    """
    # Average consecutive edges along last dimension
    return 0.5 * (edges[..., 1:] + edges[..., :-1])


def centers_to_edges(centers: torch.Tensor) -> torch.Tensor:
    """
    Convert N centers → N+1 edges.
    For endpoints, we extrapolate so that the first and last bin widths match neighbors.
    centers: (..., N)
    Returns: (..., N+1)
    """
    # Compute interior edges = midpoints of adjacent centers
    mid = 0.5 * (centers[..., 1:] + centers[..., :-1])  # (..., N-1)

    # Extrapolate start and end
    # First edge is 2*c0 - mid0
    first = (2 * centers[..., :1]) - mid[..., :1]
    # Last edge is 2*c_last - mid_last
    last = (2 * centers[..., -1:]) - mid[..., -1:]

    # Concatenate: [first, mid, last]
    return torch.cat([first, mid, last], dim=-1)


# -----------------------------------------------------------------------------
# B. Stratified jitter (training only)
# -----------------------------------------------------------------------------

def stratified_from_edges(
    edges: torch.Tensor,                    # (..., N+1)
    rng: torch.Generator | None = None,     # optional per-worker generator
) -> torch.Tensor:
    """
    Draw one uniform sample per bin [e_i, e_{i+1}).
    Returns: (..., N)
    """
    if edges.shape[-1] < 2:
        raise ValueError(f"edges must have length >= 2 on last dim, got {edges.shape}")
    # Bin starts & widths
    starts = edges[..., :-1]                      # (..., N)
    widths = (edges[..., 1:] - edges[..., :-1])  # (..., N)

    # Uniforms in [0,1)
    u = torch.rand(
        *starts.shape,
        device=edges.device,
        dtype=edges.dtype,
        generator=rng,
    )
    return starts + u * widths


def deterministic_centers_from_edges(
    edges: torch.Tensor,     # (..., N+1)
) -> torch.Tensor:
    """
    Eval-mode: centers of bins.
    Returns: (..., N)
    """
    if edges.shape[-1] < 2:
        raise ValueError(f"edges must have length >= 2 on last dim, got {edges.shape}")
    return 0.5 * (edges[..., 1:] + edges[..., :-1])


# -----------------------------------------------------------------------------
# C. Convenience wrappers
# -----------------------------------------------------------------------------

def sample_coarse_linear(
    o_w: torch.Tensor,                 # (..., 3) ray origins (unused here, but kept for API symmetry)
    d_w: torch.Tensor,                 # (..., 3) ray directions (unused here for linear depth sampling)
    near: float | torch.Tensor,
    far: float | torch.Tensor,
    n_samples: int,
    stratified: bool = True,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample uniformly in depth along rays (coarse pass).

    Args:
        o_w: (..., 3) ray origins in world coordinates
        d_w: (..., 3) ray directions in world coordinates
        near, far: scalar or broadcastable (..., 1)
        n_samples: number of samples along each ray
        stratified: True = random jitter in bins (training), False = midpoints (eval)
        rng: optional pytorch RNG for reproducibility

    Returns:
        t_vals: (..., n_samples) sampled depth values
        points: (..., n_samples, 3) 3D sample positions along rays
    """
    # Import local helpers
    from .sampling import (
        make_linear_edges,
        stratified_from_edges,
        deterministic_centers_from_edges,
    )

    # Generate (n_samples+1) edges between near and far
    edges = make_linear_edges(near, far, n_samples)  # (..., n+1)

    # Convert to sample depths
    if stratified:
        t_vals = stratified_from_edges(edges, rng=rng)   # (..., n)
    else:
        t_vals = deterministic_centers_from_edges(edges) # (..., n)

    # Compute 3D points: o + t * d   (broadcast across last dimension)
    #   o: (..., 3)
    #   d: (..., 3)
    #   t: (..., n) → (..., n, 1)
    points = o_w[..., None, :] + d_w[..., None, :] * t_vals[..., :, None]

    return t_vals, points


# -----------------------------------------------------------------------------
# D. Depths → 3D points (per ray)
# -----------------------------------------------------------------------------

def depths_to_points(
    origins: torch.Tensor,   # (..., 3)
    dirs: torch.Tensor,      # (..., 3)
    depths: torch.Tensor,    # (..., N)
) -> torch.Tensor:
    """
    X = o + t * d  (broadcasted)
    Returns: (..., N, 3)
    """
    if origins.shape[-1] != 3 or dirs.shape[-1] != 3:
        raise ValueError(f"bad shapes: origins{origins.shape}, dirs{dirs.shape}")
    if depths.dim() < 1:
        raise ValueError(f"depths must have at least 1 dim, got {depths.shape}")

    # harmonize dtype/device
    depths = depths.to(device=origins.device, dtype=origins.dtype)
    dirs   = dirs.to(device=origins.device, dtype=origins.dtype)

    # broadcast: (..., 1, 3) + (..., N, 1) * (..., 1, 3) -> (..., N, 3)
    return origins[..., None, :] + depths[..., :, None] * dirs[..., None, :]


# -----------------------------------------------------------------------------
# E. Small utilities
# -----------------------------------------------------------------------------

def add_noise_to_depths(
    depths: torch.Tensor,    # (..., N)
    noise_std: float,
    rng: torch.Generator | None = None,
    keep_sorted: bool = True,
) -> torch.Tensor:
    if noise_std <= 0:
        return depths
    noise = torch.randn_like(depths, generator=rng) * noise_std
    d = depths + noise
    if keep_sorted:
        d, _ = torch.sort(d, dim=-1)
    return d


def merge_and_sort_depths(
    a: torch.Tensor,         # (..., Na)
    b: torch.Tensor,         # (..., Nb)
) -> torch.Tensor:
    d = torch.cat([a, b], dim=-1)
    d, _ = torch.sort(d, dim=-1)
    return d


def clamp_depths(
    depths: torch.Tensor,    # (..., N)
    near: float | torch.Tensor,
    far: float | torch.Tensor,
) -> torch.Tensor:
    if not isinstance(near, torch.Tensor):
        near = torch.tensor(near, dtype=depths.dtype, device=depths.device)
    else:
        near = near.to(dtype=depths.dtype, device=depths.device)
    if not isinstance(far, torch.Tensor):
        far = torch.tensor(far, dtype=depths.dtype, device=depths.device)
    else:
        far = far.to(dtype=depths.dtype, device=depths.device)
    d = torch.maximum(depths, near)
    d = torch.minimum(d, far)
    return d
