# nerf/nerf3d/rays/pixels_to_rays.py
from __future__ import annotations
from typing import Tuple
import torch


def build_c2w(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    R: (...,3,3), t: (...,3) or (...,3,1)  →  T_c2w: (...,4,4)
    """
    if t.dim() == R.dim():
        t = t[..., 0]  # (...,3,1) -> (...,3)
    if t.shape[-1] != 3 or R.shape[-2:] != (3, 3):
        raise ValueError(f"bad shapes: R{R.shape}, t{t.shape}")
    *batch, _ = t.shape
    device, dtype = R.device, R.dtype

    T = torch.zeros(*batch, 4, 4, device=device, dtype=dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def invert_w2c(T_w2c: torch.Tensor) -> torch.Tensor:
    """
    T_w2c: (...,4,4)  →  T_c2w: (...,4,4)
    """
    if T_w2c.shape[-2:] != (4, 4):
        raise ValueError(f"bad shape: {T_w2c.shape}")
    R_wc = T_w2c[..., :3, :3]
    t_wc = T_w2c[..., :3, 3]

    R_cw = R_wc.transpose(-1, -2)
    t_cw = -(R_cw @ t_wc.unsqueeze(-1)).squeeze(-1)

    *batch, _ = t_cw.shape
    device, dtype = T_w2c.device, T_w2c.dtype
    T = torch.zeros(*batch, 4, 4, device=device, dtype=dtype)
    T[..., :3, :3] = R_cw
    T[..., :3, 3] = t_cw
    T[..., 3, 3] = 1.0
    return T


def pixel_to_cam_dir(
    K: torch.Tensor,
    uv: torch.Tensor,
    pixel_center: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Back-project pixel(s) to unit direction(s) in CAMERA space.

    Inputs:
      - K:  (..., 3, 3) intrinsics (shared or per-image); supports batching.
      - uv: (..., 2)    pixel coords (u, v) in pixels.
      - pixel_center:   if True, shift by +0.5 for center-of-pixel rays.
      - eps:            small epsilon for norm stability.

    Returns:
      - d_c: (..., 3) unit direction(s) in camera coordinates.

    Flow:
      1) If pixel_center: uv = uv + (0.5, 0.5).
      2) Form homogeneous pixel p = [u, v, 1].
      3) x_c ∝ K^{-1} p. (Use batched-safe solve/inverse.)
      4) Normalize to unit length -> d_c.
      5) Convention guard: we assume OpenGL camera (forward -Z). We keep as-is;
         any dataset flip will be handled by a separate convention utility (later if needed).
    """
    # TODO: implement
    raise NotImplementedError


def cam_to_world_ray(
    T_c2w: torch.Tensor,
    d_c: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform camera-space ray direction(s) to world space.

    Inputs:
      - T_c2w: (..., 4, 4) camera-to-world transform(s)
      - d_c:   (..., 3)    camera-space direction(s) (ideally unit)
      - normalize:         if True, re-normalize world directions
      - eps:               epsilon for norm

    Returns:
      - o_w: (..., 3)  world-space ray origin(s)  (o_w = translation part of T_c2w)
      - d_w: (..., 3)  world-space ray direction(s)

    Flow:
      1) Extract R = T_c2w[..., :3, :3], t = T_c2w[..., :3, 3].
      2) o_w = t (origin at camera center in world).
      3) d_w = R @ d_c (pure rotation on direction).
      4) If normalize: d_w = d_w / ||d_w||.
    """
    # TODO: implement
    raise NotImplementedError
