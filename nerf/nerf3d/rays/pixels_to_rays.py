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
    K:  (...,3,3) intrinsics (can be shared or per-item)
    uv: (...,2)   pixel coords (u,v) in pixels
    ->  d_c: (...,3) unit camera-space directions
    """
    if uv.shape[-1] != 2 or K.shape[-2:] != (3, 3):
        raise ValueError(f"bad shapes: K{K.shape}, uv{uv.shape}")

    # dtype/device harmonization
    uv = uv.to(device=K.device, dtype=K.dtype)

    # center-of-pixel shift
    if pixel_center:
        uv = uv + 0.5

    # homogeneous pixel
    ones = torch.ones_like(uv[..., :1])
    p = torch.cat([uv, ones], dim=-1)                     # (...,3)

    # broadcast K to match leading dims of p if needed
    if K.dim() == 2:
        K = K.expand(p.shape[:-1] + (3, 3))               # (...,3,3)

    # solve K * x = p  (avoid explicit inverse)
    x = torch.linalg.solve(K, p[..., None]).squeeze(-1)   # (...,3)

    # normalize to unit length
    n = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(torch.as_tensor(eps, dtype=K.dtype, device=K.device))
    d_c = x / n
    return d_c


def cam_to_world_ray(
    T_c2w: torch.Tensor,   # (...,4,4)
    d_c: torch.Tensor,     # (...,3)
    normalize: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if T_c2w.shape[-2:] != (4, 4) or d_c.shape[-1] != 3:
        raise ValueError(f"bad shapes: T{T_c2w.shape}, d_c{d_c.shape}")
    R = T_c2w[..., :3, :3]
    t = T_c2w[..., :3, 3]
    d_c = d_c.to(dtype=R.dtype, device=R.device)

    # d_w = R @ d_c
    d_w = (R @ d_c[..., None]).squeeze(-1)
    if normalize:
        n = torch.linalg.norm(d_w, dim=-1, keepdim=True).clamp_min(torch.as_tensor(eps, dtype=R.dtype, device=R.device))
        d_w = d_w / n
    o_w = t
    return o_w, d_w


def pixel_to_ray(
    K: torch.Tensor,        # (...,3,3)
    T_c2w: torch.Tensor,    # (...,4,4)
    uv: torch.Tensor,       # (...,2)
    pixel_center: bool = True,
    normalize: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .pixels_to_rays import pixel_to_cam_dir  # local import to avoid cycles if split
    d_c = pixel_to_cam_dir(K, uv, pixel_center=pixel_center, eps=eps)
    o_w, d_w = cam_to_world_ray(T_c2w, d_c, normalize=normalize, eps=eps)
    return o_w, d_w
