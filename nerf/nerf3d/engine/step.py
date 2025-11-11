# nerf/nerf3d/engine/step.py

from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch

# deps we already wrote:
# from nerf.nerf3d.rays.sampling import sample_coarse_linear, depths_to_points
# from nerf.nerf3d.viz.volume_rendering import (
#     compute_deltas, alpha_from_sigma, transmittance_from_alpha,
#     weights_from_alpha_T, composite_rgb, composite_depth
# )

def _prepare_rays(rays: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    rays: {'o':(B,3), 'd':(B,3)}
    Returns:
      o:(B,3), d_hat:(B,3) normalized directions
    """
    o = rays["o"]
    d = rays["d"]
    assert o.shape == d.shape and o.shape[-1] == 3, "rays must have o,d with shape (B,3)"
    d_norm = torch.clamp(d.norm(dim=-1, keepdim=True), min=1e-8)
    d_hat = d / d_norm
    return o, d_hat


def _sample_depths(
    B: int,
    n_samples: int,
    near: torch.Tensor,   # (B,) or scalar-tensor on correct device/dtype
    far: torch.Tensor,    # (B,) or scalar-tensor
    perturb: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds per-ray linspace in [near, far].
    Returns:
      t_edges: (B, n_samples+1)
      t_mid:   (B, n_samples)  (jittered if perturb=True)
    """
    # Create a (1, n+1) linspace template on the same device/dtype
    device, dtype = near.device, near.dtype
    lin = torch.linspace(0.0, 1.0, steps=n_samples + 1, device=device, dtype=dtype).unsqueeze(0)  # (1, n+1)

    # Broadcast near/far to (B,1)
    if near.ndim == 0:
        near = near.expand(B)
    if far.ndim == 0:
        far = far.expand(B)
    near = near.view(B, 1)
    far = far.view(B, 1)

    # Edges per ray
    t_edges = near + (far - near) * lin  # (B, n+1)

    # Centers
    t_mid = 0.5 * (t_edges[:, :-1] + t_edges[:, 1:])  # (B, n)

    if perturb:
        # Stratified jitter within each interval
        deltas = t_edges[:, 1:] - t_edges[:, :-1]      # (B, n)
        t_mid = t_edges[:, :-1] + deltas * torch.rand_like(t_mid)

    return t_edges, t_mid


def _query_model_chunks(
    model,
    x_world: torch.Tensor,  # (B, n, 3)
    d_hat: torch.Tensor,    # (B, 3)
    chunk: Optional[int] = None,
    amp: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens (B,n,3) → (B*n,3), repeats d̂ accordingly, runs MLP in chunks.
    Returns:
      rgb_samples: (B, n, 3)
      sigma:       (B, n, 1)
    """
    B, n, _ = x_world.shape
    device = x_world.device

    x_flat = x_world.reshape(B * n, 3)                    # (BN,3)
    d_rep  = d_hat.unsqueeze(1).expand(B, n, 3).reshape(B * n, 3)  # (BN,3)

    rgb_outs = []
    sig_outs = []

    use_amp = amp and (device.type == "cuda")
    if chunk is None or chunk <= 0:
        chunk = B * n

    with torch.cuda.amp.autocast(enabled=use_amp):
        for s in range(0, B * n, chunk):
            e = min(s + chunk, B * n)
            rgb_chunk, sigma_chunk = model(x_flat[s:e], d_rep[s:e])  # (m,3), (m,1)
            rgb_outs.append(rgb_chunk)
            sig_outs.append(sigma_chunk)

    rgb = torch.cat(rgb_outs, dim=0).reshape(B, n, 3)
    sigma = torch.cat(sig_outs, dim=0).reshape(B, n, 1)
    return rgb, sigma

def forward_batch(
    model,
    rays: Dict[str, torch.Tensor],   # {'o':(B,3), 'd':(B,3)}
    n_samples: int,
    near: float,
    far: float,
    perturb: bool,
    bg_color: float | torch.Tensor = 0.0,
    chunk: Optional[int] = None,
    amp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    # local imports to avoid circulars
    from nerf.nerf3d.rays.sampling import depths_to_points
    from nerf.nerf3d.viz.volume_rendering import (
        compute_deltas, alpha_from_sigma, transmittance_from_alpha,
        weights_from_alpha_T, composite_rgb, composite_depth
    )

    o, d_hat = _prepare_rays(rays)  # (B,3),(B,3)
    B = o.shape[0]
    device, dtype = o.device, o.dtype

    # near/far → (B,)
    near_t = torch.as_tensor(near, device=device, dtype=dtype)
    far_t  = torch.as_tensor(far,  device=device, dtype=dtype)
    if near_t.ndim == 0: near_t = near_t.expand(B)
    if far_t.ndim  == 0: far_t  = far_t.expand(B)

    # depths
    t_edges, t_mid = _sample_depths(B, n_samples, near_t, far_t, perturb)  # (B,n+1),(B,n)

    # points along rays
    pts = depths_to_points(o, d_hat, t_mid)  # (B,n,3)

    # MLP queries
    rgb_s, sigma = _query_model_chunks(model, pts, d_hat, chunk=chunk, amp=amp)  # (B,n,3),(B,n,1)

    # volume rendering terms
    deltas = compute_deltas(t_edges)                 # (B,n)
    alpha  = alpha_from_sigma(sigma, deltas)         # (B,n)
    T      = transmittance_from_alpha(alpha)         # (B,n)
    w      = weights_from_alpha_T(alpha, T)          # (B,n)

    # bg color → (B,3)
    if isinstance(bg_color, (int, float)):
        bg = torch.tensor([bg_color]*3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3)
    else:
        bg = bg_color.to(device=device, dtype=dtype)
        if bg.ndim == 1 and bg.shape[0] == 3:
            bg = bg.unsqueeze(0).expand(B, 3)
        elif bg.ndim == 0:
            bg = torch.tensor([bg.item()]*3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3)

    # composites
    rgb   = composite_rgb(rgb_s, w, bg)              # (B,3)
    depth = composite_depth(t_mid, w)                # (B,)

    extras = {
        "t_edges": t_edges,
        "t_mid": t_mid,
        "weights": w,
        "alpha": alpha,
        "deltas": deltas,
    }
    return rgb, depth, extras