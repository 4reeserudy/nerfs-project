# nerf/nerf3d/data/losses.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import torch


# -----------------------------------------------------------------------------
# Core photometric loss (simple)
# -----------------------------------------------------------------------------

def rgb_l2(
    pred_rgb: torch.Tensor, 
    gt_rgb: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Basic L2 (MSE) loss between predicted and ground-truth RGB.

    Args:
        pred_rgb: (N, 3) float, in [0,1]
        gt_rgb:   (N, 3) float, in [0,1]
        mask:     (N,) or (N,1), optional. 1 = include, 0 = ignore.

    Returns:
        Scalar tensor loss (mean over valid pixels)
    """
    # Squared error per pixel
    se = (pred_rgb - gt_rgb) ** 2  # (N,3)

    if mask is not None:
        # Ensure mask shape is broadcast-compatible with (N,3)
        if mask.dim() == 1:
            mask = mask[:, None]  # (N,1)

        se = se * mask            # zero out masked pixels
        denom = mask.sum() * 3.0  # total number of valid color channels

        if denom < 1e-8:
            # If mask removes everything, return zero to avoid NaN
            return torch.zeros((), device=pred_rgb.device)

        return se.sum() / denom

    # Unmasked: mean over all pixels and all 3 channels
    return se.mean()


# -----------------------------------------------------------------------------
# PSNR metric (for logging)
# -----------------------------------------------------------------------------

def psnr(mse: torch.Tensor) -> torch.Tensor:
    """
    Convert MSE to PSNR (Peak Signal-to-Noise Ratio).

    Args:
        mse: Scalar tensor mean squared error (in RGB space, values in [0,1]).

    Returns:
        PSNR value (scalar tensor)
    """
    # Clamp for numerical stability (avoid log(0))
    mse = torch.clamp(mse, min=1e-8)
    return -10.0 * torch.log10(mse)


# -----------------------------------------------------------------------------
# Background compositing helper
# -----------------------------------------------------------------------------

def composite_on_bg(
    pred_rgb: torch.Tensor,
    bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha: Optional[torch.Tensor] = None,
    premultiplied: bool = True,
) -> torch.Tensor:
    """
    Composite predicted colors onto a solid background.

    Args:
        pred_rgb: (N,3) in [0,1]. If alpha is None, returns clamped pred.
        bg:       background color tuple in [0,1].
        alpha:    (N,) or (N,1) in [0,1], optional.
        premultiplied: if True, pred_rgb is assumed premultiplied by alpha;
                       else, pred_rgb is straight (unassociated) color.

    Returns:
        (N,3) composited color in [0,1]
    """
    # Fast path: no alpha provided → nothing to composite
    if alpha is None:
        return pred_rgb.clamp(0.0, 1.0)

    if alpha.dim() == 1:
        alpha = alpha[:, None]  # (N,1)

    bg_t = pred_rgb.new_tensor(bg)[None, :]  # (1,3)

    if premultiplied:
        # C = C_premult + (1 - a) * Bg
        out = pred_rgb + (1.0 - alpha) * bg_t
    else:
        # C = a * C_straight + (1 - a) * Bg
        out = alpha * pred_rgb + (1.0 - alpha) * bg_t

    return out.clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# Top-level orchestrator
# -----------------------------------------------------------------------------

def compute_losses(
    batch: Dict[str, torch.Tensor],          # expects: "rgb" (N,3) and optional "mask" (N,) or (N,1)
    render_out: Dict[str, torch.Tensor],     # expects: "rgb" (N,3) and optional "alpha" (N,) or (N,1)
    cfg: Dict[str, Any],                     # e.g. {"w_l2":1.0, "use_bg":False, "bg":(1,1,1), "premultiplied":True}
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes total training loss and logging metrics.
    Current recipe (simple):
      - L2 (MSE) on RGB (optionally masked)
      - PSNR metric from the same MSE
    """
    if "rgb" not in batch or "rgb" not in render_out:
        raise KeyError("compute_losses expects 'rgb' in both batch and render_out")

    pred_rgb: torch.Tensor = render_out["rgb"]
    gt_rgb: torch.Tensor = batch["rgb"]
    mask: Optional[torch.Tensor] = batch.get("mask", None)
    alpha: Optional[torch.Tensor] = render_out.get("alpha", None)

    # Optional background composite (e.g., white background assumption)
    use_bg = bool(cfg.get("use_bg", False))
    if use_bg:
        bg = tuple(cfg.get("bg", (1.0, 1.0, 1.0)))
        premult = bool(cfg.get("premultiplied", True))
        pred_rgb = composite_on_bg(pred_rgb, bg=bg, alpha=alpha, premultiplied=premult)

    # Core photometric loss (MSE); returns scalar (mean over valid channels)
    w_l2 = float(cfg.get("w_l2", 1.0))
    mse = rgb_l2(pred_rgb, gt_rgb, mask=mask)
    l_rgb = w_l2 * mse

    total = l_rgb

    # Metrics (detach so logs don't keep graph)
    metrics = {
        "l2": float(mse.detach().item()),
        "psnr": float(psnr(mse.detach()).item()),
    }

    return total, metrics