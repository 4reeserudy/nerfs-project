# nerf/nerf3d/viz/visualize.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


# -----------------------------------------------------------------------------
# Conversions
# -----------------------------------------------------------------------------

def to_cpu_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Tensor → numpy (CPU), no grad."""
    if isinstance(x, torch.Tensor):
        x = x.detach().to("cpu")
        return x.numpy()
    return x


def minmax_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(H,W) or (H,W,1) → [0,1]."""
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    rng = max(mx - mn, eps)
    return (x - mn) / rng


# -----------------------------------------------------------------------------
# Colorization / formatting
# -----------------------------------------------------------------------------

def colorize_depth(depth: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """(H,W) float → (H,W,3) uint8 via colormap."""
    d = minmax_normalize(depth)
    cm = plt.get_cmap(cmap)
    rgba = cm(d)  # (H,W,4)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce to (H,W,3) uint8 if in [0,1] float or already uint8."""
    arr = img
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def to_image_grid(
    images: List[np.ndarray],
    pad: int = 8,
    bg: Tuple[int, int, int] = (20, 20, 20)
) -> np.ndarray:
    """Horizontally tile images (same H), returns uint8."""
    if not images:
        raise ValueError("to_image_grid: empty images list")
    imgs = [ _ensure_uint8_rgb(im) for im in images ]
    H = imgs[0].shape[0]
    for im in imgs:
        if im.shape[0] != H:
            raise ValueError("All images must share the same height for tiling.")
    widths = [im.shape[1] for im in imgs]
    total_w = sum(widths) + pad * (len(imgs) + 1)
    total_h = H + 2 * pad
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[...] = np.array(bg, dtype=np.uint8)
    x = pad
    for im in imgs:
        canvas[pad:pad+H, x:x+im.shape[1]] = im
        x += im.shape[1] + pad
    return canvas


def side_by_side(
    img_left: np.ndarray,
    img_right: np.ndarray,
    labels: Tuple[str, str] = ("pred", "gt"),
    pad: int = 8
) -> np.ndarray:
    """Two-panel comparison."""
    L = _ensure_uint8_rgb(img_left)
    R = _ensure_uint8_rgb(img_right)
    if L.shape[0] != R.shape[0]:
        # match height of left by letterboxing right (or resize)
        raise ValueError("side_by_side expects same height images; resize beforehand.")
    grid = to_image_grid([L, R], pad=pad)
    # simple title bars with matplotlib (optional). Keep minimal: skip text drawing here.
    return grid


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

def save_png(img: np.ndarray, path: Path) -> None:
    """Write uint8 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_ensure_uint8_rgb(img)).save(path)


def _reshape_rgb_flat(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.reshape(H, W, 3)
    if arr.ndim == 3 and arr.shape[:2] == (H, W):
        return arr
    raise ValueError(f"Unexpected RGB shape for reshape: {arr.shape} with hw={hw}")


def _reshape_scalar_flat(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    if arr.ndim == 1 and arr.shape[0] == H * W:
        return arr.reshape(H, W)
    if arr.ndim == 2 and arr.shape == (H, W):
        return arr
    raise ValueError(f"Unexpected scalar shape for reshape: {arr.shape} with hw={hw}")


def save_rgb(
    rgb: torch.Tensor | np.ndarray,
    hw: Tuple[int, int],
    path: Path
) -> None:
    """(H*W,3) or (H,W,3) → (H,W,3) uint8 → PNG."""
    arr = to_cpu_np(rgb)
    arr = _reshape_rgb_flat(arr, hw)
    save_png(arr, path)


def save_depth(
    depth: torch.Tensor | np.ndarray,
    hw: Tuple[int, int],
    path: Path,
    cmap: str = "magma"
) -> None:
    """(H*W,) → (H,W) → colorize → PNG."""
    d = to_cpu_np(depth)
    d = _reshape_scalar_flat(d, hw)
    rgb = colorize_depth(d, cmap=cmap)
    save_png(rgb, path)


def save_acc(
    acc: torch.Tensor | np.ndarray,
    hw: Tuple[int, int],
    path: Path
) -> None:
    """(H*W,) opacity map → grayscale PNG."""
    a = to_cpu_np(acc)
    a = _reshape_scalar_flat(a, hw)
    a01 = minmax_normalize(a)
    gray = (a01 * 255.0).astype(np.uint8)
    gray3 = np.stack([gray, gray, gray], axis=-1)
    save_png(gray3, path)


# -----------------------------------------------------------------------------
# Logs / curves
# -----------------------------------------------------------------------------

def plot_psnr_curve(
    iters: List[int] | np.ndarray,
    vals: List[float] | np.ndarray,
    title: str,
    out_path: Path
) -> None:
    """Simple PSNR vs iters plot."""
    it = np.asarray(iters)
    va = np.asarray(vals, dtype=np.float32)
    plt.figure(figsize=(6.5, 4))
    plt.plot(it, va)
    plt.xlabel("Iterations")
    plt.ylabel("PSNR (dB)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Loading helpers (optional)
# -----------------------------------------------------------------------------

def load_render_npz(path: Path) -> Dict[str, Any]:
    """Load dict with keys like rgb, depth, acc (numpy)."""
    data = np.load(path, allow_pickle=False)
    out: Dict[str, Any] = {}
    for k in data.files:
        out[k] = data[k]
    return out


def reshape_flat(img_flat: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """(H*W,3)→(H,W,3) or (H*W,)→(H,W)."""
    H, W = hw
    if img_flat.ndim == 2 and img_flat.shape[1] == 3:
        return img_flat.reshape(H, W, 3)
    if img_flat.ndim == 1:
        return img_flat.reshape(H, W)
    if img_flat.ndim == 3 and img_flat.shape[:2] == (H, W):
        return img_flat
    raise ValueError(f"reshape_flat: unexpected shape {img_flat.shape} for hw={hw}")


# -----------------------------------------------------------------------------
# High-level convenience
# -----------------------------------------------------------------------------

def save_pred_depth_acc_triplet(
    render_out: Dict[str, torch.Tensor | np.ndarray],
    hw: Tuple[int, int],
    out_dir: Path,
    stem: str
) -> Dict[str, Path]:
    """
    Save rgb/depth/acc with shared stem; returns paths.
    Expects keys: 'rgb' (H*W,3), 'depth' (H*W,), 'acc' (H*W,)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p_rgb = out_dir / f"{stem}_rgb.png"
    p_d   = out_dir / f"{stem}_depth.png"
    p_a   = out_dir / f"{stem}_acc.png"
    save_rgb(render_out["rgb"], hw, p_rgb)
    if "depth" in render_out:
        save_depth(render_out["depth"], hw, p_d)
    if "acc" in render_out:
        save_acc(render_out["acc"], hw, p_a)
    return {"rgb": p_rgb, "depth": p_d, "acc": p_a}


def save_comparison(
    pred_rgb: torch.Tensor | np.ndarray,
    gt_rgb: torch.Tensor | np.ndarray,
    hw: Tuple[int, int],
    out_path: Path,
    labels: Tuple[str, str] = ("pred", "gt")
) -> None:
    """Side-by-side PNG."""
    pr = to_cpu_np(pred_rgb)
    gt = to_cpu_np(gt_rgb)
    pr = _reshape_rgb_flat(pr, hw)
    gt = _reshape_rgb_flat(gt, hw)
    grid = side_by_side(pr, gt, labels=labels, pad=8)
    save_png(grid, out_path)
