# nerf/nerf3d/datasets/lego_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np


def _build_shared_K(focal: float, H: int, W: int) -> np.ndarray:
    """Make a (3,3) pinhole intrinsic with fx=fy=focal and principal point at (W/2, H/2)."""
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = float(focal)
    K[1, 1] = float(focal)
    K[0, 2] = float(W) * 0.5
    K[1, 2] = float(H) * 0.5
    return K


def _to_float01(imgs: np.ndarray) -> np.ndarray:
    """Ensure images are float32 in [0,1]. Accepts uint8 [0,255] or float already in [0,1]."""
    if imgs.dtype == np.uint8:
        return (imgs.astype(np.float32) / 255.0)
    imgs = imgs.astype(np.float32, copy=False)
    # best-effort clamp if slightly outside
    return np.clip(imgs, 0.0, 1.0, out=imgs)


def load_lego200(npz_path: Path | str) -> Dict[str, Any]:
    """
    Thin adapter for lego_200x200.npz:
      expects keys:
        - images_train: (Nt,H,W,3) uint8 or float
        - images_val:   (Nv,H,W,3)
        - c2ws_train:   (Nt,4,4)
        - c2ws_val:     (Nv,4,4)
        - c2ws_test:    (Nq,4,4)
        - focal:        float
      returns dict with:
        images_train/val (float32 [0,1]),
        c2ws_train/val/test (float32),
        K (3,3 float32),
        H, W (ints),
        focal (float)
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        # required
        images_train = z["images_train"]
        images_val   = z["images_val"]
        c2ws_train   = z["c2ws_train"]
        c2ws_val     = z["c2ws_val"]
        c2ws_test    = z["c2ws_test"]
        focal        = float(z["focal"])

    # shapes
    if images_train.ndim != 4 or images_train.shape[-1] != 3:
        raise ValueError(f"images_train must be (N,H,W,3); got {images_train.shape}")
    if images_val.ndim != 4 or images_val.shape[-1] != 3:
        raise ValueError(f"images_val must be (N,H,W,3); got {images_val.shape}")
    for name, arr in [("c2ws_train", c2ws_train), ("c2ws_val", c2ws_val), ("c2ws_test", c2ws_test)]:
        if arr.ndim != 3 or arr.shape[1:] != (4, 4):
            raise ValueError(f"{name} must be (N,4,4); got {arr.shape}")

    H, W = int(images_train.shape[1]), int(images_train.shape[2])
    K = _build_shared_K(focal, H, W)

    out: Dict[str, Any] = {
        "images_train": _to_float01(images_train),
        "images_val":   _to_float01(images_val),
        "c2ws_train":   c2ws_train.astype(np.float32, copy=False),
        "c2ws_val":     c2ws_val.astype(np.float32, copy=False),
        "c2ws_test":    c2ws_test.astype(np.float32, copy=False),
        "K":            K,
        "H":            H,
        "W":            W,
        "focal":        float(focal),
        # defaults for downstream (can be overridden by caller)
        "near":         0.1,
        "far":          2.0,
        "convention":   "c2w_opengl_right_handed",
        "color_space":  "srgb_[0,1]",
    }
    return out
