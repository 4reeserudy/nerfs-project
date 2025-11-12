# scripts/visualize_poses_viser.py
from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import viser


# ---------- Math helpers ----------

def align_to_y_up(c2w: np.ndarray) -> np.ndarray:
    """
    Convert CV convention (x right, y down, z forward) to a y-up world.
    Applies Rx(pi): diag(1, -1, -1) to both rotation and translation.
    """
    R_align = np.array([[1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]], dtype=np.float64)
    out = c2w.copy()
    out[:3, :3] = R_align @ c2w[:3, :3]
    out[:3, 3] = R_align @ c2w[:3, 3]
    return out


def intrinsics_from_focal(focal: float, W: int, H: int) -> Tuple[float, float]:
    """Return (fov_x, aspect) given scalar focal and image size."""
    fov_x = 2.0 * np.arctan2(W / 2.0, focal)
    aspect = float(W) / float(H)
    return fov_x, aspect


# ---------- Scene builders ----------

def add_camera_frustum(
    server: viser.ViserServer,
    name: str,
    c2w: np.ndarray,
    fov_x: float,
    aspect: float,
    scale: float,
    image_rgb: Optional[np.ndarray] = None,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),  # white frusta
) -> None:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    q = viser.transforms.SO3.from_matrix(R).wxyz
    server.scene.add_camera_frustum(
        name,
        fov=fov_x,
        aspect=aspect,
        scale=scale,
        wxyz=q,
        position=t,
        image=image_rgb,
        color=color,
    )


# ---------- Orchestrator ----------

def visualize_npz_viser(
    npz_path: Path,
    split: str = "all",                 # "train" | "val" | "test" | "all"
    scale_frustum: float = 0.02,
    share: bool = True,
    up_axis: str = "y",                 # "y" (y-up) or "cv" (no flip)
    bg: str = "black",                  # "black" | "white"
) -> None:
    data = np.load(str(npz_path))

    # Pull arrays (handle absence gracefully)
    images_train = data.get("images_train")
    images_val   = data.get("images_val")
    c2ws_train   = data.get("c2ws_train")
    c2ws_val     = data.get("c2ws_val")
    c2ws_test    = data.get("c2ws_test")
    focal        = float(data["focal"]) if "focal" in data else None

    # Determine a reference image size (prefer train, then val)
    W = H = None
    ref_images = None
    if images_train is not None and len(images_train) > 0:
        ref_images = images_train
    elif images_val is not None and len(images_val) > 0:
        ref_images = images_val

    if ref_images is not None and ref_images.size > 0:
        H, W = int(ref_images.shape[1]), int(ref_images.shape[2])

    if focal is None or W is None or H is None:
        raise ValueError(
            "Could not infer focal or image size from NPZ. "
            "Make sure your NPZ contains 'focal' and at least one of images_train/images_val."
        )

    fov_x, aspect = intrinsics_from_focal(focal, W, H)

    # Start viser server
    server = viser.ViserServer(share=share)
    try:
        server.gui.configure_theme(dark=(bg == "black"))
    except Exception:
        pass
    try:
        server.scene.set_background_color((0.0, 0.0, 0.0) if bg == "black" else (1.0, 1.0, 1.0))
    except Exception:
        pass

    # Helper to optionally apply up-axis conversion
    def maybe_align(c2w: np.ndarray) -> np.ndarray:
        return align_to_y_up(c2w) if up_axis.lower() == "y" else c2w

    # Add TRAIN frusta (+ images)
    if split in ("train", "all"):
        if c2ws_train is not None and len(c2ws_train) > 0:
            n = len(c2ws_train)
            for i in range(n):
                img = None
                if images_train is not None and len(images_train) > i:
                    img = images_train[i]
                add_camera_frustum(
                    server,
                    name=f"/train/{i:04d}",
                    c2w=maybe_align(c2ws_train[i]),
                    fov_x=fov_x,
                    aspect=aspect,
                    scale=scale_frustum,
                    image_rgb=img,
                    color=(1.0, 1.0, 1.0),
                )

    # Add VAL frusta (+ images)
    if split in ("val", "all"):
        if c2ws_val is not None and len(c2ws_val) > 0:
            n = len(c2ws_val)
            for i in range(n):
                img = None
                if images_val is not None and len(images_val) > i:
                    img = images_val[i]
                add_camera_frustum(
                    server,
                    name=f"/val/{i:04d}",
                    c2w=maybe_align(c2ws_val[i]),
                    fov_x=fov_x,
                    aspect=aspect,
                    scale=scale_frustum,
                    image_rgb=img,
                    color=(0.6, 0.9, 1.0),   # slight tint to distinguish from train
                )

    # Add TEST frusta (no images in v2)
    if split in ("test", "all"):
        if c2ws_test is not None and len(c2ws_test) > 0:
            n = len(c2ws_test)
            for i in range(n):
                add_camera_frustum(
                    server,
                    name=f"/test/{i:04d}",
                    c2w=maybe_align(c2ws_test[i]),
                    fov_x=fov_x,
                    aspect=aspect,
                    scale=scale_frustum,
                    image_rgb=None,
                    color=(1.0, 0.6, 0.6),
                )

    # Keep the server alive
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize NPZ (v2) cameras in Viser (y-up default, white frusta on black bg)."
    )
    ap.add_argument("--npz", type=Path, required=True, help="e.g., data/bird/bird_dataset_v2.npz")
    ap.add_argument("--split", type=str, default="all",
                    choices=["train", "val", "test", "all"],
                    help="Which split(s) to draw.")
    ap.add_argument("--scale_frustum", type=float, default=0.02)
    ap.add_argument("--no_share", action="store_true", help="Disable Viser sharing.")
    ap.add_argument("--up_axis", type=str, default="y", choices=["y", "cv"],
                    help="'y' applies CV->y-up flip; 'cv' leaves as-is.")
    ap.add_argument("--bg", type=str, default="black", choices=["black", "white"])
    return ap.parse_args()


def main():
    args = parse_args()
    visualize_npz_viser(
        npz_path=args.npz,
        split=args.split,
        scale_frustum=args.scale_frustum,
        share=not args.no_share,
        up_axis=args.up_axis,
        bg=args.bg,
    )


if __name__ == "__main__":
    main()