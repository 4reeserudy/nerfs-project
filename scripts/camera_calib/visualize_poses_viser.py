# scripts/visualize_poses_viser.py
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import cv2
import viser


# ---------- I/O ----------

def load_poses_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if "K_new" not in data or "frames" not in data:
        raise ValueError("poses JSON must contain 'K_new' and 'frames'.")
    return data


def image_size_for(path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    h, w = img.shape[:2]
    return (w, h)


def load_image_rgb(path: Path, max_long_edge: Optional[int] = 1280) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if max_long_edge is not None:
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge > max_long_edge:
            scale = max_long_edge / float(long_edge)
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img


# ---------- Math helpers ----------

def rt_to_c2w(R: np.ndarray, t: np.ndarray, mm_to_m: bool = True) -> np.ndarray:
    """
    OpenCV gives T_cw = [R|t] (world->camera).
    We need camera->world (T_wc):
      T_wc = [[R^T | -R^T @ t],
              [0 0 0 1]]
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    Rt = R.T
    tw = -Rt @ t
    if mm_to_m:
        tw = tw * 1e-3
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rt
    T[:3, 3:4] = tw
    return T


def align_to_y_up(c2w: np.ndarray) -> np.ndarray:
    """
    Convert CV convention (x right, y down, z out) to y-up.
    Apply Rx(pi): diag(1, -1, -1).
    """
    R_align = np.array([[1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]], dtype=np.float64)
    out = c2w.copy()
    out[:3, :3] = R_align @ c2w[:3, :3]
    out[:3, 3]   = (R_align @ c2w[:3, 3])
    return out


def intrinsics_to_fov_aspect(K: np.ndarray, W: int, H: int) -> Tuple[float, float]:
    K = np.asarray(K, dtype=np.float64)
    fx = float(K[0, 0])
    fov_x = 2.0 * np.arctan2(W / 2.0, fx)
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
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),  # white frames
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

def visualize_poses_viser(
    poses_json: Path,
    images_dir: Path,
    scale_frustum: float = 0.02,
    share: bool = True,
    max_long_edge: Optional[int] = 1280,
) -> None:
    data = load_poses_json(poses_json)
    K_new = np.array(data["K_new"], dtype=np.float64)
    frames = data["frames"]

    # Determine image size from the first available frame
    W = H = None
    for f in frames:
        ipath = images_dir / f["file"]
        if ipath.exists():
            W, H = image_size_for(ipath)
            break
    if W is None or H is None:
        raise FileNotFoundError("Could not infer image size; none of the listed images were found.")

    fov_x, aspect = intrinsics_to_fov_aspect(K_new, W, H)

    server = viser.ViserServer(share=share)
    # Try to make background black / dark theme (best-effort).
    try:
        server.gui.configure_theme(dark=True)
    except Exception:
        pass
    try:
        server.scene.set_background_color((0.0, 0.0, 0.0))
    except Exception:
        pass

    # Add all camera frusta
    for f in frames:
        R = np.array(f["R"], dtype=np.float64)
        t = np.array(f["t"], dtype=np.float64)
        c2w = rt_to_c2w(R, t, mm_to_m=True)
        c2w = align_to_y_up(c2w)

        ipath = images_dir / f["file"]
        img_rgb = load_image_rgb(ipath, max_long_edge=max_long_edge) if ipath.exists() else None

        add_camera_frustum(
            server,
            name=f"/cameras/{Path(f['file']).stem}",
            c2w=c2w,
            fov_x=fov_x,
            aspect=aspect,
            scale=scale_frustum,
            image_rgb=img_rgb,
            color=(1.0, 1.0, 1.0),
        )

    # Keep the server alive
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Visualize PnP poses as camera frusta in Viser (y-up, white frusta, black bg).")
    ap.add_argument("--poses_json", type=Path, required=True,
                    help="e.g., data/object_3d/results/poses_pnp.json")
    ap.add_argument("--images_dir", type=Path, required=True,
                    help="e.g., data/object_3d/images")
    ap.add_argument("--scale_frustum", type=float, default=0.02)
    ap.add_argument("--no_share", action="store_true")
    ap.add_argument("--max_long_edge", type=int, default=1280,
                    help="Downscale images for faster streaming (None to disable).")
    return ap.parse_args()


def main():
    args = parse_args()
    visualize_poses_viser(
        poses_json=args.poses_json,
        images_dir=args.images_dir,
        scale_frustum=args.scale_frustum,
        share=not args.no_share,
        max_long_edge=args.max_long_edge,
    )


if __name__ == "__main__":
    main()
