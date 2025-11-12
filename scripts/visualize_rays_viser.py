# scripts/visualize_rays_viser.py
from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import viser

from nerf.nerf3d.rays.rays_data_for_vis import RaysData, sample_along_rays


# ---------- I/O ----------

def load_npz(npz_path: Path) -> Dict[str, Any]:
    data = np.load(str(npz_path), allow_pickle=True)
    return {k: data[k] for k in data.files}

def compute_intrinsics_from_focal(focal, W: int, H: int) -> np.ndarray:
    """Fallback: build K from scalar or (fx,fy) focal."""
    if np.isscalar(focal) or (isinstance(focal, np.ndarray) and focal.ndim == 0):
        fx = fy = float(focal)
    else:
        focal = np.asarray(focal).reshape(-1)
        if focal.size == 2:
            fx, fy = float(focal[0]), float(focal[1])
        else:
            f = float(np.mean(focal))
            fx = fy = f
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


# ---------- Math helpers ----------

def intrinsics_to_fov_aspect(K: np.ndarray, W: int, H: int) -> Tuple[float, float]:
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
    color=(1.0, 1.0, 1.0),  # white frusta on black bg
) -> None:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    q = viser.transforms.SO3.from_matrix(R).wxyz
    # use scene API (consistent with your poses script)
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

def visualize_rays_viser(
    npz_path: Path,
    split: str = "train",
    num_rays: int = 100,
    from_image: Optional[int] = None,  # if set, sample only from this view index
    scale_frustum: float = 0.15,
    share: bool = True,
    stride_cameras: int = 1,           # plot every Nth camera to declutter
    n_samples_along: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    seed: Optional[int] = 42,
    skip_uv_check: bool = False,
) -> None:
    raw = load_npz(npz_path)

    img_key = f"images_{split}"
    c2w_key = f"c2ws_{split}"
    if img_key not in raw or c2w_key not in raw:
        raise KeyError(f"NPZ must contain '{img_key}' and '{c2w_key}'.")

    images = raw[img_key]                    # (N,H,W,3) uint8 or float
    c2ws   = raw[c2w_key].astype(np.float32) # (N,4,4)

    N, H, W, _ = images.shape

    # get/build intrinsics K (shared 3x3 for FOV; RaysData accepts shared or per-view)
    if "K" in raw:
        Kraw = raw["K"]
        if Kraw.ndim == 2:
            K = Kraw.astype(np.float32)
        elif Kraw.ndim == 3:
            K = np.mean(Kraw.astype(np.float32), axis=0)  # shared for FOV
        else:
            raise ValueError("Unsupported K shape in NPZ.")
    elif "focal" in raw:
        K = compute_intrinsics_from_focal(raw["focal"], W, H)
    else:
        raise KeyError("NPZ must contain 'K' or 'focal' for intrinsics.")

    # rays dataset
    rays_data = RaysData(images, K, c2ws)

    # optional UV sanity check (robust to uint8 vs float32)
    if not skip_uv_check:
        uvs_start, uvs_end = 0, min(H * W, 40_000)
        uv = rays_data.uvs[uvs_start:uvs_end]  # (xy)
        ref = images[0, uv[:, 1], uv[:, 0]]
        if ref.dtype == np.uint8:
            ref = ref.astype(np.float32) / 255.0
        np.testing.assert_allclose(
            ref,
            rays_data.pixels[uvs_start:uvs_end],
            rtol=0.0,
            atol=(1.0 / 255.0 + 1e-7),
        )

    # RNG + ray sample selection
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    rays_o, rays_d, _ = rays_data.sample_rays(
        B=num_rays,
        from_image=from_image,
        rng=rng
    )

    # sample 3D points along rays
    pts = sample_along_rays(
        rays_o, rays_d,
        n_samples=n_samples_along,
        near=near, far=far,
        perturb=True,
        rng=rng
    )  # (B,S,3)

    # Viser setup (black bg, white frusta)
    fov_x, aspect = intrinsics_to_fov_aspect(K, W, H)
    server = viser.ViserServer(share=share)
    try:
        server.gui.configure_theme(dark=True)
    except Exception:
        pass
    try:
        server.scene.set_background_color((0.0, 0.0, 0.0))
    except Exception:
        pass

    # Add camera frusta (optionally subsampled)
    for i in range(0, N, max(1, int(stride_cameras))):
        img = images[i]
        if img.dtype != np.uint8:
            img_disp = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_disp = img
        add_camera_frustum(
            server,
            name=f"/cameras/{i}",
            c2w=c2ws[i],
            fov_x=fov_x,
            aspect=aspect,
            scale=scale_frustum,
            image_rgb=img_disp,
            color=(1.0, 1.0, 1.0),
        )

    # Add ray splines
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        positions = np.stack([o, o + d * (far + 0.5 * (far - near))], axis=0)
        server.scene.add_spline_catmull_rom(f"/rays/{i}", positions=positions)

    # Add sampled points as a point cloud (black)
    server.scene.add_point_cloud(
        "/samples",
        colors=np.zeros_like(pts, dtype=np.float32).reshape(-1, 3),
        points=pts.reshape(-1, 3),
        point_size=0.03,
    )

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize cameras, sampled rays, and points in Viser (black bg, white frusta)."
    )
    ap.add_argument("--npz", type=Path, required=True,
                    help="Path to dataset NPZ (expects images_{split}, c2ws_{split}, and K or focal).")
    ap.add_argument("--split", type=str, default="train", choices=["train","val","test"])
    ap.add_argument("--num_rays", type=int, default=100, help="How many rays to draw.")
    ap.add_argument("--from_image", type=int, default=None,
                    help="If set, sample rays only from this image index.")
    ap.add_argument("--scale_frustum", type=float, default=0.15)
    ap.add_argument("--stride_cameras", type=int, default=1,
                    help="Plot every Nth frustum to declutter.")
    ap.add_argument("--n_samples_along", type=int, default=64)
    ap.add_argument("--near", type=float, default=2.0)
    ap.add_argument("--far", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_uv_check", action="store_true",
                    help="Skip UV/pixel alignment assertion.")
    ap.add_argument("--no_share", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    visualize_rays_viser(
        npz_path=args.npz,
        split=args.split,
        num_rays=args.num_rays,
        from_image=args.from_image,
        scale_frustum=args.scale_frustum,
        share=not args.no_share,
        stride_cameras=args.stride_cameras,
        n_samples_along=args.n_samples_along,
        near=args.near,
        far=args.far,
        seed=args.seed,
        skip_uv_check=args.skip_uv_check,
    )

if __name__ == "__main__":
    main()
