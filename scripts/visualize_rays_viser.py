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


# ---------- Small helpers ----------

def intrinsics_to_fov_aspect(K: np.ndarray, W: int, H: int) -> Tuple[float, float]:
    fx = float(K[0, 0])
    fov_x = 2.0 * np.arctan2(W / 2.0, fx)
    aspect = float(W) / float(H)
    return fov_x, aspect


def parse_color(name: str) -> Tuple[float, float, float]:
    s = (name or "").lower()
    if s in ("black", "k"): return (0.0, 0.0, 0.0)
    if s in ("white", "w"): return (1.0, 1.0, 1.0)
    if s in ("gray", "grey"): return (0.5, 0.5, 0.5)
    if s.startswith("#") and len(s) == 7:
        r = int(s[1:3], 16) / 255.0
        g = int(s[3:5], 16) / 255.0
        b = int(s[5:7], 16) / 255.0
        return (r, g, b)
    return (0.0, 0.0, 0.0)  # default black


def auto_near_far(c2ws: np.ndarray) -> Tuple[float, float]:
    """
    Pick sensible near/far from camera positions (meters).
    near = 0.1 * median_radius, far = 3.0 * median_radius (clamped).
    """
    cams = c2ws[:, :3, 3]
    r = np.linalg.norm(cams - cams.mean(axis=0, keepdims=True), axis=1)
    med = float(np.median(r)) if r.size else 1.0
    near = max(0.05, 0.10 * med)
    far = max(near * 2.0, 3.00 * med)
    return near, far


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
    scale_frustum: float = 0.03,
    share: bool = True,
    stride_cameras: int = 1,           # plot every Nth camera to declutter
    n_samples_along: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    seed: Optional[int] = 42,
    skip_uv_check: bool = False,
    # v2 additions
    ray_color: str = "black",
    point_size: float = 0.02,
    no_points: bool = False,
    no_images: bool = False,
    auto_bounds_flag: bool = False,
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

    # near/far: auto if requested OR if user passes negative values (convenience)
    use_auto = auto_bounds_flag or (near < 0 or far < 0)
    if use_auto:
        near_auto, far_auto = auto_near_far(c2ws)
        near = near_auto
        far = far_auto

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
        if no_images:
            img_disp = None
        else:
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

    # Add ray splines (force single color)
    ray_rgb = parse_color(ray_color)
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        P0 = o
        P1 = o + d * (0.5 * (near + far))
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}",
            positions=np.stack([P0, P1], axis=0),
            color=ray_rgb,
        )

    # Add sampled points as a point cloud (usually black)
    if not no_points:
        server.scene.add_point_cloud(
            "/samples",
            colors=np.zeros_like(pts, dtype=np.float32).reshape(-1, 3),
            points=pts.reshape(-1, 3),
            point_size=float(point_size),
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
    ap.add_argument("--scale_frustum", type=float, default=0.03)
    ap.add_argument("--stride_cameras", type=int, default=1,
                    help="Plot every Nth frustum to declutter.")
    ap.add_argument("--n_samples_along", type=int, default=64)
    ap.add_argument("--near", type=float, default=2.0)
    ap.add_argument("--far", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_uv_check", action="store_true",
                    help="Skip UV/pixel alignment assertion.")
    ap.add_argument("--no_share", action="store_true")

    # v2 additions
    ap.add_argument("--ray_color", type=str, default="black",
                    help="Ray color: black/white/gray or hex like #ff8800")
    ap.add_argument("--point_size", type=float, default=0.02)
    ap.add_argument("--no_points", action="store_true")
    ap.add_argument("--no_images", action="store_true",
                    help="Do not texture frusta with images (faster/cleaner).")
    ap.add_argument("--auto_bounds", action="store_true",
                    help="Auto-pick near/far based on camera layout. Also triggers if near<0 or far<0.")
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
        ray_color=args.ray_color,
        point_size=args.point_size,
        no_points=args.no_points,
        no_images=args.no_images,
        auto_bounds_flag=args.auto_bounds,
    )

if __name__ == "__main__":
    main()
