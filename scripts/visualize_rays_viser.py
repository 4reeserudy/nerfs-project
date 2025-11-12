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
    """Fallback: build K from scalar or (fx, fy) focal."""
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


def align_to_y_up(c2w: np.ndarray) -> np.ndarray:
    """
    Convert CV convention (x right, y down, z out) to y-up.
    Apply Rx(pi): diag(1, -1, -1).
    """
    R_align = np.array([[1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]], dtype=np.float32)
    out = c2w.copy()
    out[:3, :3] = R_align @ c2w[:3, :3]
    out[:3, 3]  = (R_align @ c2w[:3, 3])
    return out


def estimate_near_far_from_cameras(c2ws: np.ndarray) -> Tuple[float, float]:
    """
    Heuristic near/far based on camera positions.
    Works better than hard-coded [2,6] when scene scale is unknown.
    """
    cams = c2ws[:, :3, 3]
    center = np.median(cams, axis=0)
    dists = np.linalg.norm(cams - center, axis=1)
    med = float(np.median(dists) + 1e-6)
    near = max(0.05 * med, 1e-3)
    far = 3.0 * med
    if far < near * 2:
        far = near * 2.0
    return near, far


# ---------- Scene builders ----------

def add_camera_frustum(
    server: viser.ViserServer,
    name: str,
    c2w: np.ndarray,
    fov_x: float,
    aspect: float,
    scale: float,
    image_rgb_uint8: Optional[np.ndarray] = None,
    color=(1.0, 1.0, 1.0),
) -> None:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    q = viser.transforms.SO3.from_matrix(R.astype(np.float32)).wxyz
    server.scene.add_camera_frustum(
        name,
        fov=fov_x,
        aspect=aspect,
        scale=scale,
        wxyz=q,
        position=t.astype(np.float32),
        image=image_rgb_uint8,
        color=color,
    )


def downscale_image_uint8(img: np.ndarray, max_long_edge: Optional[int]) -> np.ndarray:
    """CPU-light downscale for frustum textures only."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if max_long_edge is None:
        return img
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / float(long_edge)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    # Use numpy slicing for nearest-neighbor-like quick shrink if scale small
    try:
        import cv2  # use if available
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        ys = (np.linspace(0, h - 1, new_h)).astype(np.int64)
        xs = (np.linspace(0, w - 1, new_w)).astype(np.int64)
        return img[ys][:, xs]


# ---------- Orchestrator ----------

def visualize_rays_viser(
    npz_path: Path,
    split: str = "train",
    num_rays: int = 200,
    from_image: Optional[int] = None,
    scale_frustum: float = 0.06,
    share: bool = True,
    stride_cameras: int = 4,
    n_samples_along: int = 48,
    near: float = -1.0,
    far: float = -1.0,
    seed: Optional[int] = 0,
    skip_uv_check: bool = False,
    max_long_edge: Optional[int] = 640,
    no_images: bool = False,
    align_y_up_flag: bool = True,
    max_points_vis: int = 200_000,
) -> None:
    raw = load_npz(npz_path)

    img_key = f"images_{split}"
    c2w_key = f"c2ws_{split}"
    if img_key not in raw or c2w_key not in raw:
        raise KeyError(f"NPZ must contain '{img_key}' and '{c2w_key}'.")

    images = raw[img_key]                    # (N,H,W,3) uint8 or float
    c2ws   = raw[c2w_key].astype(np.float32) # (N,4,4)

    # Optional axis alignment to match your poses viewer
    if align_y_up_flag:
        c2ws = np.stack([align_to_y_up(c) for c in c2ws], axis=0).astype(np.float32)

    N, H, W, _ = images.shape

    # Intrinsics for FOV (shared)
    if "K" in raw:
        Kraw = raw["K"]
        if Kraw.ndim == 2:
            K = Kraw.astype(np.float32)
        elif Kraw.ndim == 3:
            K = np.mean(Kraw.astype(np.float32), axis=0)
        else:
            raise ValueError("Unsupported K shape in NPZ.")
    elif "focal" in raw:
        K = compute_intrinsics_from_focal(raw["focal"], W, H)
    else:
        raise KeyError("NPZ must contain 'K' or 'focal' for intrinsics.")

    # RaysData (keeps full-res for ray math; we only downscale *display* textures)
    rays_data = RaysData(images, K, c2ws)

    # Optional UV sanity check
    if not skip_uv_check:
        uvs_start, uvs_end = 0, min(H * W, 30_000)
        uv = rays_data.uvs[uvs_start:uvs_end]  # (xy)
        ref = images[0, uv[:, 1], uv[:, 0]]
        if ref.dtype == np.uint8:
            ref = ref.astype(np.float32) / 255.0
        np.testing.assert_allclose(
            ref,
            rays_data.pixels[uvs_start:uvs_end],
            rtol=0.0,
            atol=(1.0 / 255.0 + 1e-6),
        )

    # RNG + ray selection
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    rays_o, rays_d, _ = rays_data.sample_rays(B=num_rays, from_image=from_image, rng=rng)

    # Auto near/far if not set
    if near <= 0 or far <= 0:
        near_est, far_est = estimate_near_far_from_cameras(c2ws)
        near = near if near > 0 else near_est
        far  = far  if far  > 0 else far_est

    # Sample 3D points along rays
    pts = sample_along_rays(
        rays_o, rays_d,
        n_samples=n_samples_along,
        near=near, far=far,
        perturb=True,
        rng=rng
    ).astype(np.float32)  # (B,S,3)

    # Subsample points for display if needed
    total_pts = pts.shape[0] * pts.shape[1]
    if total_pts > max_points_vis:
        flat = pts.reshape(-1, 3)
        take = rng.choice(total_pts, size=max_points_vis, replace=False)
        pts = flat[take].reshape(-1, 1, 3)  # keep shape compatible below

    # Viser setup
    fov_x, aspect = intrinsics_to_fov_aspect(K, W, H)
    server = viser.ViserServer(share=share)
    try:
        server.gui.configure_theme(dark=True)
        server.scene.set_background_color((0.0, 0.0, 0.0))
    except Exception:
        pass

    # Camera frusta (optionally downscaled textures)
    for i in range(0, N, max(1, int(stride_cameras))):
        img_disp = None
        if not no_images:
            img_disp = downscale_image_uint8(images[i], max_long_edge)
        add_camera_frustum(
            server,
            name=f"/cameras/{i}",
            c2w=c2ws[i],
            fov_x=fov_x,
            aspect=aspect,
            scale=scale_frustum,
            image_rgb_uint8=img_disp,
            color=(1.0, 1.0, 1.0),
        )

    # Ray polylines (short)
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        P0 = o
        P1 = o + d * (0.5 * (near + far))
        server.scene.add_spline_catmull_rom(f"/rays/{i}", positions=np.stack([P0, P1], axis=0))

    # Sample points cloud (gray)
    server.scene.add_point_cloud(
        "/samples",
        colors=np.full_like(pts, 0.6, dtype=np.float32).reshape(-1, 3),
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
    ap.add_argument("--num_rays", type=int, default=200)
    ap.add_argument("--from_image", type=int, default=None,
                    help="If set, sample rays only from this image index.")
    ap.add_argument("--scale_frustum", type=float, default=0.06)
    ap.add_argument("--stride_cameras", type=int, default=4,
                    help="Plot every Nth frustum to declutter.")
    ap.add_argument("--n_samples_along", type=int, default=48)
    ap.add_argument("--near", type=float, default=-1.0,
                    help="<=0 uses auto near based on camera spread.")
    ap.add_argument("--far", type=float, default=-1.0,
                    help="<=0 uses auto far based on camera spread.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_uv_check", action="store_true",
                    help="Skip UV/pixel alignment assertion.")
    ap.add_argument("--no_share", action="store_true")
    ap.add_argument("--max_long_edge", type=int, default=640,
                    help="Downscale camera textures for streaming speed. Use 0 or negative to disable.")
    ap.add_argument("--no_images", action="store_true", help="Do not attach image textures to frusta.")
    ap.add_argument("--no_align_y_up", action="store_true", help="Keep CV axes (do not flip to y-up).")
    ap.add_argument("--max_points_vis", type=int, default=200_000,
                    help="Cap number of visualized 3D samples.")
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
        max_long_edge=(None if (args.max_long_edge is not None and args.max_long_edge <= 0) else args.max_long_edge),
        no_images=args.no_images,
        align_y_up_flag=not args.no_align_y_up,
        max_points_vis=args.max_points_vis,
    )


if __name__ == "__main__":
    main()
