# nerf/nerf3d/viz/video_render.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable, Optional
from pathlib import Path
import math
import subprocess

import numpy as np
import torch
from PIL import Image


# -----------------------------------------------------------------------------
# Orbit path construction
# -----------------------------------------------------------------------------

def estimate_orbit_center_radius(c2ws: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Args:
      c2ws: (N, 4, 4) OpenGL-style camera-to-world transforms
    Returns:
      center: (3,) mean of camera centers
      radius: median distance of centers to mean
    """
    assert c2ws.ndim == 3 and c2ws.shape[-2:] == (4, 4)
    centers = c2ws[..., :3, 3]                         # (N,3)
    center = centers.mean(dim=0)                        # (3,)
    dists = torch.linalg.norm(centers - center, dim=-1) # (N,)
    radius = float(torch.median(dists).item())
    return center, radius


def look_at(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Build OpenGL-style c2w given eye/center/up.
    z (forward) points from eye -> center (then normalized and negated for OpenGL).
    x = normalize(cross(up, z)); y = cross(z, x).
    """
    fwd = (center - eye)
    fwd = fwd / (torch.linalg.norm(fwd) + 1e-8)
    # OpenGL convention: camera looks along -Z in camera space; in world, z axis = +fwd
    z = fwd
    x = torch.linalg.cross(up, z)
    x = x / (torch.linalg.norm(x) + 1e-8)
    y = torch.linalg.cross(z, x)
    y = y / (torch.linalg.norm(y) + 1e-8)

    c2w = torch.eye(4, dtype=eye.dtype, device=eye.device)
    # Columns are world-space axes of the camera
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


def generate_orbit_c2ws(
    center: torch.Tensor,
    up: torch.Tensor,
    radius: float,
    n_frames: int,
    elevation_deg: float,
    sweep_deg: float = 360.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Circular path around 'center' with fixed elevation (degrees).
    Returns (n_frames, 4, 4) c2w transforms.
    """
    device = device or center.device
    phi = math.radians(elevation_deg)
    thetas = torch.linspace(0.0, math.radians(sweep_deg), n_frames, device=device)
    # Orbit points in world coords
    xs = radius * torch.cos(thetas) * torch.cos(torch.tensor(phi, device=device))
    ys = radius * torch.sin(torch.tensor(phi, device=device)).repeat(n_frames)
    zs = radius * torch.sin(thetas) * torch.cos(torch.tensor(phi, device=device))

    c2ws: List[torch.Tensor] = []
    for i in range(n_frames):
        eye = torch.stack([center[0] + xs[i], center[1] + ys[i], center[2] + zs[i]], dim=0)
        c2w = look_at(eye, center, up)
        c2ws.append(c2w)
    return torch.stack(c2ws, dim=0)  # (N,4,4)


# -----------------------------------------------------------------------------
# Ray generation (per-frame batch)
# -----------------------------------------------------------------------------

def _pixels_to_cam_dirs(H: int, W: int, fx: float, fy: float, cx: float, cy: float, device: torch.device) -> torch.Tensor:
    """
    Returns (H*W, 3) normalized camera-space directions with 0.5 pixel center offset.
    """
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xs = (xs + 0.5 - cx) / fx
    ys = (ys + 0.5 - cy) / fy
    zs = torch.ones_like(xs)
    dirs = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)  # (H*W,3)
    dirs = dirs / (torch.linalg.norm(dirs, dim=-1, keepdim=True) + 1e-8)
    return dirs


def pixels_to_rays(
    H: int, W: int,
    fx: float, fy: float, cx: float, cy: float,
    c2w: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-frame rays from a single c2w.
    Returns:
      rays_o: (H*W, 3)
      rays_d: (H*W, 3) normalized in world space
    """
    dirs_cam = _pixels_to_cam_dirs(H, W, fx, fy, cx, cy, device)  # (Npix,3)
    R = c2w[:3, :3]  # (3,3)
    t = c2w[:3, 3]   # (3,)
    rays_d = (dirs_cam @ R.T)  # (Npix,3)
    rays_d = rays_d / (torch.linalg.norm(rays_d, dim=-1, keepdim=True) + 1e-8)
    rays_o = t.expand_as(rays_d)
    return rays_o, rays_d


# -----------------------------------------------------------------------------
# Rendering loop
# -----------------------------------------------------------------------------

# Expected renderer signature (callable):
#   renderer(rays_o: (N,3), rays_d: (N,3), near: float, far: float, n_samples: int, chunk: int, white_bg: bool)
# -> returns dict with keys: 'rgb' (N,3) in [0,1], and optionally 'depth' (N,), 'acc' (N,)
RendererFn = Callable[[torch.Tensor, torch.Tensor, float, float, int, int, bool], Dict[str, torch.Tensor]]

def render_frames(
    renderer: RendererFn,
    c2ws: torch.Tensor,
    H: int, W: int,
    fx: float, fy: float, cx: float, cy: float,
    near: float, far: float, n_samples: int,
    chunk: int,
    device: torch.device,
    white_bg: bool = False,
) -> Dict[str, List[np.ndarray]]:
    """
    Returns per-frame numpy RGB (H,W,3) uint8, and optional depth/acc visualizations (arrays).
    """
    rgb_frames: List[np.ndarray] = []
    depth_frames: List[np.ndarray] = []
    acc_frames: List[np.ndarray] = []

    for i in range(c2ws.shape[0]):
        c2w = c2ws[i].to(device)
        rays_o, rays_d = pixels_to_rays(H, W, fx, fy, cx, cy, c2w, device)  # (Npix,3)

        # Chunked rendering
        Npix = rays_o.shape[0]
        out_rgb = torch.empty((Npix, 3), device=device, dtype=torch.float32)
        out_depth = torch.empty((Npix,), device=device, dtype=torch.float32)
        out_acc = torch.empty((Npix,), device=device, dtype=torch.float32)
        has_depth = False
        has_acc = False

        for s in range(0, Npix, chunk):
            e = min(s + chunk, Npix)
            r = renderer(rays_o[s:e], rays_d[s:e], near, far, n_samples, chunk, white_bg)
            out_rgb[s:e] = r["rgb"]
            if "depth" in r:
                out_depth[s:e] = r["depth"]; has_depth = True
            if "acc" in r:
                out_acc[s:e] = r["acc"]; has_acc = True

        # to images
        rgb_np = (out_rgb.clamp(0, 1).reshape(H, W, 3).detach().cpu().numpy() * 255.0).astype(np.uint8)
        rgb_frames.append(rgb_np)

        if has_depth:
            d = out_depth.reshape(H, W).detach().cpu().numpy()
            d = _colorize_depth_magma(d)  # (H,W,3) uint8
            depth_frames.append(d)
        if has_acc:
            a = out_acc.reshape(H, W).detach().cpu().numpy()
            a01 = (a - a.min()) / (a.max() - a.min() + 1e-8)
            g = (a01 * 255.0).astype(np.uint8)
            acc_frames.append(np.stack([g, g, g], axis=-1))

    out: Dict[str, List[np.ndarray]] = {"rgb": rgb_frames}
    if depth_frames:
        out["depth"] = depth_frames
    if acc_frames:
        out["acc"] = acc_frames
    return out


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def save_frames_png(frames: List[np.ndarray], out_dir: Path, stem: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i, img in enumerate(frames):
        p = out_dir / f"{stem}_{i:04d}.png"
        Image.fromarray(img).save(p)
        paths.append(p)
    return paths


def encode_video_ffmpeg(frames_dir: Path, pattern: str, fps: int, out_mp4: Path) -> None:
    """
    Encodes PNG frames into an MP4 using ffmpeg if available.
    pattern example: 'rgb_%04d.png'
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / pattern),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        # Silently skip if ffmpeg not available; user can encode later
        pass


# -----------------------------------------------------------------------------
# High-level one-call API
# -----------------------------------------------------------------------------

def render_orbit_video(
    renderer: RendererFn,
    train_c2ws: torch.Tensor,            # (N,4,4) from training set
    intr: Dict[str, float],              # keys: fx, fy, cx, cy
    H: int, W: int,
    n_frames: int = 60,
    fps: int = 15,
    elevation_deg: float = 30.0,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
    chunk: int = 32768,
    device: Optional[torch.device] = None,
    out_dir: Path = Path("results/nerf3d/video"),
    stem: str = "orbit",
    radius: Optional[float] = None,
    white_bg: bool = False,              # False = black background
) -> Dict[str, Path]:
    """
    Builds an orbit path (auto center + radius unless provided), renders frames, saves PNGs,
    and encodes an MP4 (if ffmpeg present).

    Returns:
      dict with keys: 'rgb_dir', 'rgb_mp4', optional 'depth_dir', 'acc_dir'
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)

    # Center + radius
    center, auto_radius = estimate_orbit_center_radius(train_c2ws.to(device))
    R = float(radius if (radius is not None) else auto_radius)

    # Path
    c2ws_orbit = generate_orbit_c2ws(center, up, R, n_frames, elevation_deg, sweep_deg=360.0, device=device)

    # Render
    frames = render_frames(
        renderer=renderer,
        c2ws=c2ws_orbit,
        H=H, W=W,
        fx=float(intr["fx"]), fy=float(intr["fy"]), cx=float(intr["cx"]), cy=float(intr["cy"]),
        near=near, far=far, n_samples=n_samples,
        chunk=chunk, device=device, white_bg=white_bg,
    )

    # Save PNGs
    out_paths: Dict[str, Path] = {}
    rgb_dir = out_dir / "rgb"
    rgb_paths = save_frames_png(frames["rgb"], rgb_dir, f"{stem}")
    out_paths["rgb_dir"] = rgb_dir

    # Optional depth/acc frames
    if "depth" in frames:
        depth_dir = out_dir / "depth"
        save_frames_png(frames["depth"], depth_dir, f"{stem}_depth")
        out_paths["depth_dir"] = depth_dir
    if "acc" in frames:
        acc_dir = out_dir / "acc"
        save_frames_png(frames["acc"], acc_dir, f"{stem}_acc")
        out_paths["acc_dir"] = acc_dir

    # MP4 (RGB)
    mp4_path = out_dir / f"{stem}.mp4"
    encode_video_ffmpeg(rgb_dir, f"{stem}_%04d.png", fps=fps, out_mp4=mp4_path)
    out_paths["rgb_mp4"] = mp4_path
    return out_paths


# -----------------------------------------------------------------------------
# Small internal helper
# -----------------------------------------------------------------------------

def _colorize_depth_magma(depth_hw: np.ndarray) -> np.ndarray:
    """
    Minimal magma-like mapping without importing matplotlib here.
    We normalize to [0,1] and use a simple 3-channel polynomial ramp.
    """
    d = depth_hw.astype(np.float32)
    d -= d.min()
    d /= (d.max() + 1e-8)
    # Simple colormap approximation (dark→bright)
    r = np.clip(2.0 * d, 0.0, 1.0)
    g = np.clip(2.0 * (d - 0.25), 0.0, 1.0)
    b = np.clip(2.0 * (d - 0.5), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)
