# nerf/nerf3d/data/dataloader.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import torch
import random
import os
import numpy as np

# Reuse Scene + samplers
from nerf.nerf3d.rays.ray_sampler import Scene, RaySamplerGlobal, RaySamplerPerImage, RaysDataset


# -----------------------------------------------------------------------------
# Core I/O / standardization
# -----------------------------------------------------------------------------

def load_npz(npz_path: str) -> Dict[str, Any]:
    """
    Loads an .npz file and returns a dict-like object.
    Does NOT modify types or normalize values – raw load only.
    """
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def normalize_images_to_float(arr: np.ndarray) -> np.ndarray:
    """
    Converts uint8 images to float32 in [0,1]. If already float,
    returns float32 unchanged but clipped to [0,1].
    Shapes: (N,H,W,3) or (H,W,3).
    """
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return arr

def ensure_c2w_opengl(c2ws: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV-style c2w to OpenGL convention by flipping Y,Z.
    If your dataset is already OpenGL, disable this.
    """
    assert c2ws.ndim == 3 and c2ws.shape[1:] == (4, 4), "c2ws must be (N,4,4)"
    flip = np.diag([1, -1, -1, 1]).astype(np.float32)  # 4x4
    return c2ws @ flip

def compute_intrinsics(focal, H: int, W: int) -> np.ndarray:
    """Build 3x3 K from focal (scalar or [fx,fy]) and image size."""
    if np.isscalar(focal):
        fx = fy = float(focal)
    else:
        assert len(focal) == 2, "focal must be scalar or [fx, fy]"
        fx, fy = float(focal[0]), float(focal[1])
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


# -----------------------------------------------------------------------------
# Split selection / mapping
# -----------------------------------------------------------------------------

def select_split(
    images: np.ndarray,
    c2ws: np.ndarray,
    split: str,
    train_frac: float = 0.9,
    val_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Selects a split of (images, c2ws) based on a split name.
    If the dataset already provides separate splits, you won't use this.
    """
    N = images.shape[0]
    assert N == c2ws.shape[0], "images and c2ws must have same length"

    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    if split == "train":
        start, end = 0, n_train
    elif split == "val":
        start, end = n_train, n_train + n_val
    elif split == "test":
        start, end = n_train + n_val, N
    else:
        raise ValueError(f"Unknown split: {split}")

    return images[start:end], c2ws[start:end]

def map_to_torch(batch: dict, device: torch.device) -> dict:
    """Move numpy arrays to torch tensors on device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if torch.is_tensor(v):
            v = v.to(device)
        out[k] = v
    return out


# -----------------------------------------------------------------------------
# Scene assembly + near/far
# -----------------------------------------------------------------------------

def choose_near_far(
    c2ws: torch.Tensor,
    images_hw: Tuple[int, int],
    focal: Optional[float | torch.Tensor] = None,
    fallback: Tuple[float, float] = (2.0, 6.0),
) -> Tuple[float, float]:
    """
    Heuristic near/far if dataset doesn't specify.
    near ~ 0.05*median_radius, far ~ 3*median_radius
    """
    try:
        centers = c2ws[:, :3, 3]
        center = centers.mean(dim=0, keepdim=True)
        r = torch.median(torch.linalg.norm(centers - center, dim=-1)).item()
        if not np.isfinite(r) or r <= 0:
            return fallback
        near = max(0.05 * r, 1e-2)
        far = max(3.0 * r, near + 1e-3)
        return float(near), float(far)
    except Exception:
        return fallback

def make_scene_from_npz(
    npz_path: str,
    split: str = "train",
    near: Optional[float] = None,
    far: Optional[float] = None,
) -> Tuple[Scene, Dict[str, Any]]:
    """
    Load LEGO-style NPZ and return:
      - Scene(images, Ks, c2ws, near, far)
      - meta: {"H","W","focal","count","split","npz_path","Ks_shared","near","far"}
    """
    raw = load_npz(npz_path)

    img_key = f"images_{split}"
    c2w_key = f"c2ws_{split}"
    if img_key not in raw or c2w_key not in raw:
        raise KeyError(f"Expected keys '{img_key}' and '{c2w_key}' in {npz_path}")

    images_np = normalize_images_to_float(raw[img_key])      # (M,H,W,3)
    c2ws_np   = raw[c2w_key].astype(np.float32)              # (M,4,4)
    c2ws_np   = ensure_c2w_opengl(c2ws_np)                   # toggle if needed

    M, H, W, _ = images_np.shape

    # intrinsics
    if "K" in raw:
        K_np = raw["K"].astype(np.float32)
        if K_np.ndim == 2:
            Ks_np = K_np[None, ...]
            focal_val = float(K_np[0, 0])
        elif K_np.ndim == 3:
            Ks_np = K_np
            focal_val = float(np.mean(K_np[:, (0, 1), (0, 1)]))
        else:
            raise ValueError("Unsupported K shape")
    else:
        focal_raw = raw.get("focal", None)
        if focal_raw is None:
            raise KeyError("NPZ missing 'focal' and 'K'")
        if np.isscalar(focal_raw) or (isinstance(focal_raw, np.ndarray) and focal_raw.ndim == 0):
            K_single = compute_intrinsics(float(focal_raw), H, W)
            Ks_np = K_single[None, ...]
            focal_val = float(focal_raw)
        elif isinstance(focal_raw, np.ndarray) and focal_raw.ndim == 1 and focal_raw.size == 2:
            K_single = compute_intrinsics([focal_raw[0], focal_raw[1]], H, W)
            Ks_np = K_single[None, ...]
            focal_val = float(np.mean(focal_raw))
        else:
            f_arr = np.array(focal_raw, dtype=np.float32).reshape(-1)
            if f_arr.size == M:
                Ks_np = np.stack([compute_intrinsics(float(f), H, W) for f in f_arr], axis=0)
                focal_val = float(np.mean(f_arr))
            else:
                raise ValueError("Unsupported 'focal' format")

    # --- torch tensors ---
    images_t = torch.from_numpy(images_np).contiguous()   # (M,H,W,3)
    c2ws_t   = torch.from_numpy(c2ws_np).contiguous()     # (M,4,4)
    Ks_t     = torch.from_numpy(Ks_np).contiguous()       # (1,3,3) or (M,3,3)

    # If K is shared (1,3,3), expand to per-image (M,3,3) for safe indexing
    if Ks_t.ndim == 3 and Ks_t.shape[0] == 1:
        Ks_t = Ks_t.expand(images_t.shape[0], -1, -1).contiguous()

    if near is None or far is None:
        n, f = choose_near_far(c2ws_t, (H, W), focal=focal_val, fallback=(2.0, 6.0))
    else:
        n, f = float(near), float(far)

    scene = Scene(
        images=images_t,
        Ks=Ks_t,
        c2ws=c2ws_t,
        near=n,
        far=f,
        names=[f"{split}_{i:04d}" for i in range(M)],
    )

    meta: Dict[str, Any] = {
        "H": H, "W": W, "focal": focal_val, "count": M, "split": split,
        "npz_path": npz_path, "Ks_shared": (Ks_t.shape[0] == 1), "near": n, "far": f,
    }
    return scene, meta


# -----------------------------------------------------------------------------
# Sampler builders
# -----------------------------------------------------------------------------

def build_ray_sampler(
    scene: Scene,
    policy: str = "global",
    seed: Optional[int] = None,
    choose_M: Optional[int] = None,
):
    """Return a RaySampler based on policy."""
    policy = policy.lower()
    if policy == "global":
        return RaySamplerGlobal(scene, seed=seed)
    elif policy == "per_image":
        return RaySamplerPerImage(scene, seed=seed, choose_M=choose_M)
    else:
        raise ValueError(f"unknown policy: {policy!r} (expected 'global' or 'per_image')")

def make_iterable_dataset(
    scene: Scene,
    policy: str,
    batch_size: int,
    device: Optional[torch.device],
    seed: Optional[int],
    choose_M: Optional[int],
    pixel_center: bool = True,
) -> RaysDataset:
    """Wrap sampler policy into an IterableDataset that yields batches."""
    return RaysDataset(
        scene=scene,
        policy=policy,
        batch_size=batch_size,
        device=device,
        seed=seed,
        choose_M=choose_M,
        pixel_center=pixel_center,
    )

def make_loader(
    dataset: RaysDataset,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> torch.utils.data.DataLoader:
    """Torch DataLoader for an IterableDataset that already yields full batches."""
    kwargs = dict(
        dataset=dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(**kwargs)


# -----------------------------------------------------------------------------
# Utilities / entrypoint
# -----------------------------------------------------------------------------

def seed_everything(seed: Optional[int]) -> None:
    """Seed python/np/torch RNGs (cpu+cuda)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all tensor-like values in dict to device."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
        out[k] = v
    return out

def make_loaders(
    npz_path: str,
    batch_size: int = 4096,
    policy: str = "global",             # or "per_image"
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    choose_M: Optional[int] = None,
    near: Optional[float] = None,
    far: Optional[float] = None,
    pixel_center: bool = True,
    return_scene: bool = False,
) -> Tuple[torch.utils.data.DataLoader,
           torch.utils.data.DataLoader,
           Optional[Dict[str, Any]]]:
    """
    Build TRAIN and VAL loaders from a single NPZ dataset and (optionally) return
    a small 'scene' dict for snapshot anchoring (images_val, c2ws_val, focal).
    """
    seed_everything(seed)

    # device default
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- TRAIN ---
    train_scene, _train_meta = make_scene_from_npz(
        npz_path=npz_path, split="train", near=near, far=far)
    train_dataset = make_iterable_dataset(
        scene=train_scene, policy=policy, batch_size=batch_size,
        device=device, seed=seed, choose_M=choose_M, pixel_center=pixel_center,
    )
    train_loader = make_loader(
        dataset=train_dataset, num_workers=num_workers,
        pin_memory=pin_memory, prefetch_factor=2,
    )

    # --- VAL ---
    val_scene, _val_meta = make_scene_from_npz(
        npz_path=npz_path, split="val", near=near, far=far)
    val_dataset = make_iterable_dataset(
        scene=val_scene, policy="global", batch_size=batch_size,
        device=device, seed=seed, choose_M=None, pixel_center=True,
    )
    val_loader = make_loader(
        dataset=val_dataset, num_workers=num_workers,
        pin_memory=pin_memory, prefetch_factor=2,
    )

    # Optional scene bundle for consistent snapshots
    scene_bundle = None
    if return_scene:
        scene_bundle = {
            "images_val": val_scene.images,   # attr access (Scene object)
            "c2ws_val":  val_scene.c2ws,
            "focal":     float(_val_meta["focal"]),  # from meta
        }

    return train_loader, val_loader, scene_bundle
