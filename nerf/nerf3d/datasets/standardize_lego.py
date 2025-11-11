# nerf3d/datasets/standardize_lego.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Any
import numpy as np


# ---------- I/O & inspection ----------

def load_npz(src_npz: Path) -> Dict[str, Any]:
    """
    Load an NPZ and return a plain dict of arrays (no pickles).
    """
    src_npz = Path(src_npz)
    with np.load(src_npz, allow_pickle=False) as z:
        out: Dict[str, Any] = {k: z[k] for k in z.files}
    return out


def inspect_keys(raw: Dict[str, Any]) -> Dict[str, Tuple[int, ...]]:
    """
    Return {key: shape} for quick visibility.
    """
    shapes: Dict[str, Tuple[int, ...]] = {}
    for k, v in raw.items():
        try:
            shapes[k] = tuple(v.shape)  # type: ignore[attr-defined]
        except Exception:
            shapes[k] = tuple()  # non-array scalars
    return shapes


# ---------- mapping to our spec ----------

def map_to_core_fields(raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Normalize raw keys to our canonical names without modifying values:
      - images_uint8: (N,H,W,3) uint8
      - K: (3,3) or (N,3,3) float/np
      - c2w or w2c: (N,4,4)
    """
    core: Dict[str, np.ndarray] = {}

    # images
    for k in ("images", "imgs", "rgb"):
        if k in raw:
            core["images_uint8"] = raw[k]
            break
    else:
        raise KeyError("Expected images key among ['images','imgs','rgb'].")

    # intrinsics
    for k in ("K", "Ks", "intrinsics"):
        if k in raw:
            core["K"] = raw[k]
            break
    else:
        raise KeyError("Expected intrinsics key among ['K','Ks','intrinsics'].")

    # poses (prefer c2w if both)
    if "c2w" in raw:
        core["c2w"] = raw["c2w"]
    elif "w2c" in raw:
        core["w2c"] = raw["w2c"]
    else:
        raise KeyError("Expected pose key 'c2w' or 'w2c'.")

    # basic shape checks (light)
    imgs = core["images_uint8"]
    if imgs.ndim != 4 or imgs.shape[-1] != 3:
        raise ValueError(f"images must be (N,H,W,3); got {imgs.shape}")

    K = core["K"]
    if not ((K.ndim == 2 and K.shape == (3, 3)) or (K.ndim == 3 and K.shape[1:] == (3, 3))):
        raise ValueError(f"K must be (3,3) or (N,3,3); got {K.shape}")

    return core

def ensure_c2w_opengl(core: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Ensure poses are present as c2w (N,4,4). If only w2c is present, invert.
    This function does NOT guess OpenCV→OpenGL flips; it standardizes presence and shape.
    """
    out = dict(core)

    if "c2w" in out:
        c2w = out["c2w"]
    elif "w2c" in out:
        w2c = out["w2c"]
        if w2c.ndim != 3 or w2c.shape[1:] != (4, 4):
            raise ValueError(f"w2c must be (N,4,4); got {w2c.shape}")
        # Invert each 4x4 SE(3)
        N = w2c.shape[0]
        c2w = np.empty_like(w2c, dtype=w2c.dtype)
        for i in range(N):
            W = w2c[i]
            R = W[:3, :3]
            t = W[:3, 3:4]
            R_inv = R.T
            t_inv = -R_inv @ t
            C = np.eye(4, dtype=W.dtype)
            C[:3, :3] = R_inv
            C[:3, 3] = t_inv[:, 0]
            c2w[i] = C
    else:
        raise KeyError("Expected 'c2w' or 'w2c' in core fields.")

    # Basic validation
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"c2w must be (N,4,4); got {c2w.shape}")

    out["c2w"] = c2w.astype(np.float32, copy=False)
    return out

def recenter_and_scale(core: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Recenter world so camera-centroid is at origin, then scale so median camera
    distance ≈ 1.0. Applies the same world transform to all c2w.
    Returns (updated_core, meta).
    """
    if "c2w" not in core:
        raise KeyError("recenter_and_scale expects a 'c2w' field (ensure_c2w_opengl first).")

    c2w = core["c2w"].astype(np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"c2w must be (N,4,4); got {c2w.shape}")

    N = c2w.shape[0]
    cam_centers = c2w[:, :3, 3]  # (N,3)

    # recenter: translate world by -mean(center)
    center = cam_centers.mean(axis=0)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -center

    # apply translation to all c2w (left-multiply world transform)
    c2w_t = (T[None, ...] @ c2w)

    # scale: set median distance to ~1
    centers_t = c2w_t[:, :3, 3]
    dists = np.linalg.norm(centers_t, axis=1)
    med = float(np.median(dists)) if N > 0 else 1.0
    scale = 1.0 if med == 0.0 else (1.0 / med)

    S = np.eye(4, dtype=np.float32)
    S[0, 0] = S[1, 1] = S[2, 2] = scale

    c2w_ts = (S[None, ...] @ c2w_t)

    out = dict(core)
    out["c2w"] = c2w_ts

    meta: Dict[str, Any] = {
        "scene_scale": float(scale),
        "center_transform": (S @ T).astype(np.float32),  # world transform applied to the left
    }
    return out, meta

def choose_near_far(core: Dict[str, np.ndarray], meta: Dict[str, Any]) -> Tuple[float, float]:
    """
    Heuristic near/far selection in *normalized* space (after recenter_and_scale):
      - near: small constant margin (default 0.1)
      - far:  based on camera spread (2.0 * 95th-percentile cam radius), with a floor
    Ensures far >> near and returns plain floats.
    """
    if "c2w" not in core:
        raise KeyError("choose_near_far expects 'c2w' (run ensure_c2w_opengl + recenter_and_scale first).")

    c2w = core["c2w"]
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"c2w must be (N,4,4); got {c2w.shape}")

    # camera centers in normalized coordinates
    centers = c2w[:, :3, 3]  # (N,3)
    if centers.size == 0:
        # fallback if no cameras (edge case)
        return float(0.1), float(2.0)

    radii = np.linalg.norm(centers, axis=1)  # distance to origin
    r95 = float(np.percentile(radii, 95)) if radii.size >= 2 else float(radii.max())
    r95 = max(r95, 1.0)  # after our normalization, median ≈ 1; keep sane lower bound

    near = 0.1
    far_floor = 2.0
    far = max(far_floor, 2.0 * r95)

    # guard: ensure far sufficiently larger than near
    if far < near * 5.0:
        far = near * 5.0

    return float(near), float(far)

def make_image_splits(N: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic 90/5/5 split over image indices.
    Returns (idx_train, idx_val, idx_test) as int32 arrays.
    """
    if N <= 0:
        return (np.zeros((0,), np.int32),) * 3

    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)

    n_train = int(round(0.90 * N))
    n_val = max(1, int(round(0.05 * N))) if N >= 20 else max(1, N // 10)
    n_test = N - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, N - n_val - n_test)

    idx_train = perm[:n_train].astype(np.int32)
    idx_val = perm[n_train:n_train + n_val].astype(np.int32)
    idx_test = perm[n_train + n_val:].astype(np.int32)

    return idx_train, idx_val, idx_test


# ---------- packaging to standardized NPZ ----------

def normalize_images_to_float(images_uint8: np.ndarray) -> np.ndarray:
    """
    Convert uint8 sRGB (N,H,W,3) to float32 in [0,1] without copying more than needed.
    """
    if images_uint8.dtype != np.uint8:
        raise TypeError(f"Expected uint8 images, got {images_uint8.dtype}")
    if images_uint8.ndim != 4 or images_uint8.shape[-1] != 3:
        raise ValueError(f"Expected (N,H,W,3) images, got {images_uint8.shape}")
    return (images_uint8.astype(np.float32) / 255.0)

from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
import json


def assemble_standardized_payload(
    core: Dict[str, np.ndarray],
    idx_splits: Tuple[np.ndarray, np.ndarray, np.ndarray],
    near: float,
    far: float,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the standardized payload for saving.
    Expects in `core`:
      - core["images_uint8"]: (N,H,W,3) uint8
      - core["c2w"]:          (N,4,4)   float
      - core["K"]:            (3,3) or (N,3,3)
    """
    # ----- unpack & validate -----
    if "images_uint8" not in core or "c2w" not in core or "K" not in core:
        raise KeyError("core must contain 'images_uint8', 'c2w', and 'K'")

    images_u8: np.ndarray = core["images_uint8"]
    c2w_all:    np.ndarray = core["c2w"]
    K_all:      np.ndarray = core["K"]

    if images_u8.ndim != 4 or images_u8.shape[-1] != 3:
        raise ValueError(f"images_uint8 must be (N,H,W,3); got {images_u8.shape}")
    if c2w_all.ndim != 3 or c2w_all.shape[1:] != (4, 4):
        raise ValueError(f"c2w must be (N,4,4); got {c2w_all.shape}")
    if not ((K_all.ndim == 2 and K_all.shape == (3, 3)) or (K_all.ndim == 3 and K_all.shape[1:] == (3, 3))):
        raise ValueError(f"K must be (3,3) or (N,3,3); got {K_all.shape}")

    N, H, W, _ = images_u8.shape
    idx_train, idx_val, idx_test = idx_splits
    idx_train = np.asarray(idx_train, dtype=np.int32)
    idx_val   = np.asarray(idx_val,   dtype=np.int32)
    idx_test  = np.asarray(idx_test,  dtype=np.int32)

    # Helper to create correctly-shaped empty arrays
    def empty_images(n: int) -> np.ndarray:
        return np.zeros((n, H, W, 3), dtype=np.float32)
    def empty_poses(n: int) -> np.ndarray:
        return np.zeros((n, 4, 4), dtype=np.float32)
    def empty_Ks(n: int) -> np.ndarray:
        return np.zeros((n, 3, 3), dtype=np.float32)

    # ----- normalize images to [0,1] float32 -----
    images_f32 = (images_u8.astype(np.float32) / 255.0)

    # Split images
    images_train = images_f32[idx_train] if idx_train.size else empty_images(0)
    images_val   = images_f32[idx_val]   if idx_val.size   else empty_images(0)
    images_test  = images_f32[idx_test]  if idx_test.size  else empty_images(0)

    # Split poses
    c2w_train = c2w_all[idx_train].astype(np.float32) if idx_train.size else empty_poses(0)
    c2w_val   = c2w_all[idx_val].astype(np.float32)   if idx_val.size   else empty_poses(0)
    c2w_test  = c2w_all[idx_test].astype(np.float32)  if idx_test.size  else empty_poses(0)

    # Intrinsics handling
    K_is_per_image = (K_all.ndim == 3)
    if K_is_per_image:
        Ks_train = K_all[idx_train].astype(np.float32) if idx_train.size else empty_Ks(0)
        Ks_val   = K_all[idx_val].astype(np.float32)   if idx_val.size   else empty_Ks(0)
        Ks_test  = K_all[idx_test].astype(np.float32)  if idx_test.size  else empty_Ks(0)
        K_payload = {
            "K_train": Ks_train,
            "K_val":   Ks_val,
            "K_test":  Ks_test,
        }
    else:
        # single shared K (3,3)
        K_shared = K_all.astype(np.float32)
        K_payload = {
            "K": K_shared,
        }

    # ----- meta (json, pickle-free) -----
    meta_merged: Dict[str, Any] = dict(meta)  # copy
    meta_merged.setdefault("convention", "c2w_opengl_right_handed")
    meta_merged.setdefault("K_is_per_image", bool(K_is_per_image))
    meta_merged.setdefault("color_space", "srgb_[0,1]")
    meta_merged.setdefault("H", int(H))
    meta_merged.setdefault("W", int(W))
    # Ensure numpy scalars/arrays are JSON encodable
    def _to_py(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj
    meta_clean = {k: _to_py(v) for k, v in meta_merged.items()}
    meta_json = json.dumps(meta_clean, separators=(",", ":"))

    # ----- assemble payload -----
    payload: Dict[str, Any] = {
        # images
        "images_train": images_train.astype(np.float32, copy=False),
        "images_val":   images_val.astype(np.float32, copy=False),
        "images_test":  images_test.astype(np.float32, copy=False),

        # poses
        "c2w_train": c2w_train,
        "c2w_val":   c2w_val,
        "c2w_test":  c2w_test,

        # intrinsics
        **K_payload,

        # splits & shapes
        "idx_train": idx_train,
        "idx_val":   idx_val,
        "idx_test":  idx_test,
        "H": np.int32(H),
        "W": np.int32(W),

        # near/far (scalars)
        "near": np.float32(near),
        "far":  np.float32(far),

        # meta (pickle-free)
        "meta_json": meta_json,
    }

    return payload


def save_standardized_npz(payload: Dict[str, Any], out_npz: Path) -> None:
    """
    Save standardized payload using compressed NPZ.
    """
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed handles dict expansion via kwargs
    np.savez_compressed(out_npz, **payload)


# ---------- CLI driver ----------

def main() -> None:
    """
    Steps:
      1) load_npz(src)
      2) inspect_keys(raw)  # log only
      3) map_to_core_fields(raw)
      4) ensure_c2w_opengl(core)
      5) recenter_and_scale(core) -> meta
      6) choose_near_far(core, meta)
      7) make_image_splits(N)
      8) normalize_images_to_float(images)
      9) assemble_standardized_payload(...)
     10) save_standardized_npz(payload, out)
    """


# if __name__ == "__main__": main()
