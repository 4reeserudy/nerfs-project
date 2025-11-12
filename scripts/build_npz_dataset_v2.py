# scripts/build_npz_dataset_v2.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import cv2

try:
    from tqdm import tqdm
except Exception:
    # lightweight fallback if tqdm isn't installed
    def tqdm(x, **kwargs): return x


# ---------- I/O ----------

def load_poses_json(poses_path: Path) -> Dict[str, Any]:
    data = json.loads(poses_path.read_text())
    if "K_new" not in data or "frames" not in data:
        raise ValueError("poses JSON must have 'K_new' and 'frames'.")
    # normalize keys we’ll use repeatedly
    data["_fx"] = float(data["K_new"][0][0])
    data["_fy"] = float(data["K_new"][1][1])
    data["_cx"] = float(data["K_new"][0][2])
    data["_cy"] = float(data["K_new"][1][2])
    data["_dist"] = np.array(data.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).ravel()
    return data


def load_and_undistort(img_path: Path, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Load BGR, undistort via cv2.undistort, return RGB uint8."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    undist = cv2.undistort(img, K, dist)  # if dist==0, this is a no-op
    rgb = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    return rgb


# ---------- Pose Conversion ----------

def rt_to_c2w(R: np.ndarray, t: np.ndarray, mm_to_m: bool = True) -> np.ndarray:
    """
    OpenCV returns T_cw=[R|t] (world->camera).
    We need camera->world: T_wc = [[R^T, -R^T t],[0 0 0 1]]
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


# ---------- Split Strategy A ----------

def split_indices(n: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Uniform angular-ish split by index:
      test = every 8th starting at 0
      val  = every 8th starting at 4
      train = rest
    """
    test = list(range(0, n, 8))
    val  = list(range(4, n, 8))
    keep = set(test) | set(val)
    train = [i for i in range(n) if i not in keep]
    return train, val, test


# ---------- Helpers ----------

def scale_intrinsics(K: np.ndarray, src_wh: Tuple[int,int], dst_wh: Tuple[int,int]) -> np.ndarray:
    """Scale intrinsics from (W,H) -> (W',H')."""
    W, H = src_wh
    W2, H2 = dst_wh
    sx = W2 / float(W)
    sy = H2 / float(H)
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2


def maybe_resize(img: np.ndarray, out_wh: Tuple[int,int]|None) -> np.ndarray:
    if out_wh is None:
        return img
    W2, H2 = out_wh
    if img.shape[1] == W2 and img.shape[0] == H2:
        return img
    return cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA)


# ---------- Packing Data ----------

def build_arrays(
    images_dir: Path,
    poses_data: Dict[str, Any],
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    max_err_px: float,
    resize_wh: Tuple[int,int] | None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float,
    Dict[str, Any]
]:
    # Intrinsics (original)
    K = np.array(poses_data["K_new"], dtype=np.float64)
    dist = poses_data["_dist"]

    frames_all = poses_data["frames"]

    # --- Filter frames by reprojection error gate ---
    keep_mask = [ (f.get("reproj_err_px", 0.0) <= max_err_px) for f in frames_all ]
    frames = [f for f, keep in zip(frames_all, keep_mask) if keep]

    if len(frames) == 0:
        raise ValueError(f"No frames pass max_err_px={max_err_px}. Try increasing the threshold.")

    # Construct full paths and sanity check
    paths = [images_dir / f["file"] for f in frames]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Listed image not found: {p}")

    # Determine native size from the first image (after undistort)
    sample = load_and_undistort(paths[0], K, dist)
    H0, W0 = sample.shape[:2]

    # Handle optional resize and update intrinsics accordingly
    if resize_wh is not None:
        W2, H2 = resize_wh
        K2 = scale_intrinsics(K, (W0, H0), (W2, H2))
        sample = maybe_resize(sample, (W2, H2))
        H, W = sample.shape[:2]
        K_work = K2
    else:
        H, W = H0, W0
        K_work = K

    fx, fy, cx, cy = K_work[0, 0], K_work[1, 1], K_work[0, 2], K_work[1, 2]
    focal = float((fx + fy) * 0.5)

    # Rebuild split indices relative to the *filtered* list
    n = len(frames)
    train_ids, val_ids, test_ids = split_indices(n)

    # Collect arrays
    def collect(idxs: List[int]) -> Tuple[np.ndarray, np.ndarray, List[str], List[float]]:
        imgs: List[np.ndarray] = []
        mats: List[np.ndarray] = []
        files: List[str] = []
        errs: List[float] = []
        for i in idxs:
            f = frames[i]
            img_rgb = sample if i == 0 else load_and_undistort(paths[i], K, dist)
            img_rgb = maybe_resize(img_rgb, (W, H))
            imgs.append(img_rgb.astype(np.uint8))

            R = np.array(f["R"], dtype=np.float64)
            t = np.array(f["t"], dtype=np.float64)
            c2w = rt_to_c2w(R, t, mm_to_m=True).astype(np.float32)  # store as float32
            mats.append(c2w)

            files.append(f["file"])
            errs.append(float(f.get("reproj_err_px", 0.0)))

        if len(imgs) == 0:
            return (np.zeros((0, H, W, 3), dtype=np.uint8),
                    np.zeros((0, 4, 4), dtype=np.float32),
                    [], [])
        return np.stack(imgs, axis=0), np.stack(mats, axis=0), files, errs

    images_train, c2ws_train, files_train, errs_train = collect(train_ids)
    images_val,   c2ws_val,   files_val,   errs_val   = collect(val_ids)
    # test split: poses only
    _,            c2ws_test,  files_test,  errs_test  = collect(test_ids)

    meta = {
        "H": int(H),
        "W": int(W),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "focal": focal,
        "resize_applied": resize_wh is not None,
        "max_err_px": float(max_err_px),
        "files_train": files_train,
        "files_val": files_val,
        "files_test": files_test,
        "errs_train": errs_train,
        "errs_val": errs_val,
        "errs_test": errs_test,
        "num_total_after_filter": int(n),
    }

    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal, meta


# ---------- Save ----------

def save_npz(
    out_path: Path,
    images_train: np.ndarray,
    c2ws_train: np.ndarray,
    images_val: np.ndarray,
    c2ws_val: np.ndarray,
    c2ws_test: np.ndarray,
    focal: float,
    meta: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        images_train=images_train,    # (N_train, H, W, 3) uint8
        c2ws_train=c2ws_train,        # (N_train, 4, 4) float32
        images_val=images_val,        # (N_val, H, W, 3) uint8
        c2ws_val=c2ws_val,            # (N_val, 4, 4) float32
        c2ws_test=c2ws_test,          # (N_test, 4, 4) float32
        focal=np.float32(focal),      # float32
        H=np.int32(meta["H"]),
        W=np.int32(meta["W"]),
        fx=np.float32(meta["fx"]),
        fy=np.float32(meta["fy"]),
        cx=np.float32(meta["cx"]),
        cy=np.float32(meta["cy"]),
        max_err_px=np.float32(meta["max_err_px"]),
        files_train=np.array(meta["files_train"]),
        files_val=np.array(meta["files_val"]),
        files_test=np.array(meta["files_test"]),
        errs_train=np.array(meta["errs_train"], dtype=np.float32),
        errs_val=np.array(meta["errs_val"], dtype=np.float32),
        errs_test=np.array(meta["errs_test"], dtype=np.float32),
    )


# ---------- Orchestrator ----------

def build_npz_dataset_v2(
    poses_path: Path,
    images_dir: Path,
    out_path: Path,
    max_err_px: float = 3.0,
    resize_wh: Tuple[int,int] | None = None,
) -> None:
    data = load_poses_json(poses_path)
    # Fast path: just to verify files exist before heavy work
    frames = data["frames"]
    for f in tqdm(frames, desc="verifying files"):
        p = images_dir / f["file"]
        if not p.exists():
            raise FileNotFoundError(f"Listed image not found: {p}")

    # Initial split placeholders (the function recomputes splits AFTER filtering)
    train_ids, val_ids, test_ids = split_indices(len(frames))

    arrays = build_arrays(
        images_dir, data, train_ids, val_ids, test_ids,
        max_err_px=max_err_px,
        resize_wh=resize_wh,
    )
    save_npz(out_path, *arrays)


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Build NPZ dataset v2 (undistort, error-gate, optional resize).")
    ap.add_argument("--poses_json", type=Path, required=True,
                    help="data/object_3d/results/poses_pnp.filtered.json")
    ap.add_argument("--images_dir", type=Path, required=True,
                    help="data/object_3d/images")
    ap.add_argument("--out", type=Path, required=True,
                    help="data/object_3d/results/my_data_v2.npz")
    ap.add_argument("--max_err_px", type=float, default=3.0,
                    help="Reject frames with reprojection error above this (default: 3.0).")
    ap.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                    help="Optionally resize images to W H and rescale intrinsics.")
    return ap.parse_args()


def main():
    args = parse_args()
    resize_wh = tuple(args.resize) if args.resize is not None else None
    build_npz_dataset_v2(
        args.poses_json, args.images_dir, args.out,
        max_err_px=args.max_err_px,
        resize_wh=resize_wh,
    )


if __name__ == "__main__":
    main()
