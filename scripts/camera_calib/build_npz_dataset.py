# scripts/build_npz_dataset.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import cv2


# ---------- I/O ----------

def load_poses_json(poses_path: Path) -> Dict[str, Any]:
    data = json.loads(poses_path.read_text())
    if "K_new" not in data or "frames" not in data:
        raise ValueError("poses JSON must have 'K_new' and 'frames'.")
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

def _numeric_sort_key(p: Path):
    s = p.stem
    return (s.isdigit(), int(s) if s.isdigit() else s)

def split_indices(n: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Uniform angular split:
      test = every 8th starting at 0
      val  = every 8th starting at 4
      train = rest
    """
    test = list(range(0, n, 8))
    val  = list(range(4, n, 8))
    keep = set(test) | set(val)
    train = [i for i in range(n) if i not in keep]
    return train, val, test


# ---------- Packing Data ----------

def build_arrays(
    images_dir: Path,
    poses_data: Dict[str, Any],
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    # Intrinsics
    K = np.array(poses_data["K_new"], dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    focal = float((fx + fy) * 0.5)

    # Distortion (could be zeros if already undistorted workflow)
    dist = np.array(poses_data.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).ravel()

    frames = poses_data["frames"]

    # Sort frame file paths numerically to define index order
    paths = [images_dir / f["file"] for f in frames]
    # The JSON frames list is already aligned to detection order; we keep that order.
    # But ensure the path exists:
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Listed image not found: {p}")

    # Determine size from first image
    sample = load_and_undistort(paths[0], K, dist)
    H, W = sample.shape[:2]

    def collect(idxs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        imgs: List[np.ndarray] = []
        mats: List[np.ndarray] = []
        for i in idxs:
            # image
            img_rgb = sample if i == 0 else load_and_undistort(paths[i], K, dist)
            if img_rgb.shape[0] != H or img_rgb.shape[1] != W:
                # In case sizes differ, resize to the first one
                img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
            imgs.append(img_rgb.astype(np.uint8))
            # pose
            R = np.array(frames[i]["R"], dtype=np.float64)
            t = np.array(frames[i]["t"], dtype=np.float64)
            c2w = rt_to_c2w(R, t, mm_to_m=True)  # keep raw OpenCV convention; just convert units
            mats.append(c2w.astype(np.float64))
        if len(imgs) == 0:
            return np.zeros((0, H, W, 3), dtype=np.uint8), np.zeros((0, 4, 4), dtype=np.float64)
        return np.stack(imgs, axis=0), np.stack(mats, axis=0)

    images_train, c2ws_train = collect(train_ids)
    images_val,   c2ws_val   = collect(val_ids)
    _,            c2ws_test  = collect(test_ids)   # test usually images not needed for training

    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal


# ---------- Save ----------

def save_npz(
    out_path: Path,
    images_train: np.ndarray,
    c2ws_train: np.ndarray,
    images_val: np.ndarray,
    c2ws_val: np.ndarray,
    c2ws_test: np.ndarray,
    focal: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        images_train=images_train,    # (N_train, H, W, 3) uint8
        c2ws_train=c2ws_train,        # (N_train, 4, 4) float64
        images_val=images_val,        # (N_val, H, W, 3) uint8
        c2ws_val=c2ws_val,            # (N_val, 4, 4) float64
        c2ws_test=c2ws_test,          # (N_test, 4, 4) float64
        focal=focal,                  # float
    )


# ---------- Orchestrator ----------

def build_npz_dataset(poses_path: Path, images_dir: Path, out_path: Path) -> None:
    data = load_poses_json(poses_path)
    n = len(data["frames"])
    train_ids, val_ids, test_ids = split_indices(n)
    arrays = build_arrays(images_dir, data, train_ids, val_ids, test_ids)
    save_npz(out_path, *arrays)


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Build NPZ dataset (undistort + raw OpenCV c2w).")
    ap.add_argument("--poses_json", type=Path, required=True,
                    help="data/object_3d/results/poses_pnp.json")
    ap.add_argument("--images_dir", type=Path, required=True,
                    help="data/object_3d/images")
    ap.add_argument("--out", type=Path, required=True,
                    help="data/object_3d/results/my_data.npz")
    return ap.parse_args()


def main():
    args = parse_args()
    build_npz_dataset(args.poses_json, args.images_dir, args.out)


if __name__ == "__main__":
    main()
