# scripts/pnp_single_tag.py
from __future__ import annotations
import json, argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import cv2

# ---------- Specs ----------

@dataclass(frozen=True)
class TagSpec:
    id: int                 # e.g., 9
    size_mm: float          # e.g., 98.0
    dict_name: str = "DICT_4X4_50"

@dataclass(frozen=True)
class CameraCalib:
    K: np.ndarray           # (3,3)
    dist: np.ndarray        # (N,)

@dataclass
class PoseEntry:
    file: str
    reproj_err_px: float
    R: np.ndarray           # (3,3)
    t: np.ndarray           # (3,1)

# ---------- Helpers ----------

def _aruco_dict_from_name(name: str):
    key = name if name.startswith("DICT_") else f"DICT_{name}"
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))

def load_calibration(calib_json: Path) -> CameraCalib:
    d = json.loads(calib_json.read_text())
    K = np.array(d["K"], dtype=np.float64)
    dist = np.array(d["dist"], dtype=np.float64).ravel()
    return CameraCalib(K=K, dist=dist)

def tag_object_corners_mm(spec: TagSpec) -> np.ndarray:
    s = float(spec.size_mm) / 2.0
    # TL, TR, BR, BL on z=0
    return np.array([[-s, -s, 0.0],
                     [ s, -s, 0.0],
                     [ s,  s, 0.0],
                     [-s,  s, 0.0]], dtype=np.float32)

def undistort_and_detect(
    image_bgr: np.ndarray,
    calib: CameraCalib,
    dict_name: str,
    target_id: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(calib.K, calib.dist, (w, h), 0)
    undist = cv2.undistort(image_bgr, calib.K, calib.dist, None, newK)
    aruco_dict = _aruco_dict_from_name(dict_name)
    aruco_params = cv2.aruco.DetectorParameters()
    corners_list, ids, _ = cv2.aruco.detectMarkers(undist, aruco_dict, parameters=aruco_params)
    if ids is None:
        raise RuntimeError("No ArUco markers detected.")
    ids_flat = ids.flatten()
    matches = np.where(ids_flat == int(target_id))[0]
    if len(matches) == 0:
        raise RuntimeError(f"Target ArUco id {target_id} not found in this image.")
    idx = int(matches[0])
    corners = corners_list[idx].reshape(4, 2).astype(np.float32)  # TL,TR,BR,BL
    return undist, newK.astype(np.float64), corners

def pnp_solve_and_error(
    corners_2d: np.ndarray,           # (4,2)
    object_corners_mm: np.ndarray,    # (4,3)
    K_new: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    ok, rvec, tvec = cv2.solvePnP(object_corners_mm, corners_2d, K_new, None,
                                  flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise RuntimeError("solvePnP failed (EPNP).")
    try:
        rvec, tvec = cv2.solvePnPRefineLM(object_corners_mm, corners_2d, K_new, None, rvec, tvec)
    except Exception:
        pass
    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(object_corners_mm, rvec, tvec, K_new, None)
    proj = proj.reshape(-1, 2)
    err = float(np.linalg.norm(proj - corners_2d, axis=1).mean())
    return R.astype(np.float64), tvec.astype(np.float64), err

def build_and_save_json(
    out_path: Path,
    K_new: np.ndarray,
    tag_spec: TagSpec,
    frames: List[PoseEntry]
) -> None:
    def tolist(x):
        return x.tolist() if isinstance(x, np.ndarray) else x
    payload = {
        "K_new": tolist(K_new.astype(float)),
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],  # undistorted output model
        "tag_spec": asdict(tag_spec),
        "frames": [
            {
                "file": f.file,
                "reproj_err_px": f.reproj_err_px,
                "R": tolist(f.R.astype(float)),
                "t": tolist(f.t.astype(float)),
            } for f in frames
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

# ---------- Orchestrator ----------

def pnp_from_single_tag(
    images_dir: Path,
    calib_json: Path,
    out_json: Path,
    tag_spec: TagSpec,
    max_err_px: Optional[float] = None
) -> None:
    calib = load_calibration(calib_json)
    obj_corners = tag_object_corners_mm(tag_spec)
    paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
                   key=lambda p: (p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem))
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    frames: List[PoseEntry] = []
    K_new_global: Optional[np.ndarray] = None
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            undist, K_new, corners_2d = undistort_and_detect(img, calib, tag_spec.dict_name, tag_spec.id)
            if K_new_global is None:
                K_new_global = K_new
            R, t, err = pnp_solve_and_error(corners_2d, obj_corners, K_new)
            if (max_err_px is None) or (err <= max_err_px):
                frames.append(PoseEntry(file=p.name, reproj_err_px=err, R=R, t=t))
        except RuntimeError:
            continue
    if not frames:
        raise RuntimeError("No valid poses computed.")
    build_and_save_json(out_json, K_new_global if K_new_global is not None else calib.K, tag_spec, frames)

# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(description="PnP poses from a single ArUco tag per image (bird dataset).")
    ap.add_argument("--images_dir", type=Path, default=Path("data/bird/images_raw"),
                    help="Folder with bird object images (400x300)")
    ap.add_argument("--calib_json", type=Path, default=Path("data/bird/intrinsics/camera_calib.json"),
                    help="Intrinsics JSON produced by calibrate_camera.py")
    ap.add_argument("--out", type=Path, default=Path("data/bird/results/poses_pnp.json"),
                    help="Output poses JSON")
    ap.add_argument("--tag_id", type=int, default=0, help="Your tag ID in the frame")
    ap.add_argument("--tag_size_mm", type=float, default=40.0, help="Marker size (edge length) in mm")
    ap.add_argument("--dict", type=str, default="DICT_4X4_50")
    ap.add_argument("--max_err_px", type=float, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    spec = TagSpec(id=args.tag_id, size_mm=args.tag_size_mm, dict_name=args.dict)
    pnp_from_single_tag(args.images_dir, args.calib_json, args.out, spec, max_err_px=args.max_err_px)

if __name__ == "__main__":
    main()