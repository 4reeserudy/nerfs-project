# scripts/calibrate_camera.py
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse

import numpy as np
import cv2

# ----------------- Data specs -----------------

@dataclass(frozen=True)
class BoardSpec:
    rows: int          # e.g., 3  (rows, top->bottom)
    cols: int          # e.g., 2  (cols, left->right)
    tag_size_mm: float # e.g., 60.0
    dx_mm: float       # horizontal center-to-center spacing
    dy_mm: float       # vertical   center-to-center spacing

@dataclass
class CalibResult:
    K: np.ndarray
    dist: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    rms: float
    per_image_err: Optional[List[float]] = None
    used_images: Optional[List[str]] = None

# ----------------- Core helpers -----------------

def _dict_from_name(dict_name: str):
    # e.g., "DICT_4X4_50" -> cv2.aruco.DICT_4X4_50
    key = dict_name if dict_name.startswith("DICT_") else f"DICT_{dict_name}"
    const = getattr(cv2.aruco, key)
    return cv2.aruco.getPredefinedDictionary(const)

def create_detector_and_layout(
    dict_name: str,
    spec: BoardSpec
) -> Tuple[object, object, Dict[int, np.ndarray]]:
    aruco_dict = _dict_from_name(dict_name)
    aruco_params = cv2.aruco.DetectorParameters()
    # Precompute 3D corners per ID (row-major IDs: 0..rows*cols-1)
    half = spec.tag_size_mm / 2.0
    offsets = np.array([[-half, -half, 0.0],
                        [ half, -half, 0.0],
                        [ half,  half, 0.0],
                        [-half,  half, 0.0]], dtype=np.float32)  # TL,TR,BR,BL
    layout: Dict[int, np.ndarray] = {}
    for k in range(spec.rows * spec.cols):
        r, c = divmod(k, spec.cols)  # row-major
        center = np.array([c * spec.dx_mm, r * spec.dy_mm, 0.0], dtype=np.float32)
        layout[k] = center + offsets  # (4,3)
    return aruco_dict, aruco_params, layout

def process_image(
    image_bgr: np.ndarray,
    aruco_dict: object,
    aruco_params: object,
    layout_3d_by_id: Dict[int, np.ndarray],
    min_tags: int = 2
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    corners, ids, _ = cv2.aruco.detectMarkers(image_bgr, aruco_dict, parameters=aruco_params)
    if ids is None or len(ids) < min_tags:
        return None
    # Normalize: sort by ID asc; keep TL,TR,BR,BL order per tag
    order = np.argsort(ids.flatten())
    obj_list, img_list = [], []
    for idx in order:
        tag_id = int(ids[idx][0])
        if tag_id not in layout_3d_by_id:
            continue
        img_corners = corners[idx].reshape(4, 2).astype(np.float32)  # (4,2)
        obj_corners = layout_3d_by_id[tag_id].astype(np.float32)     # (4,3)
        obj_list.append(obj_corners)
        img_list.append(img_corners)
    if not obj_list:
        return None
    obj_pts = np.vstack(obj_list)  # (M,3)
    img_pts = np.vstack(img_list)  # (M,2)
    return obj_pts, img_pts

def calibrate_and_evaluate(
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    image_size: Tuple[int, int]
) -> CalibResult:
    flags = 0  # keep default; we want full estimation
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags
    )
    # Per-image reprojection error
    per_img_err = []
    for i, (obj_i, img_i) in enumerate(zip(object_points, image_points)):
        proj, _ = cv2.projectPoints(obj_i, rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - img_i, axis=1).mean()
        per_img_err.append(float(err))
    return CalibResult(K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, rms=float(ret), per_image_err=per_img_err)

def save_result(
    result: CalibResult,
    out_path: Path,
    spec: BoardSpec,
    image_size: Tuple[int, int]
) -> None:
    def nparr(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return [xi.tolist() if isinstance(xi, np.ndarray) else xi for xi in x]

    payload = {
        "board_spec": asdict(spec),
        "image_size_wh": list(map(int, image_size)),
        "K": nparr(result.K),
        "dist": nparr(result.dist),
        "rms": result.rms,
        "per_image_err": result.per_image_err,
        "used_images": result.used_images,
        "rvecs": nparr(np.array(result.rvecs, dtype=object)),
        "tvecs": nparr(np.array(result.tvecs, dtype=object)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

# ----------------- Orchestrator -----------------

def calibrate_camera(
    images_dir: Path,
    out_path: Path,
    spec: BoardSpec,
    dict_name: str = "DICT_4X4_50",
    min_tags_per_image: int = 2
) -> CalibResult:
    results_dir = images_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / out_path.name  # always drop result into /results/

    image_paths = sorted(
        [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: (p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem)
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    aruco_dict, aruco_params, layout = create_detector_and_layout(dict_name, spec)

    object_points, image_points, used = [], [], []
    im0 = cv2.imread(str(image_paths[0]))
    if im0 is None:
        raise RuntimeError(f"Failed to read {image_paths[0]}")
    h, w = im0.shape[:2]
    image_size = (w, h)

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        res = process_image(img, aruco_dict, aruco_params, layout, min_tags=min_tags_per_image)
        if res is None:
            continue
        obj_i, img_i = res
        object_points.append(obj_i.astype(np.float32))
        image_points.append(img_i.astype(np.float32))
        used.append(p.name)

    if len(object_points) < 1:
        raise RuntimeError("No valid images with sufficient ArUco detections.")

    result = calibrate_and_evaluate(object_points, image_points, image_size)
    result.used_images = used
    save_result(result, out_path, spec, image_size)
    return result

# ----------------- CLI -----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Calibrate camera using a 3x2 ArUco board.")
    ap.add_argument("--images_dir", type=Path, required=True, help="Folder with images (e.g., 1.jpg..42.jpg)")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--tag_size_mm", type=float, default=60.0)
    ap.add_argument("--dx_mm", type=float, default=90.0)
    ap.add_argument("--dy_mm", type=float, default=75.67)
    ap.add_argument("--dict", type=str, default="DICT_4X4_50")
    ap.add_argument("--min_tags_per_image", type=int, default=2)
    return ap.parse_args()

def main():
    args = parse_args()
    spec = BoardSpec(rows=args.rows, cols=args.cols,
                     tag_size_mm=args.tag_size_mm, dx_mm=args.dx_mm, dy_mm=args.dy_mm)
    calibrate_camera(args.images_dir, args.out, spec, dict_name=args.dict,
                     min_tags_per_image=args.min_tags_per_image)

if __name__ == "__main__":
    main()