# postfilter_poses.py
# Usage (Colab/CLI):
#   !python postfilter_poses.py --in_json /path/to/poses_pnp.json --out_json /path/to/poses_pnp.filtered.json

import json, math, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

def load_poses(path: Path) -> Dict[str, Any]:
    d = json.loads(path.read_text())
    # Normalize R,t to numpy
    for f in d.get("frames", []):
        f["R"] = np.asarray(f["R"], dtype=float)
        f["t"] = np.asarray(f["t"], dtype=float).reshape(3, 1)
        f["reproj_err_px"] = float(f["reproj_err_px"])
    return d

def cam_center_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Object->Camera: Xc = R Xw + t ; Camera center in world/object coords: C = -R^T t
    return (-R.T @ t).reshape(3)

def az_el_from_xyz(p: np.ndarray) -> Tuple[float, float]:
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    az = math.degrees(math.atan2(y, x))                # [-180,180]
    r_xy = math.hypot(x, y)
    el = math.degrees(math.atan2(z, r_xy))             # [-90,90]
    return az, el

def det_ok(R: np.ndarray, tol: float = 0.1) -> bool:
    d = float(np.linalg.det(R))
    return (1.0 - tol) <= d <= (1.0 + tol)

def forward_facing_ok(R: np.ndarray, max_off_deg: float = 100.0) -> bool:
    # Camera forward (z_cam) expressed in world coords:
    fwd = (R.T @ np.array([0.0, 0.0, 1.0])).reshape(3)
    fwd /= max(1e-9, np.linalg.norm(fwd))
    # Tag plane normal (world z) — camera generally looks *toward* +Z or -Z depending on board printing.
    # We allow either orientation but reject if nearly tangent/backfacing: require |dot| >= cos(max_off)
    cos_th = abs(float(fwd @ np.array([0.0, 0.0, 1.0])))
    return cos_th >= math.cos(math.radians(max_off_deg))

def greedy_cover_add(frames_pool: List[Dict[str, Any]],
                     kept: List[Dict[str, Any]],
                     az_thresh: float, el_thresh: float,
                     err_max: float) -> List[Dict[str, Any]]:
    def sep_ok(az: float, el: float, kept_angles: List[Tuple[float,float]]) -> bool:
        for kaz, kel in kept_angles:
            if abs(((az - kaz + 180) % 360) - 180) < az_thresh and abs(el - kel) < el_thresh:
                return False
        return True

    kept_angles = []
    for f in kept:
        C = cam_center_from_Rt(f["R"], f["t"])
        az, el = az_el_from_xyz(C)
        kept_angles.append((az, el))

    # Only consider frames with err <= err_max not already in kept
    cand = [f for f in frames_pool if f not in kept and f["reproj_err_px"] <= err_max]
    # Sort by (error asc, farther azimuth from existing bins first is optional) — keep simple
    cand.sort(key=lambda x: x["reproj_err_px"])

    for f in cand:
        C = cam_center_from_Rt(f["R"], f["t"])
        az, el = az_el_from_xyz(C)
        if sep_ok(az, el, kept_angles):
            kept.append(f)
            kept_angles.append((az, el))
    return kept

def filter_poses(d: Dict[str, Any],
                 hard_err_cut: float = 5.0,
                 core_err: float = 2.0,
                 soft_err_max: float = 3.0,
                 min_dist: float = 0.25,   # meters (if your t is in mm, we’ll convert)
                 max_dist: float = 2.0,    # meters
                 az_thresh: float = 12.0,  # degrees
                 el_thresh: float = 7.0,   # degrees
                 assume_t_in_mm: bool = True) -> Dict[str, Any]:

    frames = list(d.get("frames", []))

    # Basic sanitation: det(R), finite, forward-facing-ish
    clean = []
    dropped_basic = 0
    for f in frames:
        R = f["R"]; t = f["t"]
        if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t)) and det_ok(R)):
            dropped_basic += 1; continue
        if not forward_facing_ok(R):
            dropped_basic += 1; continue
        clean.append(f)

    # Hard error cutoff
    clean = [f for f in clean if f["reproj_err_px"] <= hard_err_cut]
    # Distance gate on camera center
    kept0 = []
    for f in clean:
        C = cam_center_from_Rt(f["R"], f["t"])  # same unit as t
        dist = float(np.linalg.norm(C))
        # Convert to meters if t was in mm
        if assume_t_in_mm:
            dist_m = dist / 1000.0
        else:
            dist_m = dist
        if (min_dist <= dist_m <= max_dist):
            f["_dist_m"] = dist_m
            kept0.append(f)

    # Core set ≤ core_err
    core = [f for f in kept0 if f["reproj_err_px"] <= core_err]
    # Greedy add 2–3 px for coverage
    final = greedy_cover_add(kept0, core, az_thresh=az_thresh, el_thresh=el_thresh, err_max=soft_err_max)

    # Sort final by filename (stable)
    final_sorted = sorted(final, key=lambda f: f.get("file",""))

    # Build output json with same schema
    out = {
        "K_new": d.get("K_new"),
        "dist": d.get("dist", [0,0,0,0,0]),
        "tag_spec": d.get("tag_spec"),
        "frames": [
            {
                "file": f["file"],
                "reproj_err_px": float(f["reproj_err_px"]),
                "R": np.asarray(f["R"]).tolist(),
                "t": np.asarray(f["t"]).reshape(3,1).tolist(),
            }
            for f in final_sorted
        ],
    }

    # Stats
    def coverage_stats(frames_list: List[Dict[str,Any]]) -> Dict[str,float]:
        if not frames_list: return {"count":0}
        az = []; el = []; dist = []
        for f in frames_list:
            C = cam_center_from_Rt(f["R"], f["t"])
            a,e = az_el_from_xyz(C); az.append(a); el.append(e)
            dist.append(float(np.linalg.norm(C)) / (1000.0 if assume_t_in_mm else 1.0))
        return {
            "count": len(frames_list),
            "az_min": float(min(az)), "az_max": float(max(az)),
            "el_min": float(min(el)), "el_max": float(max(el)),
            "dist_m_min": float(min(dist)), "dist_m_max": float(max(dist)),
            "err_px_mean": float(np.mean([f["reproj_err_px"] for f in frames_list])),
            "err_px_p95": float(np.percentile([f["reproj_err_px"] for f in frames_list], 95)),
        }

    stats_in  = coverage_stats(frames)
    stats_out = coverage_stats(final_sorted)

    print("=== Pose Post-Filter Summary ===")
    print(f"Input frames: {stats_in.get('count',0)}")
    print(f"  Azimuth range: {stats_in.get('az_min','?'):.1f}..{stats_in.get('az_max','?'):.1f} deg")
    print(f"  Elevation range: {stats_in.get('el_min','?'):.1f}..{stats_in.get('el_max','?'):.1f} deg")
    print(f"  Dist range: {stats_in.get('dist_m_min','?'):.3f}..{stats_in.get('dist_m_max','?'):.3f} m")
    print(f"  Err mean / p95: {stats_in.get('err_px_mean','?'):.3f} / {stats_in.get('err_px_p95','?'):.3f} px")
    print()
    print(f"Filtered frames: {stats_out.get('count',0)}  (basic rejects: {dropped_basic})")
    print(f"  Azimuth range: {stats_out.get('az_min','?'):.1f}..{stats_out.get('az_max','?'):.1f} deg")
    print(f"  Elevation range: {stats_out.get('el_min','?'):.1f}..{stats_out.get('el_max','?'):.1f} deg")
    print(f"  Dist range: {stats_out.get('dist_m_min','?'):.3f}..{stats_out.get('dist_m_max','?'):.3f} m")
    print(f"  Err mean / p95: {stats_out.get('err_px_mean','?'):.3f} / {stats_out.get('err_px_p95','?'):.3f} px")

    return out

def main():
    ap = argparse.ArgumentParser(description="Filter PnP pose JSON by error, geometry, and view coverage.")
    ap.add_argument("--in_json", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--hard_err_cut", type=float, default=5.0, help="Drop anything above this px error.")
    ap.add_argument("--core_err", type=float, default=2.0, help="Always keep poses at or below this error.")
    ap.add_argument("--soft_err_max", type=float, default=3.0, help="Allow poses up to this error if they add new az/el coverage.")
    ap.add_argument("--min_dist_m", type=float, default=0.35, help="Minimum camera distance (meters).")
    ap.add_argument("--max_dist_m", type=float, default=1.30, help="Maximum camera distance (meters).")
    ap.add_argument("--az_thresh_deg", type=float, default=12.0, help="Min azimuth separation to count as new coverage.")
    ap.add_argument("--el_thresh_deg", type=float, default=7.0, help="Min elevation separation to count as new coverage.")
    ap.add_argument("--t_in_mm", action="store_true", help="If set, interpret t vectors as millimeters (default true).")
    ap.add_argument("--t_in_m", action="store_true", help="If set, interpret t vectors as meters (overrides --t_in_mm).")
    args = ap.parse_args()

    assume_t_in_mm = True
    if args.t_in_m:
        assume_t_in_mm = False
    elif args.t_in_mm:
        assume_t_in_mm = True

    d = load_poses(args.in_json)
    out = filter_poses(
        d,
        hard_err_cut=args.hard_err_cut,
        core_err=args.core_err,
        soft_err_max=args.soft_err_max,
        min_dist=args.min_dist_m,
        max_dist=args.max_dist_m,
        az_thresh=args.az_thresh_deg,
        el_thresh=args.el_thresh_deg,
        assume_t_in_mm=assume_t_in_mm,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote filtered poses to: {args.out_json}")

if __name__ == "__main__":
    main()
