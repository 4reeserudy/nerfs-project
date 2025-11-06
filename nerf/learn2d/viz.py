# nerf/learn2d/viz.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def load_run_log(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        log = json.load(f)
    # minimal validation
    for k in ["iters", "train_psnr", "val_psnr", "config"]:
        if k not in log:
            raise KeyError(f"run log missing key: {k}")
    for k in ["image", "L", "W", "seed"]:
        if k not in log["config"]:
            raise KeyError(f"run log config missing key: {k}")
    return log


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def plot_psnr_curve(run_log: Dict[str, Any], out_path: Path) -> None:
    iters = run_log["iters"]
    tr = run_log["train_psnr"]
    va = run_log["val_psnr"]
    cfg = run_log["config"]
    title = f"{cfg['image']} — L={cfg['L']}, W={cfg['W']}"
    plt.figure(figsize=(7, 4))
    plt.plot(iters, tr, label="Train PSNR")
    plt.plot(iters, va, label="Val PSNR")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR (dB)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_psnr_overlay(run_logs: List[Dict[str, Any]], out_path: Path) -> None:
    _assert_same_image(run_logs)
    plt.figure(figsize=(7, 4))
    for log in run_logs:
        it = log["iters"]
        va = log["val_psnr"]
        lbl = _label_from_config(log["config"])
        plt.plot(it, va, label=lbl)
    img_name = run_logs[0]["config"]["image"]
    plt.xlabel("Iterations")
    plt.ylabel("Val PSNR (dB)")
    plt.title(f"{img_name} — Val PSNR Overlay")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Progress grid (same run, different epochs)
# -----------------------------------------------------------------------------

def make_progress_grid_by_epoch(
    run_dir: Path,
    epochs: List[int],
    out_path: Path,
    tile_size: Tuple[int, int] = (256, 256),
    pad: int = 12,
    label_prefix: str = "epoch=",
    include_final: bool = True,
) -> None:
    # Expected files
    epoch_files = [_epoch_to_filename(e) for e in epochs]
    labels = [f"{label_prefix}{e}" for e in epochs]

    if include_final:
        epoch_files.append("recon_final.png")
        labels.append("final")

    tiles = _load_and_label_tiles(run_dir, epoch_files, labels, tile_size)
    _compose_row_grid(tiles, out_path, pad)


# -----------------------------------------------------------------------------
# Metrics table
# -----------------------------------------------------------------------------

def collect_final_metrics(run_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for log in run_logs:
        cfg = log["config"]
        iters = log["iters"]
        tr = log["train_psnr"]
        va = log["val_psnr"]
        rows.append({
            "image": cfg["image"],
            "L": cfg["L"],
            "W": cfg["W"],
            "seed": cfg["seed"],
            "iters_final": int(iters[-1]) if len(iters) else 0,
            "train_psnr_final": float(tr[-1]) if len(tr) else float("nan"),
            "val_psnr_final": float(va[-1]) if len(va) else float("nan"),
        })
    return rows


def write_metrics_table(rows: List[Dict[str, Any]], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)


# -----------------------------------------------------------------------------
# Helpers (private)
# -----------------------------------------------------------------------------

def _assert_same_image(run_logs: List[Dict[str, Any]]) -> None:
    names = {log["config"]["image"] for log in run_logs}
    if len(names) != 1:
        raise ValueError(f"overlay requires same image across logs, got: {sorted(names)}")

def _label_from_config(cfg: Dict[str, Any]) -> str:
    return f"L{cfg['L']}_W{cfg['W']}"

def _epoch_to_filename(epoch: int) -> str:
    return f"recon_epoch_{epoch:04d}.png"

def _load_and_label_tiles(
    run_dir: Path,
    epoch_files: List[str],
    labels: List[str],
    tile_size: Tuple[int, int],
) -> List[Image.Image]:
    tiles: List[Image.Image] = []
    for fname, label in zip(epoch_files, labels):
        p = run_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"missing snapshot: {p}")
        img = Image.open(p).convert("RGB")
        img = _fit_into_tile(img, tile_size, bg_rgb=(20, 20, 20))  # keep AR, letterbox
        tiles.append(_draw_label(img, label))
    return tiles


def _fit_into_tile(
    img: Image.Image,
    tile_size: Tuple[int, int],
    bg_rgb: Tuple[int, int, int] = (20, 20, 20),
) -> Image.Image:
    """Resize with preserved aspect ratio; paste centered into a fixed-size tile."""
    tile_w, tile_h = tile_size
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (tile_w, tile_h), bg_rgb)

    scale = min(tile_w / w, tile_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    tile = Image.new("RGB", (tile_w, tile_h), bg_rgb)
    off_x = (tile_w - new_w) // 2
    off_y = (tile_h - new_h) // 2
    tile.paste(img_resized, (off_x, off_y))
    return tile

def _draw_label(tile: Image.Image, text: str) -> Image.Image:
    img = tile.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    tw, th = draw.textsize(text, font=font)
    w, h = img.size
    pad = 8
    box = [w//2 - tw//2 - pad, h - th - 2*pad, w//2 + tw//2 + pad, h - pad]
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((w//2 - tw//2, h - th - pad - 1), text, fill=(255, 255, 255), font=font)
    return img

def _compose_row_grid(
    tiles: List[Image.Image],
    out_path: Path,
    pad: int,
    bg_rgb: Tuple[int, int, int] = (20, 20, 20),
) -> None:
    if not tiles:
        raise ValueError("no tiles to compose")
    w, h = tiles[0].size
    W = w * len(tiles) + pad * (len(tiles) + 1)
    H = h + 2 * pad
    grid = Image.new("RGB", (W, H), bg_rgb)
    x = pad
    for t in tiles:
        grid.paste(t, (x, pad))
        x += w + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
