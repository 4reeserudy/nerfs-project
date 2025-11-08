# nerf/learn2d/viz.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

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
    """
    Compose a 1xN row of snapshots for one run. Keeps aspect ratio with letterboxing.
    Expects files like recon_epoch_0010.png, ..., and optionally recon_final.png.
    """
    # Expected files
    epoch_files = [_epoch_to_filename(e) for e in epochs]
    labels = [f"{label_prefix}{e}" for e in epochs]

    if include_final:
        epoch_files.append("recon_final.png")
        labels.append("final")

    tiles = _load_and_label_tiles(run_dir, epoch_files, labels, tile_size)
    _compose_row_grid(tiles, out_path, pad)


# -----------------------------------------------------------------------------
# Final metrics table
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
# Generic tiling with labels (NEW in v2)
# -----------------------------------------------------------------------------

def tile_images_with_labels(
    image_paths: Sequence[Path],
    labels: Optional[Sequence[str]] = None,
    grid: Tuple[int, int] = (1, 4),
    out_path: Path = Path("tiled.png"),
    tile_h: int = 256,
    pad: int = 8,
    bg: Tuple[float, float, float] = (0.15, 0.15, 0.15),
    label_color: Tuple[int, int, int] = (230, 230, 230),
) -> None:
    """
    Tile arbitrary images into a (rows, cols) grid with labels.
    Keeps aspect ratio per tile (letterboxed), dark-gray background, centered labels.

    Args:
        image_paths: list of image files to place into the grid.
        labels: optional labels of same length; defaults to filenames.
        grid: (rows, cols).
        out_path: output path.
        tile_h: target tile height in pixels (width adapts per image AR).
        pad: padding between tiles and canvas edges.
        bg: background color (float 0..1 or int 0..255).
        label_color: label text color (RGB 0..255).
    """
    rows, cols = grid
    assert rows * cols >= len(image_paths), "grid too small for number of images"
    bg_rgb = _bg_to_rgb(bg)

    # Load images
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    # Fit each image to a uniform tile height; width varies, we will center in slot
    fitted = [_fit_with_aspect_height(im, tile_h) for im in imgs]
    maxw = max((im.size[0] for im in fitted), default=tile_h)

    # Slot size for each tile (same width to align columns)
    slot_w, slot_h = maxw, tile_h + 24  # 24px label band
    canvas_w = pad + cols * (slot_w + pad)
    canvas_h = pad + rows * (slot_h + pad)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_rgb)

    # Labels
    if labels is None:
        labels = [p.stem for p in image_paths]
    else:
        labels = list(labels)
        assert len(labels) == len(image_paths), "labels length mismatch"

    # Paste tiles row-major
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x0 = pad + c * (slot_w + pad)
            y0 = pad + r * (slot_h + pad)
            if idx < len(fitted):
                im = fitted[idx]
                tw, th = im.size
                ox = x0 + (slot_w - tw) // 2
                oy = y0 + (tile_h - th) // 2
                canvas.paste(im, (ox, oy))
                # label
                _draw_label_on_canvas(canvas, labels[idx], x0, y0 + tile_h, slot_w, 24, label_color)
                idx += 1
            else:
                # leave empty slot
                pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


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

def _bg_to_rgb(bg: Tuple[float, float, float] | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(bg[0], float):
        return tuple(int(max(0.0, min(1.0, c)) * 255) + 0 for c in bg)  # type: ignore[index]
    return tuple(int(max(0, min(255, c))) for c in bg)  # type: ignore[index]

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
        img = _fit_into_tile(img, tile_size, bg_rgb=(38, 38, 38))  # keep AR, letterbox
        tiles.append(_draw_label(img, label))
    return tiles

def _fit_into_tile(
    img: Image.Image,
    tile_size: Tuple[int, int],
    bg_rgb: Tuple[int, int, int] = (38, 38, 38),
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

def _fit_with_aspect_height(img: Image.Image, target_h: int) -> Image.Image:
    """Resize to target height, width follows aspect ratio."""
    w, h = img.size
    if h == 0:
        return img
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    return img.resize((new_w, target_h), Image.BILINEAR)

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
    box = [w // 2 - tw // 2 - pad, h - th - 2 * pad, w // 2 + tw // 2 + pad, h - pad]
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((w // 2 - tw // 2, h - th - pad - 1), text, fill=(255, 255, 255), font=font)
    return img

def _draw_label_on_canvas(
    canvas: Image.Image,
    text: str,
    x: int,
    y: int,
    slot_w: int,
    band_h: int,
    color: Tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    tw, th = draw.textsize(text, font=font)
    tx = x + (slot_w - tw) // 2
    ty = y + max(2, (band_h - th) // 2)
    draw.text((tx, ty), text, fill=color, font=font)

def _compose_row_grid(
    tiles: List[Image.Image],
    out_path: Path,
    pad: int,
    bg_rgb: Tuple[int, int, int] = (38, 38, 38),
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
