# nerf/learn2d/train_one.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import argparse
import json

import torch

# local imports (assumed present)
from nerf.learn2d.dataset import PixelBatcher, make_coords, load_image_rgb, split_indices, flatten_colors
from nerf.learn2d.pe import fourier_encode, output_dim
from nerf.learn2d.models import make_mlp_relu
from nerf.learn2d.losses import mse, psnr
from nerf.learn2d.utils import set_seed, ensure_dir, save_image, short_run_name, write_json


# ---------------- CLI ----------------

def parse_args() -> Any:
    """
    Returns args with fields:
      image_path: str
      L: int
      W: int
      seed: int
      iters: int
      batch_size: int
      lr: float
      val_frac: float
      device: str           # e.g., 'cuda:1' or 'cpu'
      amp: bool
      log_every: int        # validate/log cadence
      snap_epochs: str      # e.g., '10,100,300,1000'
      save_dir: str         # root for results
    """
    p = argparse.ArgumentParser("Learn2D trainer")
    p.add_argument("--image_path", type=Path, required=True)
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--device", type=str, default=None)      # e.g., "cuda:1" or "cpu"
    p.add_argument("--amp", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--snap_epochs", type=str, default="10,100,300,1000")  # keep as string; parse later
    p.add_argument("--save_dir", type=Path, default=Path("results/learn2d"))
    return p.parse_args()


# -------------- Setup / Build ---------

def setup_device(device_str: str | None = None) -> torch.device:
    """
    Resolve computation device:
      - if CUDA available → use provided or default to cuda:0
      - else fallback to CPU.
    """
    if torch.cuda.is_available():
        if device_str is not None:
            device = torch.device(device_str)
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"[Device] Using {device}")
    return device


def setup_run_dir(image_path: Path, L: int, W: int, seed: int, save_root: Path) -> Path:
    # no seed in folder name
    img_stem = image_path.stem
    run_dir = save_root / img_stem / f"L{L}_W{W}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RunDir] {run_dir}")
    return run_dir


def build_model(in_dim: int, width: int, device: torch.device) -> torch.nn.Module:
    """
    Create MLP model and move to target device.
    """
    model = make_mlp_relu(in_dim, width=width)
    model.to(device)
    print(f"[Model] Built MLP: in_dim={in_dim}, width={width}")
    return model


def build_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Simple Adam optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[Optim] Adam(lr={lr})")
    return optimizer


# -------------- Rendering / Eval ------

def render_full_image(
    model: torch.nn.Module,
    H: int,
    W: int,
    L: int,
    device: torch.device,
    amp: bool = False,
) -> torch.Tensor:
    """
    Return (H, W, 3) float tensor in [0,1] on CPU.
    Uses small chunks to avoid GPU OOM for large images.
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        coords = make_coords(H, W).to(device)  # (H*W, 2)
        N = coords.shape[0]
        out = torch.empty((N, 3), dtype=torch.float32, device=device)

        # modest chunking (≈256K pixels per batch)
        CHUNK = 262_144
        if device.type != "cuda":
            CHUNK = 1_000_000  # CPU can do bigger chunks

        start = 0
        while start < N:
            end = min(start + CHUNK, N)
            c = coords[start:end]
            feats = fourier_encode(c, L)
            if amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    pred = model(feats)
            else:
                pred = model(feats)
            out[start:end] = pred
            start = end

        img = out.view(H, W, 3).detach().cpu()

    if was_training:
        model.train()
    return img


def eval_val_psnr(
    model: torch.nn.Module,
    coords_val: torch.Tensor,   # (Nv, 2) on CPU
    colors_val: torch.Tensor,   # (Nv, 3) on CPU
    L: int,
    device: torch.device,
    amp: bool = False,
) -> float:
    """Compute PSNR on the full validation set (no grad)."""
    was_training = model.training
    model.eval()

    with torch.no_grad():
        c = coords_val.to(device, non_blocking=True)
        y = colors_val.to(device, non_blocking=True)
        feats = fourier_encode(c, L)
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                pred = model(feats)
        else:
            pred = model(feats)
        val_psnr = float(psnr(pred, y).detach().cpu().item())

    if was_training:
        model.train()
    return val_psnr


# -------------- Train Loop ------------

def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    coords: torch.Tensor,                 # (N, 2) in [0,1], CPU
    colors: torch.Tensor,                 # (N, 3) in [0,1], CPU
    train_idx: torch.Tensor,              # (N_train,), CPU long/int
    val_data: Tuple[torch.Tensor, torch.Tensor],  # (coords_val, colors_val), CPU
    H: int,
    W: int,
    L: int,
    iters: int,
    batch_size: int,
    log_every: int,
    snap_epochs: List[int],
    run_dir: Path,
    device: torch.device,
    amp: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    model.train()

    # Fixed small subset for train-PSNR estimation (keeps logging cheap)
    N_train = train_idx.numel()
    eval_k = min(5000, N_train)
    g = torch.Generator().manual_seed(seed)
    eval_idx = train_idx[torch.randperm(N_train, generator=g)[:eval_k]]

    coords_train = coords[train_idx]    # (N_train, 2)
    colors_train = colors[train_idx]    # (N_train, 3)
    coords_val, colors_val = val_data   # CPU tensors

    # Sampler state
    perm = train_idx[torch.randperm(N_train, generator=g)]
    ptr = 0

    # AMP scaler (optional)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    # Logs
    log_iters: List[int] = []
    log_train_psnr: List[float] = []
    log_val_psnr: List[float] = []

    # Helper to fetch a batch
    def next_batch():
        nonlocal ptr, perm
        if ptr + batch_size > N_train:
            perm = train_idx[torch.randperm(N_train, generator=g)]
            ptr = 0
        idx = perm[ptr:ptr + batch_size]
        ptr += batch_size
        return coords[idx], colors[idx]

    # Precompute train eval tensors (fixed) on device
    coords_train_eval = coords[eval_idx].to(device, non_blocking=True)
    colors_train_eval = colors[eval_idx].to(device, non_blocking=True)

    for t in range(1, iters + 1):
        cb_cpu, yb_cpu = next_batch()
        cb = cb_cpu.to(device, non_blocking=True)
        yb = yb_cpu.to(device, non_blocking=True)

        feats_b = fourier_encode(cb, L)

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            with torch.cuda.amp.autocast():
                pred_b = model(feats_b)
                loss = mse(pred_b, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_b = model(feats_b)
            loss = mse(pred_b, yb)
            loss.backward()
            optimizer.step()

        # Logging / validation
        if (t % log_every) == 0 or (t == iters):
            # Train PSNR (on fixed small subset)
            with torch.no_grad():
                feats_te = fourier_encode(coords_train_eval, L)
                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        pred_te = model(feats_te)
                else:
                    pred_te = model(feats_te)
                train_psnr_val = float(psnr(pred_te, colors_train_eval).detach().cpu().item())

            # Val PSNR (full)
            val_psnr_val = eval_val_psnr(
                model, val_data[0], val_data[1], L, device, amp=scaler.is_enabled()
            )

            log_iters.append(int(t))
            log_train_psnr.append(train_psnr_val)
            log_val_psnr.append(val_psnr_val)

        # Snapshots
        if t in snap_epochs:
            img = render_full_image(model, H, W, L, device, amp=scaler.is_enabled())
            save_image(run_dir / f"recon_epoch_{t:04d}.png", img, H, W)

    # Final render
    final_img = render_full_image(model, H, W, L, device, amp=scaler.is_enabled())
    save_image(run_dir / "recon_final.png", final_img, H, W)

    # Build log dict compatible with viz.py
    cfg = {
        "image": run_dir.parent.name,  # run_dir: .../<image>/<L...>
        "L": int(L),
        "W": int(model.net[0].out_features if hasattr(model, "net") else 0),
        "seed": int(seed),
        "iters": int(iters),
        "batch_size": int(batch_size),
        "lr": float(optimizer.param_groups[0].get("lr", 0.0)),
    }

    return {
        "iters": log_iters,
        "train_psnr": log_train_psnr,
        "val_psnr": log_val_psnr,
        "config": cfg,
        "H": int(H),
        "W": int(W),
    }


# -------------- Logging ---------------

def save_run_log(run_dir: Path, run_log: Dict[str, Any]) -> None:
    out_path = run_dir / "run_log.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(run_log, f, indent=2)



# -------------- Main ------------------

def main():
    # ---------------- Parse CLI ----------------
    args = parse_args()
    snaps = [int(s) for s in args.snap_epochs.split(",") if s.strip()]

    # ---------------- Setup ----------------
    device = setup_device(args.device)
    run_dir = setup_run_dir(args.image_path, args.L, args.W, args.seed, args.save_dir)

    img, H, W = load_image_rgb(Path(args.image_path))  # your helper returns (img, H, W)

    # ensure tensor in [0,1] for downstream
    import torch
    if isinstance(img, tuple):
        raise RuntimeError("load_image_rgb returned unexpected nested tuple.")
    if not torch.is_tensor(img):
        # assume numpy uint8 HxWx3 → tensor float in [0,1]
        import numpy as np
        if isinstance(img, np.ndarray) and img.dtype == np.uint8:
            img = torch.from_numpy(img).float() / 255.0
        else:
            # last-resort: try to convert generically
            img = torch.as_tensor(img).float()
            if img.max() > 1.0:
                img = img / 255.0  # best-effort normalize

    coords = make_coords(H, W)        # (H*W, 2) in [0,1]
    colors = flatten_colors(img)      # (H*W, 3) in [0,1]
    coords, colors = coords.to("cpu"), colors.to("cpu")  # keep master copy CPU

    # Split
    train_idx, val_idx = split_indices(len(coords), args.val_frac, args.seed)
    train_idx = train_idx.to("cpu")
    val_idx = val_idx.to("cpu")
    coords_val = coords[val_idx]
    colors_val = colors[val_idx]

    # ---------------- Build Model & Optimizer ----------------
    in_dim = 2 + 4 * args.L
    model = build_model(in_dim, args.W, device)
    optimizer = build_optimizer(model, args.lr)

    # ---------------- Train ----------------
    run_log = train_loop(
        model=model,
        optimizer=optimizer,
        coords=coords,
        colors=colors,
        train_idx=train_idx,
        val_data=(coords_val, colors_val),
        H=H,
        W=W,
        L=args.L,
        iters=args.iters,
        batch_size=args.batch_size,
        log_every=args.log_every,
        snap_epochs=snaps,
        run_dir=run_dir,
        device=device,
        amp=args.amp,
        seed=args.seed,
    )

    # ---------------- Save Logs ----------------
    save_run_log(run_dir, run_log)

    print(f"\n✅ Training complete. Results saved to: {run_dir}\n")


if __name__ == "__main__":
    main()
