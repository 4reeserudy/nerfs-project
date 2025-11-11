# nerf/nerf3d/train/train_nerf.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.optim import Adam
from PIL import Image

from nerf.nerf3d.data.dataloader import make_loaders, seed_everything
from nerf.nerf3d.models.main_model import NeRFMLP
from nerf.nerf3d.engine.step import forward_batch
from nerf.nerf3d.data.losses import psnr

# ------------------------------- setup ---------------------------------

def setup_run(args) -> Path:
    save_dir = Path(getattr(args, "save_dir", "results/nerf3d"))
    dataset_stem = Path(args.dataset).stem if getattr(args, "dataset", None) else "dataset"
    tag = getattr(args, "tag", None)
    run_name = f"L{args.L_xyz}_Ld{args.L_dir}_W{args.W}" + (f"_{tag}" if tag else "")
    run_dir = save_dir / dataset_stem / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "snaps").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"[RunDir] {run_dir}")
    return run_dir

def setup_model_and_optim(cfg: Dict[str, Any]) -> Tuple[NeRFMLP, Adam, torch.cuda.amp.GradScaler | None]:
    device = torch.device(cfg.get("device", "cpu"))
    model = NeRFMLP(L_xyz=int(cfg["L_xyz"]), L_dir=int(cfg["L_dir"]), width=int(cfg["W"])).to(device)
    optim = Adam(model.parameters(),
                 lr=float(cfg.get("lr", 5e-4)),
                 betas=tuple(cfg.get("betas", (0.9, 0.999))),
                 weight_decay=float(cfg.get("weight_decay", 0.0)))
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[Model] NeRF(Lxyz={cfg['L_xyz']}, Ldir={cfg['L_dir']}, W={cfg['W']}) on {device}")
    print(f"[Optim] Adam(lr={cfg.get('lr', 5e-4)}, amp={use_amp})")
    return model, optim, scaler

# --------------------------- snapshot anchor ---------------------------

def _pixels_to_rays_anchor(pix: torch.Tensor, K: np.ndarray, c2w: np.ndarray, device: torch.device):
    # pix: (N,2) tensor (already on CPU is fine)
    Kt   = torch.from_numpy(K).to(device=device, dtype=torch.float32)       # (3,3)
    c2wt = torch.from_numpy(c2w).to(device=device, dtype=torch.float32)     # (4,4)
    pix  = pix.to(device=device, dtype=torch.float32)                        # (N,2)
    # Let pixels_to_rays_batched broadcast K/c2w to N
    rays_o, rays_d = pixels_to_rays_batched(pix, Kt, c2wt, pixel_center=True)
    return {"rays_o": rays_o, "rays_d": rays_d}

def create_snapshot_anchor(scene: Dict[str, Any], run_dir: Path, device: torch.device, snap_stride: int = 2):
    img = scene["images_val"][0]   # (H,W,3) float
    c2w = scene["c2ws_val"][0]     # (4,4)
    H, W = img.shape[:2]
    fx = fy = float(scene["focal"]); cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
    K = [[fx, 0, cx],[0, fy, cy],[0,0,1]]
    s = max(1, int(snap_stride))
    ys = torch.arange(0, H, s); xs = torch.arange(0, W, s)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack([gx, gy], dim=-1).reshape(-1, 2).cpu().numpy()
    rays = _pixels_to_rays_anchor(pix, K, c2w, device)
    torch.save({"rays": {k: v.cpu() for k, v in rays.items()}}, run_dir / "snaps" / "anchor.pt")
    with open(run_dir / "snaps" / "anchor.json", "w") as f:
        json.dump({"H":H, "W":W, "stride":s, "focal":float(scene["focal"]),
                   "c2w": torch.tensor(c2w).tolist()}, f, indent=2)

def load_snapshot_anchor(run_dir: Path, device: torch.device):
    blob = torch.load(run_dir / "snaps" / "anchor.pt", map_location=device)
    with open(run_dir / "snaps" / "anchor.json","r") as f:
        meta = json.load(f)
    rays = {k: v.to(device) for k, v in blob["rays"].items()}
    return rays, meta

@torch.no_grad()
def maybe_snapshot(model, step: int, cfg: Dict[str, Any], device: torch.device, run_dir: Path):
    anchor_json = run_dir / "snaps" / "anchor.json"
    if not anchor_json.exists(): return
    rays, meta = load_snapshot_anchor(run_dir, device)
    n = rays["o"].shape[0]
    chunk = int(cfg.get("chunk", 8192))
    near, far = float(cfg.get("near",2.0)), float(cfg.get("far",6.0))
    n_samples = int(cfg.get("n_samples",64))
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    outs = []
    for s in range(0, n, chunk):
        sub = {k: v[s:s+chunk] for k, v in rays.items()}
        with torch.cuda.amp.autocast(enabled=amp):
            rgb, _depth, _ = forward_batch(model, sub, n_samples, near, far,
                                           perturb=False, bg_color=0.0, chunk=chunk, amp=amp)
        outs.append(rgb.detach().cpu())
    rgb_all = torch.cat(outs, dim=0).clamp(0,1)
    H,W,stride = int(meta["H"]), int(meta["W"]), int(meta["stride"])
    rh, rw = (H + stride - 1)//stride, (W + stride - 1)//stride
    img = (rgb_all.view(rh, rw, 3).mul(255).byte().numpy())
    im = Image.fromarray(img, mode="RGB")
    if stride > 1: im = im.resize((W, H), Image.NEAREST)
    im.save(run_dir / "snaps" / f"step_{step:06d}.png")

# ------------------------------ validate --------------------------------

@torch.no_grad()
def validate(model: NeRFMLP, loader, cfg: Dict[str, Any], device: torch.device) -> Dict[str, float]:
    model.eval()
    n_samples = int(cfg.get("n_samples", 64))
    near, far = float(cfg.get("near",2.0)), float(cfg.get("far",6.0))
    chunk = cfg.get("chunk", None)
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    total_mse, total_psnr, total = 0.0, 0.0, 0
    for batch in loader:
        rays = {k: v.to(device, non_blocking=True) for k, v in batch["rays"].items()}
        rgb_gt = batch["rgb"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            rgb_pred, _depth, _ = forward_batch(model, rays, n_samples, near, far,
                                                perturb=False, bg_color=0.0, chunk=chunk, amp=amp)
            mse = torch.mean((rgb_pred - rgb_gt) ** 2)
        total_mse += float(mse.item()); total_psnr += float(psnr(mse).item()); total += 1
    return {"val_mse": total_mse / max(total,1), "val_psnr": total_psnr / max(total,1)}

# ------------------------------- main loop ------------------------------

def main():
    p = argparse.ArgumentParser()
    # data/run
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="results/nerf3d")
    p.add_argument("--tag", type=str, default=None)
    # model
    p.add_argument("--L_xyz", type=int, default=10)
    p.add_argument("--L_dir", type=int, default=4)
    p.add_argument("--W", type=int, default=256)
    # render
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--near", type=float, default=2.0)
    p.add_argument("--far", type=float, default=6.0)
    p.add_argument("--chunk", type=int, default=8192)
    # train
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    # cadence
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--snap_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--snap_stride", type=int, default=2)
    # resume
    p.add_argument("--resume", type=str, default=None)

    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    print(f"[Device] {device}")

    # loaders
    train_loader, val_loader, scene = make_loaders(
        args.dataset,
        batch_size=args.batch_size,
        device=device,
        return_scene=True
    )

    # run dir + anchor
    run_dir = setup_run(args)
    anchor_json = run_dir / "snaps" / "anchor.json"
    if not anchor_json.exists():
        create_snapshot_anchor(scene, run_dir, device, snap_stride=args.snap_stride)

    # model/optim
    model, optim, scaler = setup_model_and_optim(vars(args))

    # resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        if scaler and "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        print(f"[Resume] from {args.resume} @ step {start_step}")

    # run log
    run_log = {"iters": [], "train_psnr": [], "val_psnr": []}

    # training (step-based)
    amp = bool(args.amp) and device.type == "cuda"
    n_samples = args.n_samples
    near, far = float(args.near), float(args.far)
    chunk = int(args.chunk)

    train_iter = iter(train_loader)
    for step in range(start_step + 1, args.iters + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        rays = {k: v.to(device, non_blocking=True) for k, v in batch["rays"].items()}
        rgb_gt = batch["rgb"].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            rgb_pred, _depth, _ = forward_batch(
                model, rays, n_samples, near, far, perturb=True, bg_color=0.0, chunk=chunk, amp=amp
            )
            loss = torch.mean((rgb_pred - rgb_gt) ** 2)

        if scaler and amp:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        if step % args.log_every == 0:
            cur_psnr = float(psnr(loss).item())
            print(f"[Train] step={step:06d} loss={loss.item():.6f} psnr={cur_psnr:.2f}")
            run_log["iters"].append(step)
            run_log["train_psnr"].append(cur_psnr)

        if step % args.val_every == 0:
            metrics = validate(model, val_loader, vars(args), device)
            print(f"[Val] step={step:06d} psnr={metrics['val_psnr']:.2f}")
            run_log["val_psnr"].append(metrics["val_psnr"])
            with open(run_dir / "run_log.json", "w") as f:
                json.dump(run_log, f, indent=2)

        if step % args.snap_every == 0:
            maybe_snapshot(model, step, vars(args), device, run_dir)

        if step % args.ckpt_every == 0:
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": (scaler.state_dict() if (scaler and amp) else None),
                "step": step,
                "cfg": vars(args),
            }, run_dir / "checkpoints" / f"step_{step:06d}.pt")

    # final save
    torch.save({"model": model.state_dict(), "cfg": vars(args)},
               run_dir / "checkpoints" / "final.pt")
    with open(run_dir / "run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)
    print("[Done]")

if __name__ == "__main__":
    main()
