# scripts/nerf3d/sanity_check_model.py
from __future__ import annotations
import argparse, torch
from nerf.nerf3d.models.main_model import NeRFMLP

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")          # e.g., cpu, cuda, cuda:0, cuda:1
    p.add_argument("--B", type=int, default=8)         # batch size
    p.add_argument("--L_xyz", type=int, default=10)
    p.add_argument("--L_dir", type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(42)
    device = torch.device(args.device)

    model = NeRFMLP(L_xyz=args.L_xyz, L_dir=args.L_dir, width=256).to(device).eval()

    x = torch.randn(args.B, 3, device=device)          # world coords
    d = torch.randn(args.B, 3, device=device)          # view dirs (will be normalized inside)

    with torch.no_grad():
        pe_x = model.encode_xyz(x)
        pe_d = model.encode_dir(d)
        rgb, sigma = model(x, d)

    print(f"[Device] {device}")
    print(f"[PE] dim_xyz={pe_x.shape[-1]} (expected {3 + 6*args.L_xyz}), dim_dir={pe_d.shape[-1]} (expected {3 + 6*args.L_dir})")
    print(f"[Forward] rgb shape={rgb.shape}, sigma shape={sigma.shape}")
    print(f"[Ranges] rgb∈[{rgb.min().item():.4f}, {rgb.max().item():.4f}], sigma≥{sigma.min().item():.4f} (max {sigma.max().item():.4f})")

if __name__ == "__main__":
    main()
