# nerf/nerf3d/rays/ray_sampler.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterator, Dict, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset


@dataclass
class Scene:
    images: torch.Tensor        # (M,H,W,3) uint8 or float in [0,1]
    Ks: torch.Tensor            # (M,3,3) or (1,3,3)
    c2ws: torch.Tensor          # (M,4,4) OpenGL c2w
    near: float
    far: float
    names: Optional[list[str]] = None


def _normalize_rgb(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        return x.float().div_(255.0)
    return x.float().clamp_(0.0, 1.0)


def _gather_intrinsics(Ks: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if Ks.ndim == 2:  # (3,3) shared
        return Ks.expand(idx.shape[0], -1, -1)
    return Ks.index_select(0, idx)


def _gather_poses(c2ws: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return c2ws.index_select(0, idx)


def _pixels_to_world_rays(
    Ks_b: torch.Tensor,        # (N,3,3)
    c2ws_b: torch.Tensor,      # (N,4,4)
    uv: torch.Tensor,          # (N,2) float (pixel coords, center-adjusted already)
) -> tuple[torch.Tensor, torch.Tensor]:
    # Back-project to camera
    ones = torch.ones((uv.shape[0], 1), device=uv.device, dtype=uv.dtype)
    pix = torch.cat([uv, ones], dim=-1).unsqueeze(-1)             # (N,3,1)
    Kinv = torch.inverse(Ks_b)                                     # (N,3,3)
    d_c = torch.matmul(Kinv, pix).squeeze(-1)                      # (N,3)
    d_c = F.normalize(d_c, dim=-1)

    # Camera -> world
    R = c2ws_b[:, :3, :3]                                          # (N,3,3)
    t = c2ws_b[:, :3, 3]                                           # (N,3)
    d_w = torch.matmul(R, d_c.unsqueeze(-1)).squeeze(-1)           # (N,3)
    d_w = F.normalize(d_w, dim=-1)
    o_w = t                                                        # (N,3)
    return o_w, d_w


class RaySamplerGlobal:
    """
    Flatten-all-pixels logical sampler: sample image index ∝ H*W, then u,v uniform.
    """
    def __init__(self, scene: Scene, seed: Optional[int] = None):
        self.scene = scene
        imgs = scene.images
        assert imgs.ndim == 4 and imgs.shape[-1] == 3, "images must be (M,H,W,3)"
        self.M = imgs.shape[0]
        self.sizes = torch.as_tensor(
            [imgs[i].shape[0] * imgs[i].shape[1] for i in range(self.M)],
            dtype=torch.float32,
        )
        self.probs = (self.sizes / self.sizes.sum()).clamp_min(1e-12)
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
        pixel_center: bool = True,
    ) -> Dict[str, torch.Tensor]:
        imgs = self.scene.images
        # sample image indices
        img_idx = torch.multinomial(self.probs, batch_size, replacement=True, generator=self.rng)  # (N,)

        # sample u,v per chosen image
        Hs = torch.as_tensor([imgs[i].shape[0] for i in range(self.M)], dtype=torch.long)
        Ws = torch.as_tensor([imgs[i].shape[1] for i in range(self.M)], dtype=torch.long)
        H_b = Hs.index_select(0, img_idx)  # (N,)
        W_b = Ws.index_select(0, img_idx)  # (N,)

        v = torch.randint(0, H_b.max().item(), (batch_size,), generator=self.rng)
        u = torch.randint(0, W_b.max().item(), (batch_size,), generator=self.rng)
        # clamp to per-image valid range
        v = torch.minimum(v, H_b - 1)
        u = torch.minimum(u, W_b - 1)

        # gather RGB
        rgb = imgs[img_idx, v, u, :]                 # (N,3)
        rgb = _normalize_rgb(rgb).to(device)

        # pixel centers
        uv_f = torch.stack([u.float(), v.float()], dim=-1)
        if pixel_center:
            uv_f = uv_f + 0.5

        # gather intrinsics & poses
        Ks_b = _gather_intrinsics(self.scene.Ks, img_idx).to(device)
        c2ws_b = _gather_poses(self.scene.c2ws, img_idx).to(device)

        # rays
        uv_f = uv_f.to(device)
        origins, dirs = _pixels_to_world_rays(Ks_b, c2ws_b, uv_f)

        near = torch.full((batch_size,), float(self.scene.near), device=device)
        far  = torch.full((batch_size,), float(self.scene.far), device=device)

        return {
            "origins": origins,
            "dirs": dirs,
            "rgb": rgb,
            "uv": uv_f,
            "img_idx": img_idx.to(device),
            "near": near,
            "far": far,
        }


class RaySamplerPerImage:
    """
    Balanced sampler: choose M images, then N//M pixels per image.
    """
    def __init__(self, scene: Scene, seed: Optional[int] = None, choose_M: Optional[int] = None):
        self.scene = scene
        self.M_total = scene.images.shape[0]
        self.choose_M = choose_M
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
        pixel_center: bool = True,
    ) -> Dict[str, torch.Tensor]:
        imgs = self.scene.images
        M = self.choose_M if self.choose_M is not None else min(self.M_total, max(1, batch_size // 1024))
        M = max(1, min(M, self.M_total))

        # choose M distinct images
        perm = torch.randperm(self.M_total, generator=self.rng)
        chosen = perm[:M]  # (M,)

        # allocate per-image counts
        base = batch_size // M
        rem = batch_size - base * M
        counts = torch.full((M,), base, dtype=torch.long)
        if rem > 0:
            # distribute remainder to first 'rem' images
            counts[:rem] += 1

        # sample pixels per image
        u_list, v_list, img_idx_list = [], [], []
        for i, c in zip(chosen.tolist(), counts.tolist()):
            H, W = imgs[i].shape[0], imgs[i].shape[1]
            v = torch.randint(0, H, (c,), generator=self.rng)
            u = torch.randint(0, W, (c,), generator=self.rng)
            u_list.append(u)
            v_list.append(v)
            img_idx_list.append(torch.full((c,), i, dtype=torch.long))

        u = torch.cat(u_list, dim=0)                 # (N,)
        v = torch.cat(v_list, dim=0)                 # (N,)
        img_idx = torch.cat(img_idx_list, dim=0)     # (N,)

        # gather RGB
        rgb = imgs[img_idx, v, u, :]
        rgb = _normalize_rgb(rgb).to(device)

        # pixel centers
        uv_f = torch.stack([u.float(), v.float()], dim=-1)
        if pixel_center:
            uv_f = uv_f + 0.5

        # intrinsics & poses
        Ks_b = _gather_intrinsics(self.scene.Ks, img_idx).to(device)
        c2ws_b = _gather_poses(self.scene.c2ws, img_idx).to(device)

        # rays
        uv_f = uv_f.to(device)
        origins, dirs = _pixels_to_world_rays(Ks_b, c2ws_b, uv_f)

        near = torch.full((batch_size,), float(self.scene.near), device=device)
        far  = torch.full((batch_size,), float(self.scene.far), device=device)

        return {
            "origins": origins,
            "dirs": dirs,
            "rgb": rgb,
            "uv": uv_f,
            "img_idx": img_idx.to(device),
            "near": near,
            "far": far,
        }


# -----------------------------------------------------------------------------
# Stateless helpers (shared by both policies)
# -----------------------------------------------------------------------------

def gather_rgb(scene: Scene, img_idx: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """
    Gather per-pixel RGB from scene.images.

    Args:
        scene.images: (M,H,W,3) uint8 or float
        img_idx: (N,) long indices into images
        uv: (N,2) pixel coords; if float, will be floored; order = (u, v)

    Returns:
        rgb: (N,3) float32 in [0,1], on same device as scene.images
    """
    imgs = scene.images
    device = imgs.device

    if img_idx.dtype != torch.long:
        img_idx = img_idx.to(torch.long)

    # uv handling
    if uv.dtype.is_floating_point:
        u = torch.floor(uv[..., 0]).to(torch.long)
        v = torch.floor(uv[..., 1]).to(torch.long)
    else:
        u = uv[..., 0].to(torch.long)
        v = uv[..., 1].to(torch.long)

    # Clamp to per-image bounds
    M, H0, W0, _ = imgs.shape  # assumes all images same size; if variable, adapt as needed
    u = torch.clamp(u, 0, W0 - 1)
    v = torch.clamp(v, 0, H0 - 1)

    rgb = imgs[img_idx.to(device), v.to(device), u.to(device), :]  # (N,3)
    if rgb.dtype == torch.uint8:
        rgb = rgb.float().div_(255.0)
    else:
        rgb = rgb.float().clamp_(0.0, 1.0)
    return rgb


@torch.no_grad()
def pixels_to_rays_batched(
    pixels: torch.Tensor,   # (N, 2) integer/float pixels [u, v]
    Ks: torch.Tensor,       # (3,3) or (N,3,3)
    c2ws: torch.Tensor,     # (4,4) or (N,4,4)
    pixel_center: bool = True,
):
    """
    Convert a batch of pixels into rays (origins, directions).
    Accepts single K/c2w or per-ray K/c2w; broadcasts if needed.
    Returns:
        rays_o: (N, 3)
        rays_d: (N, 3) normalized
    """
    # ---- validate pixels ----
    if pixels.ndim != 2 or pixels.shape[-1] != 2:
        raise ValueError(f"`pixels` must be (N,2), got {tuple(pixels.shape)}")
    N = pixels.shape[0]
    dtype = pixels.dtype
    device = pixels.device

    # ---- coerce & broadcast Ks ----
    Ks = torch.as_tensor(Ks, dtype=dtype, device=device)
    if Ks.ndim == 2 and Ks.shape == (3, 3):
        Ks = Ks.unsqueeze(0).expand(N, -1, -1)         # (N,3,3)
    elif Ks.ndim == 3 and (Ks.shape[0] == 1 or Ks.shape[0] == N) and Ks.shape[1:] == (3, 3):
        Ks = Ks.expand(N, -1, -1)                      # (N,3,3)
    else:
        raise ValueError(f"`Ks` must be (3,3) or (N,3,3), got {tuple(Ks.shape)}")

    # ---- coerce & broadcast c2ws ----
    c2ws = torch.as_tensor(c2ws, dtype=dtype, device=device)
    if c2ws.ndim == 2 and c2ws.shape == (4, 4):
        c2ws = c2ws.unsqueeze(0).expand(N, -1, -1)     # (N,4,4)
    elif c2ws.ndim == 3 and (c2ws.shape[0] == 1 or c2ws.shape[0] == N) and c2ws.shape[1:] == (4, 4):
        c2ws = c2ws.expand(N, -1, -1)                  # (N,4,4)
    else:
        raise ValueError(f"`c2ws` must be (4,4) or (N,4,4), got {tuple(c2ws.shape)}")

    # ---- pixel centers ----
    if pixel_center:
        pixels = pixels + 0.5

    # ---- backproject to camera directions ----
    ones = torch.ones((N, 1), dtype=dtype, device=device)
    uv1 = torch.cat([pixels, ones], dim=-1).unsqueeze(-1)     # (N,3,1)
    Kinv = torch.linalg.inv(Ks)                               # (N,3,3)
    dirs_cam = (Kinv @ uv1).squeeze(-1)                       # (N,3)

    # normalize directions in camera space
    dirs_cam = dirs_cam / torch.linalg.norm(dirs_cam, dim=-1, keepdim=True).clamp_min(1e-8)

    # ---- transform to world space ----
    R = c2ws[:, :3, :3]                                       # (N,3,3)
    t = c2ws[:, :3, 3]                                        # (N,3)
    rays_d = (R @ dirs_cam.unsqueeze(-1)).squeeze(-1)         # (N,3)
    rays_o = t                                                # (N,3)

    # normalize final dirs (optional but common)
    rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True).clamp_min(1e-8)
    return rays_o, rays_d

# -----------------------------------------------------------------------------
# PyTorch Dataset wrapper (optional; works locally and on Colab)
# -----------------------------------------------------------------------------

class RaysDataset(IterableDataset):
    """
    Infinite stream of ray batches for training.
    Wraps a RaySampler policy; yields dicts with:
      {origins, dirs, rgb, uv, img_idx, near, far}
    """

    def __init__(
        self,
        scene: Scene,
        policy: str = "global",          # "global" | "per_image"
        batch_size: int = 4096,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        choose_M: Optional[int] = None,  # for per-image policy
        pixel_center: bool = True,
    ):
        super().__init__()
        self.scene = scene
        self.policy = policy
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pixel_center = pixel_center

        if policy == "global":
            self.sampler = RaySamplerGlobal(scene, seed=seed)
        elif policy == "per_image":
            self.sampler = RaySamplerPerImage(scene, seed=seed, choose_M=choose_M)
        else:
            raise ValueError(f"unknown policy: {policy}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # stateless infinite generator
        while True:
            yield self.sampler.sample_batch(
                batch_size=self.batch_size,
                device=self.device,
                pixel_center=self.pixel_center,
            )