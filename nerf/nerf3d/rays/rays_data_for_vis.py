# ==== rays_data_for_vis.py  (you can also paste into a notebook cell) ====
from __future__ import annotations
import numpy as np
import torch

# Use your repo's ray util so conventions match training
from nerf.nerf3d.rays.ray_sampler import pixels_to_rays_batched

class RaysData:
    """
    Build per-pixel rays for a multiview set.
    - images: (N,H,W,3) uint8 or float in [0,1]
    - K:      (3,3) intrinsics (shared) or (N,3,3) per-image
    - c2ws:   (N,4,4) camera-to-world (OpenGL, like your training)
    Exposes:
      - uvs:    (N*H*W, 2) int32 XY coords (x=col, y=row) in [0..W-1],[0..H-1]
      - pixels: (N*H*W, 3) float32 in [0,1]
      - rays_o: (N*H*W, 3) float32
      - rays_d: (N*H*W, 3) float32 (normalized)
      - img_idx:(N*H*W,)  int32 which image each ray came from
    """
    def __init__(self, images: np.ndarray, K: np.ndarray, c2ws: np.ndarray):
        assert images.ndim == 4 and images.shape[-1] == 3, "images must be (N,H,W,3)"
        assert c2ws.ndim == 3 and c2ws.shape[1:] == (4,4), "c2ws must be (N,4,4)"
        self.N, self.H, self.W, _ = images.shape

        # pixels → float32 [0,1]
        if images.dtype == np.uint8:
            pix = images.astype(np.float32) / 255.0
        else:
            pix = np.clip(images.astype(np.float32), 0.0, 1.0)
        self._pixels_all = pix   # (N,H,W,3)

        # intrinsics broadcast to per-image
        if K.ndim == 2:
            K = K[None, ...].repeat(self.N, axis=0)
        assert K.shape == (self.N,3,3), "K must be (N,3,3) after broadcast"
        self._K_all = K.astype(np.float32)
        self._c2w_all = c2ws.astype(np.float32)

        # build per-pixel XY grid (x=column, y=row), then tile across images
        xs = np.arange(self.W, dtype=np.int32)
        ys = np.arange(self.H, dtype=np.int32)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")       # (H,W)
        uv = np.stack([gx, gy], axis=-1).reshape(-1, 2)   # (H*W,2), XY order

        self.uvs = np.tile(uv, (self.N, 1))               # (N*H*W, 2)
        self.img_idx = np.repeat(np.arange(self.N, dtype=np.int32), self.H*self.W)  # (N*H*W,)

        # gather per-pixel colors aligned with uvs/img_idx
        col = self.uvs[:,0]; row = self.uvs[:,1]
        self.pixels = self._pixels_all[self.img_idx, row, col, :]                  # (M,3) M=N*H*W

        # compute rays via your torch util, then back to numpy
        # we do it in per-image chunks to avoid massive tensors
        chunks_o = []
        chunks_d = []
        uv_per_img = uv  # (H*W,2) for one image
        for i in range(self.N):
            Ki   = torch.as_tensor(self._K_all[i],   dtype=torch.float32, device="cpu")     # (3,3)
            c2wi = torch.as_tensor(self._c2w_all[i], dtype=torch.float32, device="cpu")     # (4,4)
            pix_t = torch.as_tensor(uv_per_img,      dtype=torch.float32, device="cpu")     # (HW,2)
            ro, rd = pixels_to_rays_batched(pix_t, Ki, c2wi, pixel_center=True)             # (HW,3),(HW,3)

            # normalize directions (safe; your sampler often returns normalized already)
            rd = rd / (torch.linalg.norm(rd, dim=-1, keepdim=True).clamp(min=1e-8))
            chunks_o.append(ro.cpu().numpy())
            chunks_d.append(rd.cpu().numpy())

        rays_o = np.concatenate(chunks_o, axis=0)  # (N*H*W, 3)
        rays_d = np.concatenate(chunks_d, axis=0)  # (N*H*W, 3)
        self.rays_o = rays_o.astype(np.float32)
        self.rays_d = rays_d.astype(np.float32)

    def sample_rays(self, B: int, from_image: int | None = None, rng: np.random.Generator | None = None):
        """Return (rays_o, rays_d, pixels), each (B,*)"""
        if rng is None:
            rng = np.random.default_rng()

        if from_image is None:
            idx = rng.integers(0, self.rays_o.shape[0], size=B, dtype=np.int64)
        else:
            assert 0 <= from_image < self.N
            start = from_image * (self.H*self.W)
            stop  = start + (self.H*self.W)
            idx   = rng.integers(start, stop, size=B, dtype=np.int64)

        return self.rays_o[idx], self.rays_d[idx], self.pixels[idx]


def sample_along_rays(rays_o: np.ndarray,
                      rays_d: np.ndarray,
                      n_samples: int = 64,
                      near: float = 2.0,
                      far: float = 6.0,
                      perturb: bool = True,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Return 3D points along rays: (B, n_samples, 3)
    """
    if rng is None:
        rng = np.random.default_rng()
    B = rays_o.shape[0]
    # Edges & midpoints per-ray, like training
    t_edges = np.linspace(near, far, n_samples+1, dtype=np.float32)[None, :]  # (1,S+1)
    if perturb:
        # Stratified jitter within each interval
        deltas = t_edges[:, 1:] - t_edges[:, :-1]                              # (1,S)
        # uniform jitter in [0,1)
        jitter = rng.random((B, n_samples), dtype=np.float32)
        t_mid = t_edges[:, :-1] + deltas * jitter                              # (B,S) broadcast
    else:
        t_mid = 0.5 * (t_edges[:, :-1] + t_edges[:, 1:])                        # (1,S) → (B,S)
        t_mid = np.repeat(t_mid, B, axis=0)

    # points = o + t * d
    pts = rays_o[:, None, :] + t_mid[..., None] * rays_d[:, None, :]
    return pts.astype(np.float32)
S