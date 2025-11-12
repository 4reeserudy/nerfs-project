# nerf/nerf3d/rays/rays_data_for_vis.py  (v2)
from __future__ import annotations
import numpy as np
import torch

# Match training conventions
from nerf.nerf3d.rays.ray_sampler import pixels_to_rays_batched


class RaysData:
    """
    Build per-pixel rays for a multiview set.

    Args:
        images: (N,H,W,3) uint8 or float in [0,1]
        K:      (3,3) or (N,3,3) intrinsics (OpenGL-style fx,fy,cx,cy)
        c2ws:   (N,4,4) camera-to-world (OpenGL convention)
        device: torch device for intermediate ray math (default 'cpu')
        chunk:  number of pixels processed per image-chunk (limits RAM)
        pixel_center: bool, pass through to pixels_to_rays_batched

    Exposes:
        uvs:     (M,2) int32  -> XY coords (x=col, y=row), row-major
        img_idx: (M,)  int32  -> which image each uv belongs to
        pixels:  (M,3) float32 in [0,1]
        rays_o:  (M,3) float32
        rays_d:  (M,3) float32 (normalized)
    """
    def __init__(
        self,
        images: np.ndarray,
        K: np.ndarray,
        c2ws: np.ndarray,
        *,
        device: str | torch.device = "cpu",
        chunk: int = 262_144,
        pixel_center: bool = True,
    ):
        assert images.ndim == 4 and images.shape[-1] == 3, "images must be (N,H,W,3)"
        assert c2ws.ndim == 3 and c2ws.shape[1:] == (4, 4), "c2ws must be (N,4,4)"
        self.N, self.H, self.W, _ = images.shape
        self.device = torch.device(device)
        self.chunk = int(max(1, chunk))
        self.pixel_center = bool(pixel_center)

        # ---- Pixels to float32 [0,1] (no gamma) ----
        if images.dtype == np.uint8:
            pix = images.astype(np.float32) / 255.0
        else:
            pix = np.clip(images.astype(np.float32), 0.0, 1.0)
        self._pixels_all = pix  # (N,H,W,3)

        # ---- Intrinsics broadcast ----
        if K.ndim == 2:
            K = np.repeat(K[None, ...], self.N, axis=0)
        assert K.shape == (self.N, 3, 3), "K must be (N,3,3) after broadcast"
        self._K_all = K.astype(np.float32)
        self._c2w_all = c2ws.astype(np.float32)

        # ---- UV grid in XY order, row-major flatten ----
        xs = np.arange(self.W, dtype=np.int32)
        ys = np.arange(self.H, dtype=np.int32)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")  # gx=x (cols), gy=y (rows)
        uv_one = np.stack([gx, gy], axis=-1).reshape(-1, 2).astype(np.int32)  # (H*W,2)

        self.uvs = np.tile(uv_one, (self.N, 1))  # (N*H*W,2)
        self.img_idx = np.repeat(np.arange(self.N, dtype=np.int32), self.H * self.W)

        # Colors aligned with (img_idx, uv)
        x = self.uvs[:, 0]
        y = self.uvs[:, 1]
        self.pixels = self._pixels_all[self.img_idx, y, x, :].astype(np.float32)

        # ---- Rays (per-image, chunked) ----
        o_chunks, d_chunks = [], []
        HW = self.H * self.W
        # Pre-build one image's UV tensor (reuse across images)
        uv_one_t = torch.as_tensor(uv_one, dtype=torch.float32, device=self.device)  # (HW,2)

        for i in range(self.N):
            Ki = torch.as_tensor(self._K_all[i], dtype=torch.float32, device=self.device)   # (3,3)
            c2wi = torch.as_tensor(self._c2w_all[i], dtype=torch.float32, device=self.device)  # (4,4)

            # chunk over pixels to bound RAM
            for s in range(0, HW, self.chunk):
                e = min(s + self.chunk, HW)
                uv_s = uv_one_t[s:e]  # (m,2)
                ro, rd = pixels_to_rays_batched(uv_s, Ki, c2wi, pixel_center=self.pixel_center)
                # normalize directions
                rd = rd / (torch.linalg.norm(rd, dim=-1, keepdim=True).clamp(min=1e-8))
                o_chunks.append(ro.detach().cpu().numpy())
                d_chunks.append(rd.detach().cpu().numpy())

        self.rays_o = np.concatenate(o_chunks, axis=0).astype(np.float32)  # (M,3)
        self.rays_d = np.concatenate(d_chunks, axis=0).astype(np.float32)  # (M,3)

    # -------- sampling helpers --------
    def sample_indices(
        self,
        B: int,
        from_image: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return 1D indices into the flattened rays/pixels buffers."""
        if rng is None:
            rng = np.random.default_rng()
        M = self.rays_o.shape[0]
        if from_image is None:
            return rng.integers(0, M, size=B, dtype=np.int64)
        assert 0 <= from_image < self.N
        HW = self.H * self.W
        start = from_image * HW
        stop = start + HW
        return rng.integers(start, stop, size=B, dtype=np.int64)

    def sample_rays(
        self,
        B: int,
        from_image: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        """Return (rays_o, rays_d, pixels) each with shape (B, …)."""
        idx = self.sample_indices(B, from_image=from_image, rng=rng)
        return self.rays_o[idx], self.rays_d[idx], self.pixels[idx]


def sample_along_rays(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    *,
    n_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    perturb: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Return points along rays: (B, n_samples, 3)
    Matches training stratified sampler (linear in t with optional jitter).
    """
    if rng is None:
        rng = np.random.default_rng()
    B = int(rays_o.shape[0])

    # Uniform edges in [near,far]
    t_edges = np.linspace(near, far, n_samples + 1, dtype=np.float32)[None, :]  # (1,S+1)

    if perturb:
        deltas = (t_edges[:, 1:] - t_edges[:, :-1]).astype(np.float32)          # (1,S)
        jitter = rng.random((B, n_samples), dtype=np.float32)                    # (B,S) in [0,1)
        t_mid = t_edges[:, :-1] + deltas * jitter                                # (B,S) via broadcast
    else:
        t_mid = 0.5 * (t_edges[:, :-1] + t_edges[:, 1:])                         # (1,S)
        t_mid = np.repeat(t_mid, B, axis=0)                                      # (B,S)

    pts = rays_o[:, None, :] + t_mid[..., None] * rays_d[:, None, :]            # (B,S,3)
    return pts.astype(np.float32)
