from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Optional
import torch


# ---------- Image I/O ----------

def load_image_rgb(path: Path) -> Tuple[torch.Tensor, int, int]:
    """
    Load an image as float32 RGB in [0,1].
    Return: (img_rgb, H, W)
      img_rgb: (H, W, 3) torch.float32
    """
    img = Image.open(path).convert("RGB")
    img_t = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3)
    H, W = img_t.shape[:2]
    return img_t, H, W


# ---------- Grid + Flatten ----------

def make_coords(H: int, W: int) -> torch.Tensor:
    """
    Return normalized pixel-center coords (N, 2) in [0,1]^2.
    Convention:
      x = (0.5 + j) / W
      y = (0.5 + i) / H
    with row-major ordering.
    """
    ys = (torch.arange(H, dtype=torch.float32) + 0.5) / H   # (H,)
    xs = (torch.arange(W, dtype=torch.float32) + 0.5) / W   # (W,)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")          # (H, W)
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)   # (N,2)
    return coords


def flatten_colors(img_rgb: torch.Tensor) -> torch.Tensor:
    """
    (H, W, 3) -> (N, 3), same ordering as make_coords().
    """
    return img_rgb.reshape(-1, 3).to(torch.float32)


def split_indices(N: int, val_frac: float = 0.05, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random split of pixel indices into train/val.
    Returns 1D LongTensors: (train_idx, val_idx)
    """
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(N, generator=g)
    n_val = int(N * val_frac)
    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]
    return train_idx.long(), val_idx.long()


# ---------- Split ----------

def split_indices(N: int, val_frac: float = 0.05, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (train_idx, val_idx) as torch.long tensors.
    Ensures both splits are non-empty when possible.
    """
    if N <= 1:
        raise ValueError(f"N must be > 1, got {N}")

    # clamp val fraction to [0.0, 0.9] to avoid degenerate splits
    vf = float(max(0.0, min(val_frac, 0.9)))
    n_val = int(round(vf * N))
    n_val = max(1, min(N - 1, n_val))  # ensure at least 1 val and 1 train

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(N, generator=g)

    val_idx = perm[:n_val].to(torch.long)
    train_idx = perm[n_val:].to(torch.long)
    return train_idx, val_idx


# ---------- Batching ----------

class PixelBatcher:
    """
    Mini-batch iterator over (coords, colors) for given pixel indices.
    """

    def __init__(
        self,
        coords: torch.Tensor,      # (N, 2)
        colors: torch.Tensor,      # (N, 3)
        idxs: torch.Tensor,        # (M,)
        batch_size: int = 10_000,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.coords = coords
        self.colors = colors
        self.idxs = idxs.clone()          # keep a private copy
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.g = torch.Generator()
        if seed is not None:
            self.g.manual_seed(seed)
        self._cursor = 0

    def __iter__(self) -> "PixelBatcher":
        # Reset for new epoch
        self._cursor = 0
        if self.shuffle:
            self.idxs = self.idxs[torch.randperm(len(self.idxs), generator=self.g)]
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # End of epoch
        if self._cursor >= len(self.idxs):
            raise StopIteration

        # Slice indices for this batch
        start = self._cursor
        end = start + self.batch_size
        batch_idxs = self.idxs[start:end]

        # Gather pixels
        coords_b = self.coords[batch_idxs]
        colors_b = self.colors[batch_idxs]

        # Advance pointer
        self._cursor = end

        return coords_b, colors_b



# ---------- Dataset Container ----------

@dataclass
class Learn2DDataset:
    """
    Thin container for one image’s pixel field.
    """
    H: int
    W: int
    coords: torch.Tensor     # (N, 2), float32 in [0,1], on CPU
    colors: torch.Tensor     # (N, 3), float32 in [0,1], on CPU
    train_idx: torch.Tensor  # (N_train,), long on CPU
    val_idx: torch.Tensor    # (N_val,), long on CPU

    def train_loader(
        self,
        batch_size: int = 10_000,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> "PixelBatcher":
        """
        Return a PixelBatcher over training indices.
        """
        return PixelBatcher(
            coords=self.coords,
            colors=self.colors,
            idxs=self.train_idx,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    def val_full(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return full validation tensors: (coords_val, colors_val).
        """
        return self.coords[self.val_idx], self.colors[self.val_idx]


def load_dataset(
    path: Path,
    val_frac: float = 0.05,
    seed: int = 0,
) -> Learn2DDataset:
    """
    Compose: load_image_rgb -> make_coords -> flatten_colors -> split_indices -> Learn2DDataset.
    Prints a one-line summary of shapes/splits.
    """
    img_rgb, H, W = load_image_rgb(path)          # (H, W, 3) float32 in [0,1]
    coords = make_coords(H, W)                    # (N, 2)
    colors = flatten_colors(img_rgb)              # (N, 3)
    N = H * W
    train_idx, val_idx = split_indices(N, val_frac=val_frac, seed=seed)

    ds = Learn2DDataset(
        H=H,
        W=W,
        coords=coords,
        colors=colors,
        train_idx=train_idx,
        val_idx=val_idx,
    )

    # One-line summary (minimal, helpful)
    print(f"[learn2d] {path.name}: HxW={H}x{W}, N={N}, train={len(train_idx)}, val={len(val_idx)}")

    return ds