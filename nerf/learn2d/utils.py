# nerf/learn2d/utils.py

import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def to_numpy_img(t: torch.Tensor, H: int, W: int):
    if t.ndim == 2:  # (3, N) case – unlikely here but safe
        t = t.permute(1, 0)
    if t.ndim == 2 and t.shape[1] == 3:
        t = t.view(H, W, 3)
    if t.ndim == 3 and t.shape[0] == 3:
        t = t.permute(1, 2, 0)
    img = torch.clamp(t, 0.0, 1.0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return img


def save_image(path: Path, img: torch.Tensor, H: int, W: int) -> None:
    from PIL import Image
    arr = to_numpy_img(img, H, W)
    ensure_dir(path.parent)
    Image.fromarray(arr).save(path)


def short_run_name(image_name: str, L: int, W: int, seed: int) -> str:
    base = Path(image_name).stem
    return f"{base}_L{L}_W{W}_s{seed}"


class time_block:
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc, tb):
        dur = time.time() - self.start
        print(f"[{self.name}] {dur:.3f}s")
