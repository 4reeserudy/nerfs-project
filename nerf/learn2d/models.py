# nerf/learn2d/models.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.init as I


class ImageMLPReLU(nn.Module):
    """
    (x,y) -> PE -> MLP -> RGB in [0,1]
    Layers: in->W -> W -> W -> 3 (hidden) -> 3 (output with Sigmoid)
    """

    def __init__(self, in_dim: int, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.ReLU(inplace=True),
            nn.Linear(width, width),  nn.ReLU(inplace=True),
            nn.Linear(width, width),  nn.ReLU(inplace=True),
            nn.Linear(width, 3),      nn.ReLU(inplace=True),  # last hidden has width 3
            nn.Linear(3, 3),
            nn.Sigmoid(),
        )

        # Xavier init for all Linear layers, zero biases by default
        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                I.xavier_uniform_(m.weight, gain=1.0)
                I.zeros_(m.bias)
        self.net.apply(_init)

        # v2: break mid-gray start — randomize bias of the final Linear before Sigmoid
        last_linear = next((m for m in reversed(self.net) if isinstance(m, nn.Linear)), None)
        if last_linear is not None:
            I.uniform_(last_linear.bias, -2.0, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_mlp_relu(in_dim: int, width: int = 128) -> ImageMLPReLU:
    return ImageMLPReLU(in_dim, width)
