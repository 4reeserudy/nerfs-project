# nerf/learn2d/models.py
from __future__ import annotations
import torch
import torch.nn as nn


class ImageMLPReLU(nn.Module):
    """
    MLP for learning a 2D image mapping:
        (x,y) --PE--> features --MLP--> RGB

    Architecture:
        Linear(in_dim → W) + ReLU
        Linear(W → W) + ReLU
        Linear(W → W) + ReLU
        Linear(W → 3) + ReLU        # last hidden
        Linear(3 → 3) + Sigmoid     # RGB ∈ [0, 1]
    """

    def __init__(self, in_dim: int, width: int = 128) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ReLU(inplace=True),

            nn.Linear(width, width),
            nn.ReLU(inplace=True),

            nn.Linear(width, width),
            nn.ReLU(inplace=True),

            nn.Linear(width, 3),
            nn.ReLU(inplace=True),

            nn.Linear(3, 3),
            nn.Sigmoid(),
        )

        def _init(m):
            import torch.nn.init as I
            if isinstance(m, nn.Linear):
                I.xavier_uniform_(m.weight, gain=1.0)
                I.zeros_(m.bias)

        self.net.apply(_init)

        # Find the last Linear (before Sigmoid) and randomize its bias
        import torch.nn as nn
        last_linear = None
        for m in reversed(self.net):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is not None:
            nn.init.uniform_(last_linear.bias, -2.0, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) positional-encoded coords
        Returns:
            (N, 3) RGB in [0,1]
        """
        return self.net(x)


def make_mlp_relu(in_dim: int, width: int = 128) -> ImageMLPReLU:
    """Factory function to create the MLP."""
    return ImageMLPReLU(in_dim, width)
