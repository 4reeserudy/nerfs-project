# nerf/nerf3d/models/main_model.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

# Expect this module to exist (you said you'll copy pe.py here):
# positional_encoding(x: Tensor[B,3], L: int, include_input: bool=True) -> Tensor[B, 3 + 6L]
from nerf.nerf3d.models.pe import positional_encoding


class NeRFMLP(nn.Module):
    """
    NeRF-style MLP:
      Inputs: x_world (B,3), view dir d (B,3)
      Encodings: PE(x) with L_xyz; PE(d̂) with L_dir (d normalized)
      Trunk: 8×Linear(ReLU), width=256, skip-concat PE(x) at layer 5
      Heads:
        sigma(x): Linear(256→1) + ReLU (view-independent)
        rgb(x,d): Linear(256→256)+ReLU → concat PE(d) → Linear(256+dir→128)+ReLU → Linear(128→3)+Sigmoid
    """

    def __init__(self, L_xyz: int = 10, L_dir: int = 4, width: int = 256) -> None:
        super().__init__()
        self.L_xyz = L_xyz
        self.L_dir = L_dir
        self.width = width

        dim_xyz = 3 + 6 * L_xyz
        dim_dir = 3 + 6 * L_dir

        # Trunk (with one skip concat after 4 layers)
        self.l1 = nn.Linear(dim_xyz, width)
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, width)
        self.l4 = nn.Linear(width, width)
        self.l5 = nn.Linear(width + dim_xyz, width)
        self.l6 = nn.Linear(width, width)
        self.l7 = nn.Linear(width, width)
        self.l8 = nn.Linear(width, width)
        self.relu = nn.ReLU(inplace=True)

        # Density head
        self.sigma_head = nn.Linear(width, 1)

        # Color head
        self.color_pre = nn.Linear(width, width)           # 256->256
        self.color_dir = nn.Linear(width + dim_dir, 128)   # (256+dir)->128
        self.color_out = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()

        # (Optional) simple init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def encode_xyz(self, x_world: torch.Tensor) -> torch.Tensor:
        """
        x_world: (B,3) → (B, 3+6*L_xyz)
        """
        return positional_encoding(x_world, L=self.L_xyz, include_input=True)

    def encode_dir(self, d: torch.Tensor) -> torch.Tensor:
        """
        d: (B,3) → normalize → (B, 3+6*L_dir)
        """
        d_unit = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        return positional_encoding(d_unit, L=self.L_dir, include_input=True)

    @torch.no_grad()
    def output_dims(self) -> Tuple[int, int]:
        return 3, 1

    def forward(self, x_world: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x_world: (B,3)
          d:       (B,3)
        Returns:
          rgb:   (B,3) in [0,1]
          sigma: (B,1) >= 0
        """
        pe_x = self.encode_xyz(x_world)  # (B, dim_xyz)
        pe_d = self.encode_dir(d)        # (B, dim_dir)

        # Trunk
        h = self.relu(self.l1(pe_x))
        h = self.relu(self.l2(h))
        h = self.relu(self.l3(h))
        h = self.relu(self.l4(h))

        h = torch.cat([h, pe_x], dim=-1)  # skip concat
        h = self.relu(self.l5(h))
        h = self.relu(self.l6(h))
        h = self.relu(self.l7(h))
        h = self.relu(self.l8(h))

        # Density (view-independent)
        sigma = self.relu(self.sigma_head(h))

        # Color (view-dependent)
        hc = self.relu(self.color_pre(h))
        hcd = torch.cat([hc, pe_d], dim=-1)
        hcd = self.relu(self.color_dir(hcd))
        rgb = self.sigmoid(self.color_out(hcd))

        return rgb, sigma