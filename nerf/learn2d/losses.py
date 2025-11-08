# nerf/learn2d/losses.py
import torch

def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over RGB."""
    return torch.mean((pred - target) ** 2)

def psnr_from_mse(mse_val: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """PSNR(dB) from MSE."""
    return -10.0 * torch.log10(mse_val + eps)

def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """PSNR(dB) between predictions and targets."""
    return psnr_from_mse(mse(pred, target), eps)
