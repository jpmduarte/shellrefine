import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(pred, target)


def mse_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target) + F.binary_cross_entropy(pred, target)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


def dice_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return dice_loss(pred, target) + F.binary_cross_entropy(pred, target)


LOSSES = {
    "mse":      mse,
    "bce":      bce,
    "mse_bce":  mse_bce,
    "dice":     dice_loss,
    "dice_bce": dice_bce,
}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSSES)}")
    return LOSSES[name]