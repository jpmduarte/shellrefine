import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(pred, target)


def mse_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target) + F.binary_cross_entropy(pred, target)


LOSSES = {
    "mse":     mse,
    "bce":     bce,
    "mse_bce": mse_bce,
}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSSES)}")
    return LOSSES[name]
