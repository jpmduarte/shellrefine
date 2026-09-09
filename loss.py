import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(pred, target)


def mse_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target) + F.binary_cross_entropy(pred, target)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


def dice_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return dice_loss(pred, target) + F.binary_cross_entropy(pred, target)


def boundary_weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    L1 weighted per-voxel, for phase 1's SDF regression. Same all-ones-is-a-no-op
    contract as boundary_weighted_bce below — train.py only calls it when enabled.
    """
    per_voxel = torch.abs(pred - target)
    return (per_voxel * weight).sum() / weight.sum().clamp_min(1e-6)


def gradient_difference_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Sum over the 3 spatial axes of the mean L1 distance between first-order finite
    differences of pred and target. Penalises mismatched local slope, which is largest
    exactly where the SDF is changing fastest (near the surface) — without needing an
    explicit weight mask. Not registered in LOSSES: matching gradients everywhere while
    the whole field is offset by a constant scores zero here, so this only makes sense
    added on top of a value-matching loss like l1, never used alone.
    """
    total = pred.new_zeros(())
    for axis in (2, 3, 4):  # (N, C, D, H, W)
        pred_grad   = pred.diff(dim=axis)
        target_grad = target.diff(dim=axis)
        total = total + torch.abs(pred_grad - target_grad).mean()
    return total


def boundary_weighted_bce(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    BCE weighted per-voxel, meant to be added on top of a base loss (not registered
    in LOSSES — it takes a weight map, unlike every other loss here). With an
    all-ones weight this reduces to plain bce(); train_phase2.py only calls it at
    all when boundary weighting is enabled, so the default run is unaffected.
    """
    per_voxel = F.binary_cross_entropy(pred, target, reduction="none")
    return (per_voxel * weight).sum() / weight.sum().clamp_min(1e-6)


LOSSES = {
    "mse":      mse,
    "l1":       l1,
    "bce":      bce,
    "mse_bce":  mse_bce,
    "dice":     dice_loss,
    "dice_bce": dice_bce,
}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSSES)}")
    return LOSSES[name]