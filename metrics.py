import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Volumetric binary Dice over a single case.
    Returns None when the target has no positive voxels (no boundary to evaluate).
    """
    pred_bin   = (pred   >= threshold).float()
    target_bin = (target >= threshold).float()

    intersection = (pred_bin * target_bin).sum().item()
    union        = pred_bin.sum().item() + target_bin.sum().item()

    if union == 0:
        return None

    return (2.0 * intersection) / union
