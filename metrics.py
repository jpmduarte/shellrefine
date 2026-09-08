import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def dice_from_sdf(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Signed distance: the interior is the negative side, not values above a threshold.
    return dice((pred < 0).float(), (target < 0).float())


def dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    # Returns None when the target has no positive voxels.
    pred_bin   = (pred   >= threshold).float()
    target_bin = (target >= threshold).float()

    intersection = (pred_bin * target_bin).sum().item()
    union        = pred_bin.sum().item() + target_bin.sum().item()

    if union == 0:
        return None

    return (2.0 * intersection) / union
