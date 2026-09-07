import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

CROPS       = "C:/Users/user/Desktop/boundary_first_then_refine/crops"
SPLITS_JSON = "C:/Users/user/Desktop/boundary_first_then_refine/splits.json"


def _augment(img: np.ndarray, prior: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    for axis in range(3):
        if rng.random() < 0.5:
            img   = np.flip(img,   axis)
            prior = np.flip(prior, axis)
            mask  = np.flip(mask,  axis)

    axes = rng.choice([(0, 1), (0, 2), (1, 2)])
    k    = rng.integers(0, 4)
    img   = np.rot90(img,   k, axes)
    prior = np.rot90(prior, k, axes)
    mask  = np.rot90(mask,  k, axes)

    # Intensity jitter on image only — prior and mask stay untouched
    img = img * rng.uniform(0.9, 1.1)
    img = img + rng.normal(0.0, 0.05, size=img.shape)

    img   = np.ascontiguousarray(img)
    prior = np.ascontiguousarray(prior)
    mask  = np.ascontiguousarray(mask)

    return img, prior, mask


class CropDataset(Dataset):

    def __init__(self, paths: list[str], augment: bool = False):
        self.paths   = paths
        self.augment = augment
        self._rng    = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        data  = np.load(self.paths[idx])
        img   = data["img"].astype(np.float32)
        prior = data["boundary_prior"].astype(np.float32)
        mask  = data["mask"].astype(np.float32)

        if self.augment:
            img, prior, mask = _augment(img, prior, mask, self._rng)

        x = np.stack([img, prior], axis=0).astype(np.float32)  # (2, D, H, W)
        y = mask[np.newaxis].astype(np.float32)                 # (1, D, H, W)

        return torch.from_numpy(x), torch.from_numpy(y)


def get_crop_splits(crops_dir: str = None, splits_json: str = None) -> tuple[list[str], list[str]]:
    _crops_dir   = crops_dir   or CROPS
    _splits_json = splits_json or SPLITS_JSON

    if not os.path.exists(_splits_json):
        raise RuntimeError(f"splits.json not found at {_splits_json} — run make_shells.py first")

    with open(_splits_json) as f:
        saved = json.load(f)

    def to_crop(shell_path: str) -> str:
        return os.path.join(_crops_dir, os.path.basename(shell_path))

    train_paths = [to_crop(p) for p in saved["train"] if os.path.exists(to_crop(p))]
    val_paths   = [to_crop(p) for p in saved["val"]   if os.path.exists(to_crop(p))]

    missing = len(saved["train"]) + len(saved["val"]) - len(train_paths) - len(val_paths)
    if missing:
        print(f"Warning: {missing} cases in splits.json have no crop yet — run make_crops.py")

    return train_paths, val_paths
