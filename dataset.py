import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

SHELLS      = "C:/Users/user/Desktop/boundary_first_then_refine/shells"
SPLITS_JSON = "C:/Users/user/Desktop/boundary_first_then_refine/splits.json"


def _augment(img: np.ndarray, mask: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    # Random flip on each axis independently
    for axis in range(3):
        if rng.random() < 0.5:
            img    = np.flip(img,    axis)
            mask   = np.flip(mask,   axis)
            target = np.flip(target, axis)

    # Random 90° rotation in a randomly chosen plane
    axes = rng.choice([(0, 1), (0, 2), (1, 2)])
    k    = rng.integers(0, 4)
    img    = np.rot90(img,    k, axes)
    mask   = np.rot90(mask,   k, axes)
    target = np.rot90(target, k, axes)

    # Intensity jitter on image only — mask and target stay untouched
    img = img * rng.uniform(0.9, 1.1)                        # random scale
    img = img + rng.normal(0.0, 0.05, size=img.shape)        # additive noise

    # np.flip returns views with negative strides — make contiguous for torch
    img    = np.ascontiguousarray(img)
    mask   = np.ascontiguousarray(mask)
    target = np.ascontiguousarray(target)

    return img, mask, target


class ShellDataset(Dataset):

    def __init__(self, paths: list[str], augment: bool = False):
        self.paths   = paths
        self.augment = augment
        self._rng    = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        data = np.load(self.paths[idx])

        img    = data["img"].astype(np.float32)
        mask   = data["mask"].astype(np.float32)
        target = data["boundary_target"].astype(np.float32)

        if self.augment:
            img, mask, target = _augment(img, mask, target, self._rng)

        x = np.stack([img, mask], axis=0).astype(np.float32)  # (2, D, H, W)
        y = target[np.newaxis].astype(np.float32)             # (1, D, H, W)

        return torch.from_numpy(x), torch.from_numpy(y)


def get_splits(val_fraction: float = 0.2, seed: int = 42, shells_dir: str = None, splits_json: str = None):
    _shells_dir  = shells_dir  or SHELLS
    _splits_json = splits_json or SPLITS_JSON

    # Load existing split so train/evaluate always use the same cases.
    if os.path.exists(_splits_json):
        with open(_splits_json) as f:
            saved = json.load(f)
        print(f"Loaded splits from {_splits_json}")
        return saved["train"], saved["val"]

    # First run: generate, save, and return.
    paths = sorted(glob.glob(os.path.join(_shells_dir, "*.npz")))

    if not paths:
        raise RuntimeError(f"No .npz files found in {_shells_dir}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(paths))

    n_val = max(1, int(len(paths) * val_fraction))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    train_paths = [paths[i] for i in train_idx]
    val_paths   = [paths[i] for i in val_idx]

    os.makedirs(os.path.dirname(os.path.abspath(_splits_json)), exist_ok=True)
    with open(_splits_json, "w") as f:
        json.dump({"train": train_paths, "val": val_paths}, f, indent=2)

    print(f"Saved splits to {_splits_json}  (train={len(train_paths)}, val={len(val_paths)})")
    return train_paths, val_paths