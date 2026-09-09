import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SHELLS      = os.path.join(BASE_DIR, "shells")
SPLITS_JSON = os.path.join(BASE_DIR, "splits.json")


def _augment(img: np.ndarray, target: np.ndarray, rng: np.random.Generator):
    # Random flip on each axis independently
    for axis in range(3):
        if rng.random() < 0.5:
            img    = np.flip(img,    axis)
            target = np.flip(target, axis)

    # Random 90° rotation in a randomly chosen plane
    axes = [(0, 1), (0, 2), (1, 2)][rng.integers(0, 3)]
    k    = rng.integers(0, 4)
    img    = np.rot90(img,    k, axes)
    target = np.rot90(target, k, axes)

    # Intensity jitter on image only — target stays untouched
    img = img * rng.uniform(0.9, 1.1)                        # random scale
    img = img + rng.normal(0.0, 0.05, size=img.shape)        # additive noise

    # np.flip returns views with negative strides — make contiguous for torch
    img    = np.ascontiguousarray(img)
    target = np.ascontiguousarray(target)

    return img, target


class ShellDataset(Dataset):

    def __init__(self, paths: list[str], augment: bool = False,
                 boundary_weight_floor: float = 1.0, boundary_weight_thr: float = 0.7,
                 seed: int = None):
        self.paths   = paths
        self.augment = augment
        # None (default) = today's behaviour, unseeded. With num_workers>0 this instance
        # gets forked into every worker before train.py's seed_worker reseeds each copy
        # independently — see train.py for why that extra step is needed.
        self._rng    = np.random.default_rng(seed)
        # 1.0 (default) disables weighting: every voxel gets weight 1, exact fallback.
        # Below 1.0, voxels where the true |sdf| >= thr (the saturated interior/exterior
        # — ~89% of voxels, error there ~8x lower than near the crossing per the
        # diagnostic that motivated this) are down-weighted instead of the near-crossing
        # region. Derived straight from the target — no extra array to carry through
        # augmentation, unlike phase 2's band (which comes from a different model).
        self.boundary_weight_floor = boundary_weight_floor
        self.boundary_weight_thr   = boundary_weight_thr

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        data = np.load(self.paths[idx])

        img    = data["img"].astype(np.float32)
        target = data["sdf"].astype(np.float32)

        if self.augment:
            img, target = _augment(img, target, self._rng)

        # Image only. The mask is supervision, not input: it built the signed
        # distance target, and feeding it back would make the target a function
        # of the input.
        x = img[np.newaxis].astype(np.float32)     # (1, D, H, W)
        y = target[np.newaxis].astype(np.float32)  # (1, D, H, W)

        if self.boundary_weight_floor < 1.0:
            w = np.where(np.abs(y) < self.boundary_weight_thr, 1.0,
                        self.boundary_weight_floor).astype(np.float32)
        else:
            w = np.ones_like(y, dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)


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