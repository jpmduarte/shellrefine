import json
import os
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CROPS       = os.path.join(BASE_DIR, "crops")
SPLITS_JSON = os.path.join(BASE_DIR, "splits.json")

PATCH_SIZE       = 64
PATCHES_PER_CASE = 8

# Padding value per array. Zero is right for the binary masks, but for the signed
# distance zero means "on the surface" — padding with it would invent a boundary
# along the volume edge. +1 is the truncated "far outside" value.
PAD_VALUES = {"img": 0.0, "sdf_prior": 1.0, "band": 0.0, "mask": 0.0, "coarse_mask": 0.0}


def _augment(img: np.ndarray, prior: np.ndarray, mask: np.ndarray, band: np.ndarray,
             rng: np.random.Generator):
    for axis in range(3):
        if rng.random() < 0.5:
            img   = np.flip(img,   axis)
            prior = np.flip(prior, axis)
            mask  = np.flip(mask,  axis)
            band  = np.flip(band,  axis)

    axes = [(0, 1), (0, 2), (1, 2)][rng.integers(0, 3)]
    k    = rng.integers(0, 4)
    img   = np.rot90(img,   k, axes)
    prior = np.rot90(prior, k, axes)
    mask  = np.rot90(mask,  k, axes)
    band  = np.rot90(band,  k, axes)

    # Intensity jitter on image only — prior, mask and band stay untouched
    img = img * rng.uniform(0.9, 1.1)
    img = img + rng.normal(0.0, 0.05, size=img.shape)

    return (
        np.ascontiguousarray(img),
        np.ascontiguousarray(prior),
        np.ascontiguousarray(mask),
        np.ascontiguousarray(band),
    )


def pad_to_patch(arrays: dict[str, np.ndarray], patch: int) -> dict[str, np.ndarray]:
    """Pad every array up to at least `patch` voxels per axis, each with its own fill."""
    shape = next(iter(arrays.values())).shape
    pads  = [(0, max(0, patch - s)) for s in shape]

    if all(p == (0, 0) for p in pads):
        return arrays

    return {
        name: np.pad(a, pads, mode="constant", constant_values=PAD_VALUES[name])
        for name, a in arrays.items()
    }


def band_patch_origins(band: np.ndarray, count: int, patch: int,
                       rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Pick `count` patch origins whose centre sits on a band voxel."""
    flat = np.flatnonzero(band)

    if flat.size == 0:
        centres = np.array([[s // 2 for s in band.shape]] * count)
    else:
        picked  = flat[rng.integers(0, flat.size, size=count)]
        centres = np.stack(np.unravel_index(picked, band.shape), axis=1)

    half = patch // 2
    return [
        tuple(int(np.clip(c - half, 0, s - patch)) for c, s in zip(centre, band.shape))
        for centre in centres
    ]


class CropDataset(Dataset):
    """One item = a stack of boundary-band patches drawn from a single case."""

    def __init__(self, paths: list[str], patch_size: int = PATCH_SIZE,
                 patches_per_case: int = PATCHES_PER_CASE,
                 augment: bool = False, deterministic: bool = False,
                 boundary_weight_floor: float = 1.0, base_seed: int = None):
        self.paths            = paths
        self.patch_size       = patch_size
        self.patches_per_case = patches_per_case
        self.augment          = augment
        self.deterministic    = deterministic
        # 1.0 (default) disables weighting: every voxel gets weight 1, so training is
        # byte-identical to before this was added. Below 1.0, voxels outside the band
        # (already computed by make_crops from phase 1's own prediction — no separate
        # threshold to keep in sync) are down-weighted instead of dropped.
        self.boundary_weight_floor = boundary_weight_floor
        # A persistent stream (like ShellDataset's self._rng), not a per-call seed: each
        # __getitem__ draws the NEXT per-item seed from it, so patch sampling still varies
        # epoch to epoch under a fixed base_seed instead of freezing to one fixed patch
        # set forever (which is what the deterministic/val path deliberately does via a
        # path-only crc32 seed below). None (default) = unseeded, today's behaviour.
        self._seed_rng = np.random.default_rng(base_seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        data = np.load(path)

        arrays = pad_to_patch({
            "img":       data["img"].astype(np.float32),
            "sdf_prior": data["sdf_prior"].astype(np.float32),
            "band":      data["band"].astype(np.float32),
            "mask":      data["mask"].astype(np.float32),
        }, self.patch_size)

        img, prior, band, mask = (
            arrays["img"], arrays["sdf_prior"], arrays["band"], arrays["mask"]
        )

        if self.deterministic:
            # Fixed seed per case keeps validation patches identical across epochs
            seed = zlib.crc32(os.path.basename(path).encode())
        else:
            # Next value from the persistent per-instance stream, not a fresh OS-entropy
            # draw — makes the whole training run reproducible under a fixed base_seed
            # while still varying patch selection across epochs (see __init__).
            seed = int(self._seed_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed)

        origins = band_patch_origins(band, self.patches_per_case, self.patch_size, rng)

        xs, ys, ws = [], [], []
        for origin in origins:
            sl = tuple(slice(o, o + self.patch_size) for o in origin)
            img_p, prior_p, band_p, mask_p = img[sl], prior[sl], band[sl], mask[sl]

            if self.augment:
                img_p, prior_p, mask_p, band_p = _augment(img_p, prior_p, mask_p, band_p, rng)

            xs.append(np.stack([img_p, prior_p], axis=0))
            ys.append(mask_p[np.newaxis])

            if self.boundary_weight_floor < 1.0:
                w_p = np.where(band_p > 0.5, 1.0, self.boundary_weight_floor).astype(np.float32)
            else:
                w_p = np.ones_like(mask_p, dtype=np.float32)
            ws.append(w_p[np.newaxis])

        x = torch.from_numpy(np.stack(xs).astype(np.float32))  # (K, 2, P, P, P)
        y = torch.from_numpy(np.stack(ys).astype(np.float32))  # (K, 1, P, P, P)
        w = torch.from_numpy(np.stack(ws).astype(np.float32))  # (K, 1, P, P, P)

        return x, y, w


def collate_patches(items):
    return (
        torch.cat([x for x, _, _ in items], dim=0),
        torch.cat([y for _, y, _ in items], dim=0),
        torch.cat([w for _, _, w in items], dim=0),
    )


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
