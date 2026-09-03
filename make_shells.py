"""
Usage:
python make_shells.py                  # all cases, grid 32
python make_shells.py --limit 3        # quick test on 3 cases
"""

import argparse
import glob
import os
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

IMAGES = "Z:/Images"
LABELS = "Z:/Labels/Head"
OUT = "C:/Users/user/Desktop/boundary_first_then_refine/shells"

def case_id(path: str) -> str:
    name = os.path.basename(path)
    for suffix in ("_0000.nii.gz", ".nii.gz"):
        if name.endswith(suffix):
            return name[:-len(suffix)]

    return name

def resize_to_grid(vol: np.ndarray, grid: int, order: int) -> np.ndarray:
    factors = [grid / n for n in vol.shape]

    return zoom(
        vol,
        factors,
        order=order,
        mode="nearest"
    )

def normalize(img: np.ndarray) -> np.ndarray:
    mean = np.mean(img)
    std = np.std(img)

    if std == 0:
        return np.zeros_like(img, dtype=np.float32)

    return ((img - mean) / std).astype(np.float32)

def boundary_target(mask: np.ndarray, width_vox: float) -> np.ndarray:

    mask = mask.astype(bool)

    # Distance from voxels inside the mask to the boundary.
    dist_inside = distance_transform_edt(mask)

    # Distance from voxels outside the mask to the boundary.
    dist_outside = distance_transform_edt(~mask)

    # Distance to boundary on both sides.
    dist = np.where(
        mask,
        dist_inside,
        dist_outside
    )

    # Continuous map: near boundary -> close to 1, far -> close to 0.
    target = np.clip(
        1.0 - (dist - 1.0) / width_vox,
        0.0,
        1.0
    )

    return target.astype(np.float32)

def build_one(
    img_path: str,
    lab_path: str,
    grid: int,
    width_vox: float,
    out_dir: str,
    verbose: bool = False
) -> None:

    t0 = time.time()

    cid = case_id(img_path)

    # Load image and mask

    img_nii = nib.load(img_path)
    lab_nii = nib.load(lab_path)

    img = np.asanyarray(
        img_nii.dataobj
    ).astype(np.float32)

    mask = np.asanyarray(
        lab_nii.dataobj
    ) > 0

    if not mask.any():
        raise ValueError(
            f"{cid}: empty label"
        )

    spacing = img_nii.header.get_zooms()[:3]

    # Normalize and downsample image

    img_small = resize_to_grid(
        normalize(img),
        grid,
        order=1
    )

    # Downsample mask
    # order=0 to preserve the discrete nature of the mask.

    mask_small = resize_to_grid(
        mask.astype(np.float32),
        grid,
        order=0
    ) > 0.5

    # Phase 1 target

    target = boundary_target(
        mask_small,
        width_vox=width_vox
    )

    elapsed = time.time() - t0

    # Save

    output_path = os.path.join(
        out_dir,
        f"{cid}.npz"
    )

    np.savez_compressed(
        output_path,

        # Phase 1 input
        img=img_small.astype(np.float32),

        # Segmentation ground truth
        mask=mask_small.astype(np.uint8),

        # Phase 1 target
        boundary_target=target.astype(np.float32),

        # Metadata
        spacing=np.array(
            spacing,
            dtype=np.float32
        ),

        native_shape=np.array(
            mask.shape,
            dtype=np.int32
        ),
    )

    if verbose:

        boundary_fraction = (
            target > 0.05
        ).mean()

        print(
            f"  {cid}"
            f"  {elapsed:.1f}s"
            f"  head={100 * mask_small.mean():.1f}%"
            f"  boundary={100 * boundary_fraction:.1f}%"
            f"  target_max={target.max():.3f}",
            flush=True
        )

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=32,
        help="cubic grid size for Phase 1"
    )

    parser.add_argument(
        "--width-vox",
        type=float,
        default=3.0,
        help="boundary band width in downsampled grid voxels"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit number of cases; 0 = all"
    )

    args = parser.parse_args()

    # Create output directory

    os.makedirs(
        OUT,
        exist_ok=True
    )

    # Find images and labels

    imgs = sorted(
        glob.glob(
            os.path.join(
                IMAGES,
                "*.nii.gz"
            )
        )
    )

    labs = {
        case_id(path): path
        for path in glob.glob(
            os.path.join(
                LABELS,
                "*.nii.gz"
            )
        )
    }

    if not imgs:
        raise RuntimeError(
            f"No images found in:\n{IMAGES}"
        )

    if not labs:
        raise RuntimeError(
            f"No labels found in:\n{LABELS}"
        )

    # Check image-label correspondence

    missing = [
        case_id(ip)
        for ip in imgs
        if case_id(ip) not in labs
    ]

    if missing:
        #ignore those labels that are missing, just print a warning
        print(
            f"Warning: {len(missing)} images have no corresponding label:\n"
            f"{missing}\n"
            f"These cases will be skipped.",
            flush=True
        )

    jobs = [
        (
            img_path,
            labs[case_id(img_path)]
        )
        for img_path in imgs
        if case_id(img_path) not in missing
    ]

    if args.limit:

        jobs = jobs[:args.limit]

    # Print summary

    print(
        f"{len(jobs)} cases -> {OUT}\n"
        f"grid={args.grid}³\n"
        f"boundary width={args.width_vox} vox",
        flush=True
    )

    # Process cases

    for i, (img_path, lab_path) in enumerate(
        jobs,
        start=1
    ):

        print(
            f"[{i}/{len(jobs)}]",
            end=" ",
            flush=True
        )

        build_one(
            img_path=img_path,
            lab_path=lab_path,
            grid=args.grid,
            width_vox=args.width_vox,
            out_dir=OUT,
            verbose=True
        )

    print(
        "\nPhase 1 dataset created successfully."
    )

if __name__ == "__main__":
    main()
