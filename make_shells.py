"""
Usage:
python make_shells.py                             # all cases, grid 32
python make_shells.py --limit 3                   # quick test on 3 cases
python make_shells.py --out-dir runs/my_run/shells
"""

import argparse
import glob
import os
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

from profiler import Profiler

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shells")


def case_id(path: str) -> str:
    name = os.path.basename(path)
    for suffix in ("_0000.nii.gz", ".nii.gz"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def resize_to_grid(vol: np.ndarray, grid: int, order: int) -> np.ndarray:
    factors = [grid / n for n in vol.shape]
    return zoom(vol, factors, order=order, mode="nearest")


def normalize(img: np.ndarray) -> np.ndarray:
    mean, std = np.mean(img), np.std(img)
    if std == 0:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mean) / std).astype(np.float32)


def signed_distance(mask: np.ndarray, spacing, trunc_mm: float) -> np.ndarray:
    """
    Signed distance to the mask surface in mm — negative inside, positive outside —
    truncated at +-trunc_mm and normalised to [-1, 1].

    Computed at native resolution on purpose: taking the distance of an already
    downsampled mask would pin the zero level set to the coarse voxel grid, which
    is precisely the quantisation this representation exists to avoid.
    """
    outside = distance_transform_edt(~mask, sampling=spacing)
    inside  = distance_transform_edt(mask,  sampling=spacing)
    return np.clip((outside - inside) / trunc_mm, -1.0, 1.0).astype(np.float32)


def build_one(img_path: str, lab_path: str, grid: int, trunc_mm: float, out_dir: str, verbose: bool = False) -> None:
    t0  = time.time()
    cid = case_id(img_path)

    img_nii = nib.load(img_path)
    lab_nii = nib.load(lab_path)

    img  = np.asanyarray(img_nii.dataobj).astype(np.float32)
    mask = np.asanyarray(lab_nii.dataobj) > 0

    if not mask.any():
        raise ValueError(f"{cid}: empty label")

    spacing = img_nii.header.get_zooms()[:3]

    img_small = resize_to_grid(normalize(img), grid, order=1)
    # The distance field is smooth, so linear sampling is correct on the way down.
    # A binary mask needed order=0; this does not.
    sdf_small = resize_to_grid(signed_distance(mask, spacing, trunc_mm), grid, order=1)

    np.savez_compressed(
        os.path.join(out_dir, f"{cid}.npz"),
        img=img_small.astype(np.float32),
        sdf=sdf_small.astype(np.float32),
        mask=(sdf_small < 0).astype(np.uint8),
        spacing=np.array(spacing, dtype=np.float32),
        native_shape=np.array(mask.shape, dtype=np.int32),
        trunc_mm=np.float32(trunc_mm),
    )

    elapsed = time.time() - t0

    if verbose:
        print(
            f"  {cid}  {elapsed:.1f}s"
            f"  head={100 * (sdf_small < 0).mean():.1f}%"
            f"  sdf=[{sdf_small.min():.2f}, {sdf_small.max():.2f}]",
            flush=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=str,   required=True, help="directory containing input .nii.gz images")
    parser.add_argument("--labels-dir", type=str,   required=True, help="directory containing label .nii.gz files")
    parser.add_argument("--grid",      type=int,   default=32,  help="cubic grid size for Phase 1")
    parser.add_argument("--trunc-mm",  type=float, default=10.0, help="signed distance truncation in mm; the field is stored as sdf/trunc_mm")
    parser.add_argument("--limit",     type=int,   default=0,   help="limit number of cases; 0 = all")
    parser.add_argument("--out-dir",   type=str,   default=OUT, help="directory to write shell .npz files")
    parser.add_argument("--profile-dir", type=str, default=None, help="directory to append profiling logs to")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(args.images_dir, "*.nii.gz")))
    labs = {case_id(p): p for p in glob.glob(os.path.join(args.labels_dir, "*.nii.gz"))}

    if not imgs:
        raise RuntimeError(f"No images found in:\n{args.images_dir}")
    if not labs:
        raise RuntimeError(f"No labels found in:\n{args.labels_dir}")

    missing = [case_id(p) for p in imgs if case_id(p) not in labs]
    if missing:
        print(f"Warning: {len(missing)} images have no corresponding label — skipping", flush=True)

    jobs = [(p, labs[case_id(p)]) for p in imgs if case_id(p) not in missing]

    if args.limit:
        jobs = jobs[:args.limit]

    print(f"{len(jobs)} cases -> {args.out_dir}\ngrid={args.grid}³\nsdf truncation={args.trunc_mm} mm", flush=True)

    prof = Profiler("make_shells", config=vars(args), profile_dir=args.profile_dir)
    prof.config_update({"data": {"cases": len(jobs)}})

    with prof:
        for i, (img_path, lab_path) in enumerate(jobs, start=1):
            print(f"[{i}/{len(jobs)}]", end=" ", flush=True)
            with prof.span("build_shell") as span:
                build_one(img_path, lab_path, args.grid, args.trunc_mm, args.out_dir, verbose=True)
                span["items"] = 1

    print("\nPhase 1 dataset created successfully.")


if __name__ == "__main__":
    main()
