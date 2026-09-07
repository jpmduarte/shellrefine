"""
Usage:
python make_crops.py
python make_crops.py --limit 3                              # quick test on 3 cases
python make_crops.py --padding 15                           # larger bounding box margin
python make_crops.py --shells-dir runs/my_run/shells --out-dir runs/my_run/crops --checkpoint runs/my_run/phase1/checkpoints/phase1_best.pth
"""

import argparse
import glob
import os
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from make_shells import case_id, normalize
from model import UNet3D

IMAGES = "Z:/Images"
LABELS = "Z:/Labels/Head"
SHELLS = "C:/Users/user/Desktop/boundary_first_then_refine/shells"
CKPT   = "C:/Users/user/Desktop/boundary_first_then_refine/best_model.pth"
OUT    = "C:/Users/user/Desktop/boundary_first_then_refine/crops"


def bbox_from_mask(binary_mask: np.ndarray, padding: int) -> list[tuple[int, int]]:
    coords = np.argwhere(binary_mask)
    if len(coords) == 0:
        return [(0, s) for s in binary_mask.shape]

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    box = []
    for lo, hi, s in zip(mins, maxs, binary_mask.shape):
        box.append((
            max(0, int(lo) - padding),
            min(s, int(hi) + padding + 1)
        ))
    return box


def build_one(
    img_path: str,
    lab_path: str,
    model: torch.nn.Module,
    device: torch.device,
    shells_dir: str,
    out_dir: str,
    padding: int,
    verbose: bool = False,
) -> None:
    t0  = time.time()
    cid = case_id(img_path)

    # Load native volume and label
    img_nii = nib.load(img_path)
    lab_nii = nib.load(lab_path)

    img_native   = np.asanyarray(img_nii.dataobj).astype(np.float32)
    mask_native  = np.asanyarray(lab_nii.dataobj) > 0
    native_shape = img_native.shape
    spacing      = img_nii.header.get_zooms()[:3]

    # Load Phase 1 input from shell and run model
    shell      = np.load(os.path.join(shells_dir, f"{cid}.npz"))
    img_small  = shell["img"].astype(np.float32)
    mask_small = shell["mask"].astype(np.float32)

    x = torch.from_numpy(
        np.stack([img_small, mask_small], axis=0)
    ).unsqueeze(0).to(device)  # (1, 2, 32, 32, 32)

    with torch.no_grad():
        boundary_pred_small = model(x)  # (1, 1, 32, 32, 32)

    # Upsample boundary prediction to native resolution (trilinear)
    boundary_prior = F.interpolate(
        boundary_pred_small,
        size=list(native_shape),
        mode="trilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()  # (D, H, W)

    # Bounding box from upsampled prediction
    bbox   = bbox_from_mask(boundary_prior > 0.1, padding)
    slices = tuple(slice(lo, hi) for lo, hi in bbox)

    # Crop all arrays to bounding box
    img_crop   = normalize(img_native[slices])
    mask_crop  = mask_native[slices].astype(np.uint8)
    prior_crop = boundary_prior[slices].astype(np.float32)

    elapsed = time.time() - t0

    # Save
    np.savez_compressed(
        os.path.join(out_dir, f"{cid}.npz"),

        # Phase 2 input: ch0=image, ch1=boundary prior
        img=img_crop,
        boundary_prior=prior_crop,

        # Ground truth
        mask=mask_crop,

        # Metadata (needed to place prediction back in full volume)
        bbox=np.array(bbox, dtype=np.int32),
        native_shape=np.array(native_shape, dtype=np.int32),
        spacing=np.array(spacing, dtype=np.float32),
    )

    if verbose:
        crop_shape = tuple(hi - lo for lo, hi in bbox)
        print(
            f"  {cid}"
            f"  {elapsed:.1f}s"
            f"  native={native_shape}"
            f"  crop={crop_shape}",
            flush=True
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padding",    type=int, default=10,
                        help="voxel margin added around bounding box")
    parser.add_argument("--base-ch",    type=int, default=16,
                        help="must match Phase 1 model")
    parser.add_argument("--limit",      type=int, default=0)
    parser.add_argument("--shells-dir", type=str, default=SHELLS,
                        help="directory containing Phase 1 shell .npz files")
    parser.add_argument("--out-dir",    type=str, default=OUT,
                        help="directory to write crop .npz files")
    parser.add_argument("--checkpoint", type=str, default=CKPT,
                        help="Phase 1 model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(in_channels=2, base_channels=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Phase 1 model loaded from {args.checkpoint}")

    os.makedirs(args.out_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(IMAGES, "*.nii.gz")))
    labs = {
        case_id(p): p
        for p in glob.glob(os.path.join(LABELS, "*.nii.gz"))
    }

    missing = [case_id(p) for p in imgs if case_id(p) not in labs]
    if missing:
        print(f"Warning: {len(missing)} images without label — skipping")

    jobs = [
        (p, labs[case_id(p)])
        for p in imgs
        if case_id(p) not in missing
    ]

    if args.limit:
        jobs = jobs[:args.limit]

    print(
        f"{len(jobs)} cases -> {args.out_dir}\n"
        f"shells from {args.shells_dir}\n"
        f"padding={args.padding} vox",
        flush=True
    )

    for i, (img_path, lab_path) in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}]", end=" ", flush=True)
        build_one(
            img_path=img_path,
            lab_path=lab_path,
            model=model,
            device=device,
            shells_dir=args.shells_dir,
            out_dir=args.out_dir,
            padding=args.padding,
            verbose=True,
        )

    print("\nPhase 2 crops created successfully.")


if __name__ == "__main__":
    main()
