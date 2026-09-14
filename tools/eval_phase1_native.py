"""
Phase-1 quality measured at native resolution, so grids of different size can be
compared on equal terms.

evaluate.py scores the predicted field on the shell's own grid, which makes a 64
run look different from a 32 run for reasons that have nothing to do with how
well the surface is placed. Here the predicted field is upsampled to the native
volume exactly as make_crops does, the coarse mask is taken as its negative
region, and everything is scored against the full-resolution label.

Reported per run: Dice of the coarse mask, mean surface distance in mm, and the
radius ratio, defined as the cube root of predicted volume over true volume. A
ratio below 1.0 means the surface sits inside the annotation.

Usage:
python -m tools.eval_phase1_native --checkpoint <ckpt> --shells-dir <dir> \
    --splits-json <splits.json>
"""

import argparse
import json
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from metrics import asd, dice
from model import UNet3D

LABELS = "/home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Labels/Head"


def coarse_at_native(model, shell_path: str, device) -> tuple[np.ndarray, tuple]:
    shell = np.load(shell_path)
    native_shape = tuple(int(v) for v in shell["native_shape"])
    spacing = tuple(float(v) for v in shell["spacing"])

    x = torch.from_numpy(shell["img"].astype(np.float32))[None, None].to(device)
    with torch.no_grad():
        sdf_small = model(x)

    sdf_native = F.interpolate(sdf_small, size=list(native_shape),
                               mode="trilinear", align_corners=False)
    return (sdf_native.squeeze().cpu().numpy() < 0), spacing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shells-dir", required=True)
    ap.add_argument("--splits-json", required=True)
    ap.add_argument("--labels-dir", default=LABELS)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, base_channels=args.base_ch, out_activation="tanh").to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    val = json.load(open(args.splits_json))["val"]

    dices, asds, ratios = [], [], []
    for path in val:
        cid = os.path.basename(path)[:-4]
        shell_path = os.path.join(args.shells_dir, f"{cid}.npz")
        pred, spacing = coarse_at_native(model, shell_path, device)

        gt = np.asanyarray(nib.load(os.path.join(args.labels_dir, f"{cid}.nii.gz")).dataobj) > 0

        dices.append(dice(torch.from_numpy(pred.astype(np.float32)),
                          torch.from_numpy(gt.astype(np.float32)), threshold=0.5))
        asds.append(asd(pred, gt, spacing))
        ratios.append((pred.sum() / max(gt.sum(), 1)) ** (1.0 / 3.0))

    name = args.label or os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))
    print(f"{name:>10s}  n={len(val):2d}  "
          f"dice={np.mean(dices):.4f}+-{np.std(dices):.4f}  "
          f"asd={np.mean(asds):.3f}mm+-{np.std(asds):.3f}  "
          f"raio={np.mean(ratios):.4f}+-{np.std(ratios):.4f}")


if __name__ == "__main__":
    main()
