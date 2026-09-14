"""
Static look at one phase-2 crop, written to a PNG so it works over SSH.

visualize_crop.py is interactive and needs a display. This one rebuilds the crop
on the fly from the shells and a phase-1 checkpoint, which is what makes it work
after the per-run crop directories have been cleaned up, and lays several slices
side by side instead of putting them behind a slider.

Rows are what phase 2 actually receives and is scored against: the cropped image
is channel 0, the upsampled distance prior is channel 1, then the band it is
allowed to rewrite, and finally the annotation.

Usage:
python -m tools.show_crop --case 40_I0000003
"""

import argparse
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from make_crops import build_one
from model import UNet3D
from tools.visualize_folds import IMAGES, LABELS, SHELLS

CKPT = "experiment/stageC/f2_plain/fold0/seed42/phase1/checkpoints/phase1_best.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case id without extension")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--shells-dir", default=SHELLS)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--band-mm", type=float, default=5.0)
    ap.add_argument("--slices", type=int, default=6)
    ap.add_argument("--out", default="images/crop_exemplo.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, base_channels=args.base_ch, out_activation="tanh").to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    tmp = tempfile.mkdtemp(prefix="showcrop_")
    build_one(os.path.join(IMAGES, f"{args.case}.nii.gz"),
              os.path.join(LABELS, f"{args.case}.nii.gz"),
              model, device, args.shells_dir, tmp,
              padding=10, band_mm=args.band_mm)

    d = np.load(os.path.join(tmp, f"{args.case}.npz"))
    img, prior = d["img"], d["sdf_prior"]
    band, coarse, gt = d["band"] > 0.5, d["coarse_mask"] > 0.5, d["mask"] > 0.5
    native, bbox = tuple(d["native_shape"]), d["bbox"]

    # Spread the slices over the part of the volume that actually contains head.
    present = np.flatnonzero(gt.reshape(gt.shape[0], -1).any(1))
    ks = np.linspace(present[0], present[-1], args.slices + 2).astype(int)[1:-1]

    rows = [
        ("Imagem\n(canal 0)", lambda k: (img[k], dict(cmap="gray"))),
        ("SDF prior\n(canal 1)", lambda k: (prior[k], dict(cmap="RdBu_r", vmin=-1, vmax=1))),
        ("Banda que a fase 2\npode reescrever", lambda k: (band[k], dict(cmap="magma"))),
        ("Anotação", lambda k: (gt[k], dict(cmap="gray"))),
    ]

    fig, axes = plt.subplots(len(rows), len(ks), figsize=(2.5 * len(ks), 2.7 * len(rows)))

    for r, (label, get) in enumerate(rows):
        for c, k in enumerate(ks):
            data, kw = get(k)
            ax = axes[r, c]
            ax.imshow(data, **kw)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"corte {k}", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)

    crop_shape = img.shape
    frac = 100.0 * np.prod(crop_shape) / np.prod(native)
    fig.suptitle(
        f"{args.case} — crop da fase 2\n"
        f"volume nativo {tuple(int(v) for v in native)}, crop {crop_shape}, "
        f"{frac:.0f}% do volume  |  banda ±{args.band_mm}mm ocupa "
        f"{100.0 * band.mean():.0f}% do crop",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")

    print(f"nativo={tuple(int(v) for v in native)}  crop={crop_shape}  "
          f"({frac:.1f}% do volume)")
    print(f"banda ocupa {100.0 * band.mean():.1f}% do crop, "
          f"coarse={100.0 * coarse.mean():.1f}%, anotacao={100.0 * gt.mean():.1f}%")
    print(f"bbox={bbox.tolist()}")
    print(f"escrito: {args.out}")


if __name__ == "__main__":
    main()
