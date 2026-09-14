"""
Architecture figure for the paper and the slides: phase 1 on the top band, phase 2 on
the bottom band, and the transition between them spelled out in the middle.

Every thumbnail is real data for one case, produced by the actual pipeline rather than
drawn by hand, so the figure cannot drift away from what the code does.

Usage:
python -m tools.figure_architecture --case 40_I0000003
"""

import argparse
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from evaluate_phase2 import predict_case
from make_crops import build_one
from model import UNet3D
from tools.visualize_folds import IMAGES, LABELS, SHELLS

RUN = "experiment/stageC/f2_plain/fold0/seed42"
P1_COLOR = "#dce7f5"
P2_COLOR = "#f7e3d3"
BOX_COLOR = "#ffffff"


def widest(mask):
    return int(mask.reshape(mask.shape[0], -1).sum(1).argmax())


def panel(fig, rect, data, title, sub="", **kw):
    ax = fig.add_axes(rect)
    ax.imshow(data, **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.35")
    ax.set_title(title, fontsize=9.5, pad=4)
    if sub:
        ax.set_xlabel(sub, fontsize=8, labelpad=3)
    return ax


def opbox(fig, x, y, w, h, text, color=BOX_COLOR):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor="0.3", linewidth=1.2,
                                transform=ax.transAxes))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9,
            transform=ax.transAxes)
    return ax


def arrow(fig, x0, y0, x1, y1, label="", fontsize=8.5):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1),
                                       transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=16,
                                       linewidth=1.4, color="0.25"))
    if label:
        fig.text((x0 + x1) / 2, max(y0, y1) + 0.014, label, ha="center",
                 va="bottom", fontsize=fontsize, color="0.2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="40_I0000003")
    ap.add_argument("--out", default="images/arquitetura.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p1 = UNet3D(in_channels=1, base_channels=16, out_activation="tanh").to(device)
    p1.load_state_dict(torch.load(f"{RUN}/phase1/checkpoints/phase1_best.pth",
                                  map_location=device))
    p1.eval()
    p2 = UNet3D(in_channels=2, base_channels=16).to(device)
    p2.load_state_dict(torch.load(f"{RUN}/phase2/checkpoints/phase2_best.pth",
                                  map_location=device))
    p2.eval()

    shell = np.load(os.path.join(SHELLS, f"{args.case}.npz"))
    img32, sdf32, mask32 = shell["img"], shell["sdf"], shell["mask"]
    native_shape = tuple(int(v) for v in shell["native_shape"])

    with torch.no_grad():
        x = torch.from_numpy(img32.astype(np.float32))[None, None].to(device)
        sdf32_pred = p1(x)
        sdf_native = F.interpolate(sdf32_pred, size=list(native_shape),
                                   mode="trilinear", align_corners=False
                                   ).squeeze().cpu().numpy()
    sdf32_pred = sdf32_pred.squeeze().cpu().numpy()

    native_img = np.asanyarray(nib.load(
        os.path.join(IMAGES, f"{args.case}.nii.gz")).dataobj).astype(np.float32)

    tmp = tempfile.mkdtemp(prefix="arch_")
    build_one(os.path.join(IMAGES, f"{args.case}.nii.gz"),
              os.path.join(LABELS, f"{args.case}.nii.gz"),
              p1, device, SHELLS, tmp, padding=10, band_mm=5.0)
    crop_path = os.path.join(tmp, f"{args.case}.npz")
    crop = np.load(crop_path)
    with torch.no_grad():
        refined, coarse, band, gtmask, _, ntiles = predict_case(
            p2, crop_path, device, 64, 32, 8)

    k32 = widest(mask32)
    kna = widest(gtmask > 0.5)
    knat = widest(sdf_native < 0)

    fig = plt.figure(figsize=(17.5, 9.2))
    fig.patch.set_facecolor("white")

    # bands
    for y0, h, color, name in [(0.545, 0.375, P1_COLOR, "FASE 1  ·  campo de distância a baixa resolução"),
                               (0.045, 0.375, P2_COLOR, "FASE 2  ·  refinamento na banda, resolução nativa")]:
        ax = fig.add_axes([0.012, y0, 0.976, h]); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor=color, edgecolor="0.6", linewidth=1.0,
                                    transform=ax.transAxes, zorder=-1))
        ax.text(0.006, 0.955, name, fontsize=11.5, weight="bold", color="0.15",
                ha="left", va="top", transform=ax.transAxes)

    W, H, Y1, Y2 = 0.118, 0.235, 0.615, 0.115

    panel(fig, [0.035, Y1, W, H], native_img[knat], "Volume nativo",
          f"{native_shape}, 0.4mm", cmap="gray")
    arrow(fig, 0.162, Y1 + H / 2, 0.205, Y1 + H / 2, "reduzir\npara 32³")
    panel(fig, [0.245, Y1, W, H], img32[k32], "Shell 32³", "input da rede", cmap="gray")
    arrow(fig, 0.372, Y1 + H / 2, 0.425, Y1 + H / 2)
    opbox(fig, 0.428, Y1 + 0.055, 0.108, 0.125,
          "U-Net 3D\n1.40M par.\nsaída tanh")
    arrow(fig, 0.545, Y1 + H / 2, 0.598, Y1 + H / 2)
    panel(fig, [0.617, Y1, W, H], sdf32_pred[k32], "SDF previsto 32³",
          "negativo = dentro", cmap="RdBu_r", vmin=-1, vmax=1)
    panel(fig, [0.800, Y1, W, H], sdf32[k32], "SDF alvo (treino)",
          "truncado a ±10mm", cmap="RdBu_r", vmin=-1, vmax=1)
    fig.text(0.859, Y1 - 0.045, "supervisão: l1 + gradient difference",
             ha="center", fontsize=8.5, style="italic", color="0.25")

    # transition strip
    fig.text(0.5, 0.512,
             "upsample trilinear ×6 para resolução nativa   →   sinal do campo dá a máscara grosseira   →   "
             "|sdf| < 5mm dá a banda   →   bounding box da banda + 10 voxels dá o crop",
             ha="center", va="center", fontsize=10.5, color="0.1",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.5"))
    arrow(fig, 0.676, Y1 - 0.012, 0.676, 0.535)
    arrow(fig, 0.30, 0.487, 0.30, Y2 + H + 0.012)

    panel(fig, [0.035, Y2, W, H], crop["img"][kna], "Crop, canal 0",
          f"imagem, {crop['img'].shape}", cmap="gray")
    panel(fig, [0.175, Y2, W, H], crop["sdf_prior"][kna], "Crop, canal 1",
          "SDF prior da fase 1", cmap="RdBu_r", vmin=-1, vmax=1)
    arrow(fig, 0.302, Y2 + H / 2, 0.352, Y2 + H / 2)
    panel(fig, [0.368, Y2, W, H], band[kna], "Banda",
          f"seleciona os tiles: {ntiles} de 64³,\n{100 * band.mean():.0f}% do crop", cmap="magma")
    arrow(fig, 0.495, Y2 + H / 2, 0.545, Y2 + H / 2, "os dois canais,\nsó nesses tiles")
    opbox(fig, 0.548, Y2 + 0.055, 0.108, 0.125,
          "U-Net 3D\n1.40M par.\n2 canais → prob.")
    arrow(fig, 0.665, Y2 + H / 2, 0.715, Y2 + H / 2, "fundir:\nfora da banda\nfica a fase 1")
    panel(fig, [0.735, Y2, W, H], (refined >= 0.5)[kna].astype(float),
          "Máscara final", "Dice médio 0.9456 ± 0.0041\n(5 folds × 2 seeds)", cmap="gray")

    fig.text(0.90, Y2 + H / 2,
             "2.80M parâmetros\n10.7MB\n3.5GB VRAM de treino",
             ha="center", va="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.5"))

    fig.suptitle("ShellRefine: SDF grosseiro, refinamento numa banda estreita",
                 fontsize=15, weight="bold", y=0.975)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"escrito: {args.out}  (tiles={ntiles}, banda={100 * band.mean():.1f}%)")


if __name__ == "__main__":
    main()
