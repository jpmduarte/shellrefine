"""
Where does the predicted surface sit relative to the annotation and to the
echogenic skull rim?

For each case, rays are cast outwards from the centroid of the ground truth on
its widest slice. Along every ray we record the image intensity, the radius at
which the ground truth ends and the radius at which the prediction ends. Radii
are then expressed as a fraction of that ray's ground-truth radius, so all rays
can be averaged into one profile with the annotation pinned at 1.0.

Reading the result: if the mean predicted radius sits above 1.0 the model
over-segments, and the position of the bright rim peak relative to 1.0 says
whether the annotation follows the inner edge, the crest or the outer edge of
the skull.

Usage:
python -m tools.diagnose_bias --model f2_plain --seed 42 --out images/bias_f2_plain.png
"""

import argparse
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from evaluate_phase2 import predict_case
from make_crops import build_one
from model import UNet3D
from tools.visualize_folds import IMAGES, LABELS, SHELLS, load_models, widest_slice

N_ANGLES = 360
N_SAMPLES = 400
MAX_RADIUS_FACTOR = 1.6


def radial_profiles(img2d, gt2d, pred2d):
    """Mean intensity, and per-ray GT/prediction radii, as a fraction of GT radius."""
    cy, cx = ndimage.center_of_mass(gt2d)
    max_r = MAX_RADIUS_FACTOR * float(np.sqrt(gt2d.sum() / np.pi)) * 1.8

    angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    radii = np.linspace(0, max_r, N_SAMPLES)

    ys = cy + radii[None, :] * np.sin(angles)[:, None]
    xs = cx + radii[None, :] * np.cos(angles)[:, None]
    coords = np.stack([ys.ravel(), xs.ravel()])

    def sample(arr, order):
        return ndimage.map_coordinates(arr.astype(np.float32), coords, order=order,
                                       mode="constant", cval=0.0).reshape(N_ANGLES, N_SAMPLES)

    img_s = sample(img2d, 1)
    gt_s = sample(gt2d, 0) > 0.5
    pred_s = sample(pred2d, 0) > 0.5

    def last_true_radius(hits):
        out = np.full(N_ANGLES, np.nan)
        for i, row in enumerate(hits):
            idx = np.flatnonzero(row)
            if idx.size:
                out[i] = radii[idx[-1]]
        return out

    r_gt = last_true_radius(gt_s)
    r_pred = last_true_radius(pred_s)

    ok = np.isfinite(r_gt) & (r_gt > 0)
    if ok.sum() == 0:
        return None

    # Resample every ray onto a common axis normalised by its own GT radius.
    grid = np.linspace(0, MAX_RADIUS_FACTOR, 200)
    prof = np.full((ok.sum(), grid.size), np.nan)
    for j, i in enumerate(np.flatnonzero(ok)):
        prof[j] = np.interp(grid * r_gt[i], radii, img_s[i])

    return dict(grid=grid,
                intensity=np.nanmean(prof, axis=0),
                ratio_pred=np.nanmean(r_pred[ok] / r_gt[ok]),
                ratio_pred_all=r_pred[ok] / r_gt[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="f2_plain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--tile-batch", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="images/bias_f2_plain.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stride = args.patch // 2
    tmp = tempfile.mkdtemp(prefix="biascrops_")

    results = []
    for fold in range(args.folds):
        run_dir = f"experiment/stageC/{args.model}/fold{fold}/seed{args.seed}"
        p1, p2 = load_models(run_dir, args.base_ch, device)
        cases = json.load(open(os.path.join(run_dir, "phase2/eval_val/summary.json")))["cases"]

        for c in cases:
            cid = c["case"][:-4]
            build_one(os.path.join(IMAGES, f"{cid}.nii.gz"),
                      os.path.join(LABELS, f"{cid}.nii.gz"),
                      p1, device, SHELLS, tmp, padding=10, band_mm=5.0)
            path = os.path.join(tmp, f"{cid}.npz")
            with torch.no_grad():
                refined, coarse, _, mask, _, _ = predict_case(
                    p2, path, device, args.patch, stride, args.tile_batch)
            crop = np.load(path)

            gt = mask > 0.5
            k = widest_slice(gt)
            r = radial_profiles(crop["img"][k], gt[k], (refined >= args.threshold)[k])
            rc = radial_profiles(crop["img"][k], gt[k], coarse[k] > 0.5)
            if r and rc:
                r["ratio_coarse"] = rc["ratio_pred"]
                r.update(fold=fold, cid=cid, dice=c["refined"])
                results.append(r)
            os.remove(path)
            print(f"fold{fold} {cid[:30]:30s} pred/GT={r['ratio_pred']:.3f} "
                  f"coarse/GT={rc['ratio_pred']:.3f}", flush=True)

    ratios = np.array([r["ratio_pred"] for r in results])
    ratios_c = np.array([r["ratio_coarse"] for r in results])
    grid = results[0]["grid"]
    mean_int = np.nanmean(np.stack([r["intensity"] for r in results]), axis=0)
    mean_int = (mean_int - mean_int.min()) / (mean_int.max() - mean_int.min() + 1e-8)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(grid, mean_int, color="0.2", lw=2, label="intensidade média (normalizada)")
    ax[0].axvline(1.0, color="red", lw=2, label="fronteira do GT (por definição)")
    ax[0].axvline(ratios.mean(), color="green", lw=2,
                  label=f"fronteira prevista (média {ratios.mean():.3f})")
    ax[0].axvline(ratios_c.mean(), color="orange", lw=1.5, ls="--",
                  label=f"fronteira coarse / fase 1 ({ratios_c.mean():.3f})")
    peak = grid[int(np.argmax(mean_int))]
    ax[0].axvline(peak, color="blue", lw=1.5, ls=":",
                  label=f"crista do anel ecogénico ({peak:.3f})")
    ax[0].set_xlabel("raio, em fracção do raio do GT")
    ax[0].set_ylabel("intensidade média")
    ax[0].set_title("Onde assenta cada fronteira em relação ao crânio")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].hist(ratios, bins=30, color="green", alpha=0.7, label="refinado")
    ax[1].hist(ratios_c, bins=30, color="orange", alpha=0.5, label="coarse (fase 1)")
    ax[1].axvline(1.0, color="red", lw=2, label="sem viés")
    over = (ratios > 1.0).mean() * 100
    ax[1].set_xlabel("raio previsto / raio do GT")
    ax[1].set_ylabel(f"casos (n={len(ratios)})")
    ax[1].set_title(f"Distribuição do viés — {over:.0f}% dos casos sobre-segmentam")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.suptitle(f"{args.model} (seed {args.seed}) — viés radial da fronteira, "
                 f"{len(ratios)} casos de validação", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")

    print(f"\nrefinado : media {ratios.mean():.4f}  mediana {np.median(ratios):.4f}  "
          f"sobre-segmenta em {over:.1f}% dos casos")
    print(f"coarse   : media {ratios_c.mean():.4f}  mediana {np.median(ratios_c):.4f}")
    print(f"crista do anel ecogenico em {peak:.3f}")
    print(f"escrito: {args.out}")


if __name__ == "__main__":
    main()
