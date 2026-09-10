"""
Per-fold qualitative figure for one Stage C model: two validation cases per fold
(the best and the worst by refined Dice), each shown across the whole chain --
phase 1 input, ground-truth SDF, predicted SDF, native crop, upsampled prior and
the final refined mask against ground truth.

The crops are rebuilt on the fly from the shells plus the fold's own phase-1
checkpoint, so this works after the crop directories have been cleaned up.

Usage:
python visualize_folds.py --model f2_plain --seed 42 --out docs/folds_f2_plain.png
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

from evaluate_phase2 import predict_case
from make_crops import build_one
from model import UNet3D

IMAGES = "/home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Images"
LABELS = "/home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Labels/Head"
SHELLS = "runs/20260907_163859/shells"


def widest_slice(mask: np.ndarray) -> int:
    return int(mask.reshape(mask.shape[0], -1).sum(1).argmax())


def agreement_rgb(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Red = ground truth only, green = prediction only, yellow = both."""
    rgb = np.zeros(gt.shape + (3,), dtype=np.float32)
    rgb[..., 0] = gt.astype(np.float32)
    rgb[..., 1] = pred.astype(np.float32)
    return rgb


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    return float(2.0 * inter / (a.sum() + b.sum() + 1e-8))


def pick_cases(run_dir: str) -> list[tuple[str, float, str]]:
    summary = json.load(open(os.path.join(run_dir, "phase2/eval_val/summary.json")))
    cases = sorted(summary["cases"], key=lambda c: c["refined"])
    worst, best = cases[0], cases[-1]
    return [(worst["case"], worst["refined"], "pior"),
            (best["case"],  best["refined"],  "melhor")]


def load_models(run_dir: str, base_ch: int, device):
    p1 = UNet3D(in_channels=1, base_channels=base_ch, out_activation="tanh").to(device)
    p1.load_state_dict(torch.load(
        os.path.join(run_dir, "phase1/checkpoints/phase1_best.pth"), map_location=device))
    p1.eval()

    p2 = UNet3D(in_channels=2, base_channels=base_ch).to(device)
    p2.load_state_dict(torch.load(
        os.path.join(run_dir, "phase2/checkpoints/phase2_best.pth"), map_location=device))
    p2.eval()
    return p1, p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="f2_plain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--tile-batch", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="docs/folds_qualitativo.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stride = args.patch // 2

    rows = []
    tmp = tempfile.mkdtemp(prefix="vizcrops_")

    for fold in range(args.folds):
        run_dir = f"experiment/stageC/{args.model}/fold{fold}/seed{args.seed}"
        p1, p2 = load_models(run_dir, args.base_ch, device)

        for case_npz, refined_dice, label in pick_cases(run_dir):
            cid = case_npz[:-4]
            print(f"fold{fold} {label:7s} {cid[:34]:34s} dice={refined_dice:.4f}", flush=True)

            shell = np.load(os.path.join(SHELLS, case_npz))
            img32, sdf32, mask32 = shell["img"], shell["sdf"], shell["mask"]

            with torch.no_grad():
                x = torch.from_numpy(img32.astype(np.float32))[None, None].to(device)
                sdf32_pred = p1(x).squeeze().cpu().numpy()

            build_one(os.path.join(IMAGES, f"{cid}.nii.gz"),
                      os.path.join(LABELS, f"{cid}.nii.gz"),
                      p1, device, SHELLS, tmp, padding=10, band_mm=5.0)

            crop_path = os.path.join(tmp, f"{cid}.npz")
            with torch.no_grad():
                refined, coarse, band, mask, _, _ = predict_case(
                    p2, crop_path, device, args.patch, stride, args.tile_batch)

            crop = np.load(crop_path)
            rows.append(dict(fold=fold, cid=cid, label=label, dice=refined_dice,
                             img32=img32, sdf32=sdf32, sdf32_pred=sdf32_pred, mask32=mask32,
                             crop_img=crop["img"], crop_prior=crop["sdf_prior"],
                             refined=refined >= args.threshold, gt=mask > 0.5))
            os.remove(crop_path)

    cols = ["1. Shell 32³\n(input fase 1)",
            "2. SDF alvo (GT)\n32³, negativo=dentro",
            "3. SDF previsto\n(output fase 1)",
            "4. Crop nativo\n(input fase 2, ch0)",
            "5. SDF prior upsampled\n(input fase 2, ch1)",
            "6. Refinado vs GT\n(final)"]

    fig, axes = plt.subplots(len(rows), 6, figsize=(19, 3.1 * len(rows)))

    for r, d in enumerate(rows):
        k32 = widest_slice(d["mask32"])
        kna = widest_slice(d["gt"])

        panels = [
            (d["img32"][k32],        dict(cmap="gray")),
            (d["sdf32"][k32],        dict(cmap="RdBu_r", vmin=-1, vmax=1)),
            (d["sdf32_pred"][k32],   dict(cmap="RdBu_r", vmin=-1, vmax=1)),
            (d["crop_img"][kna],     dict(cmap="gray")),
            (d["crop_prior"][kna],   dict(cmap="RdBu_r", vmin=-1, vmax=1)),
            (agreement_rgb(d["gt"][kna], d["refined"][kna]), dict()),
        ]

        for c, (data, kw) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(data, **kw)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)

        slice_dice = dice(d["gt"][kna], d["refined"][kna])
        axes[r, 5].set_xlabel(f"Dice volume {d['dice']:.4f}  |  corte {slice_dice:.4f}",
                              fontsize=9)
        axes[r, 0].set_ylabel(f"fold{d['fold']} — {d['label']}\n{d['cid'][:22]}",
                              fontsize=9)

    fig.suptitle(
        f"{args.model} (seed {args.seed}) — pior e melhor caso de validação por fold\n"
        "vermelho = GT só, verde = previsão só, amarelo = acordo",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nescrito: {args.out}")


if __name__ == "__main__":
    main()
