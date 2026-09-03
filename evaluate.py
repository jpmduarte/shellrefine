"""
Usage:
python evaluate.py                          # evaluates best_model.pth on val set
python evaluate.py --split train            # evaluate on train set instead
python evaluate.py --checkpoint my.pth
"""

import argparse
import os

import numpy as np
import torch

from dataset import ShellDataset, get_splits
from metrics import dice, mae
from model import UNet3D

OUT = "C:/Users/user/Desktop/boundary_first_then_refine"


def evaluate(model, dataset, device, threshold: float = 0.5) -> list[dict]:
    model.eval()
    results = []

    with torch.no_grad():
        for i in range(len(dataset)):
            path = dataset.paths[i]
            x, y = dataset[i]

            x = x.unsqueeze(0).to(device)
            y = y.to(device)

            pred = model(x).squeeze(0)

            dice_score = dice(pred, y, threshold=threshold)
            mae_score  = mae(pred, y)

            results.append({
                "case":    os.path.basename(path),
                "mae":     mae_score,
                "dice":    dice_score,
                "skipped": dice_score is None,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(OUT, "best_model.pth"))
    parser.add_argument("--split",      type=str,   default="val",
                        help="val | train")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="threshold for binarizing boundary map")
    parser.add_argument("--base-ch",    type=int,   default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths, val_paths = get_splits(val_fraction=0.2)
    paths = val_paths if args.split == "val" else train_paths

    dataset = ShellDataset(paths, augment=False)

    model = UNet3D(in_channels=2, base_channels=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded: {args.checkpoint}")
    print(f"Split: {args.split}  Cases: {len(dataset)}  Threshold: {args.threshold}\n")

    results = evaluate(model, dataset, device, threshold=args.threshold)

    valid   = [r for r in results if not r["skipped"]]
    skipped = [r for r in results if r["skipped"]]

    print(f"{'Case':<30}  {'MAE':>6}  {'Dice':>6}")
    print("-" * 48)
    for r in results:
        dice_str = f"{r['dice']:.4f}" if not r["skipped"] else "  skip"
        print(f"  {r['case']:<28}  {r['mae']:.4f}  {dice_str:>6}")
    print("-" * 48)

    mae_scores  = [r["mae"]  for r in results]
    dice_scores = [r["dice"] for r in valid]

    print(f"\n  Mean MAE  : {np.mean(mae_scores):.4f}  (primary — lower is better)")
    print(f"  Std  MAE  : {np.std(mae_scores):.4f}")

    if dice_scores:
        print(f"\n  Mean Dice : {np.mean(dice_scores):.4f}  (secondary)")
        print(f"  Std  Dice : {np.std(dice_scores):.4f}")
        print(f"  Min  Dice : {np.min(dice_scores):.4f}")
        print(f"  Max  Dice : {np.max(dice_scores):.4f}")

    if skipped:
        print(f"\n  Skipped (empty boundary target): {len(skipped)}")


if __name__ == "__main__":
    main()
