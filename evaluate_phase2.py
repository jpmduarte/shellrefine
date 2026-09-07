"""
Usage:
python evaluate_phase2.py
python evaluate_phase2.py --split train
python evaluate_phase2.py --run-dir runs/20260907_143022/phase2
"""

import argparse
import os

import numpy as np
import torch

from dataset_phase2 import CropDataset, get_crop_splits
from metrics import dice
from model import UNet3D

OUT = "C:/Users/user/Desktop/boundary_first_then_refine"


def evaluate(model, dataset, device, threshold: float, save_dir: str = None) -> list[dict]:
    model.eval()
    results = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(len(dataset)):
            path = dataset.paths[i]
            x, y = dataset[i]

            x    = x.unsqueeze(0).to(device)
            y    = y.to(device)
            pred = model(x).squeeze(0)

            if save_dir:
                np.save(
                    os.path.join(save_dir, os.path.basename(path).replace(".npz", "_pred.npy")),
                    pred.cpu().numpy()
                )

            results.append({
                "case":  os.path.basename(path),
                "dice":  dice(pred, y, threshold=threshold),
            })

    return results


def print_and_write(line: str, f):
    print(line)
    f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",     type=str,   default=None,
                        help="run folder — derives checkpoint and saves all outputs there")
    parser.add_argument("--checkpoint",  type=str,   default=None)
    parser.add_argument("--split",       type=str,   default="val")
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--base-ch",     type=int,   default=16)
    parser.add_argument("--crops-dir",   type=str,   default=None,
                        help="directory containing crop .npz files")
    parser.add_argument("--splits-json", type=str,   default=None,
                        help="path to splits.json")
    args = parser.parse_args()

    run_dir    = args.run_dir
    checkpoint = args.checkpoint or (
        os.path.join(run_dir, "checkpoints", "phase2_best.pth") if run_dir
        else os.path.join(OUT, "checkpoints", "phase2_best.pth")
    )
    save_dir     = os.path.join(run_dir, f"eval_{args.split}", "preds") if run_dir else None
    summary_path = os.path.join(run_dir, f"eval_{args.split}", "summary.txt") if run_dir else None

    if summary_path:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths, val_paths = get_crop_splits(crops_dir=args.crops_dir, splits_json=args.splits_json)
    paths   = val_paths if args.split == "val" else train_paths
    dataset = CropDataset(paths, augment=False)

    model = UNet3D(in_channels=2, base_channels=args.base_ch).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    print(f"Checkpoint: {checkpoint}")
    print(f"Split: {args.split}  Cases: {len(dataset)}  Threshold: {args.threshold}\n")

    results = evaluate(model, dataset, device, threshold=args.threshold, save_dir=save_dir)

    valid  = [r for r in results if r["dice"] is not None]
    scores = [r["dice"] for r in valid]

    with open(summary_path, "w") if summary_path else open(os.devnull, "w") as f:
        print_and_write(f"{'Case':<30}  {'Dice':>6}", f)
        print_and_write("-" * 40, f)
        for r in results:
            val = f"{r['dice']:.4f}" if r["dice"] is not None else "  skip"
            print_and_write(f"  {r['case']:<28}  {val:>6}", f)
        print_and_write("-" * 40, f)

        if scores:
            print_and_write(f"\n  Mean Dice : {np.mean(scores):.4f}", f)
            print_and_write(f"  Std       : {np.std(scores):.4f}", f)
            print_and_write(f"  Min       : {np.min(scores):.4f}", f)
            print_and_write(f"  Max       : {np.max(scores):.4f}", f)

        skipped = len(results) - len(valid)
        if skipped:
            print_and_write(f"\n  Skipped (empty mask): {skipped}", f)

    if summary_path:
        print(f"\nSummary saved to: {summary_path}")
    if save_dir:
        print(f"Predictions saved to: {save_dir}")


if __name__ == "__main__":
    main()
