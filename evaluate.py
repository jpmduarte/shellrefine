"""
Usage:
python evaluate.py
python evaluate.py --split train
python evaluate.py --run-dir runs/20260907_143022/phase1
"""

import argparse
import json
import os

import numpy as np
import torch

from dataset import ShellDataset, get_splits
from metrics import dice_from_sdf, mae
from model import UNet3D
from profiler import Profiler, model_summary

OUT = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, dataset, device, prof, save_dir: str = None) -> list[dict]:
    model.eval()
    results = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(len(dataset)):
            path = dataset.paths[i]

            with prof.span("inference_case") as span:
                x, y, _w = dataset[i]

                x    = x.unsqueeze(0).to(device)
                y    = y.to(device)
                pred = model(x).squeeze(0)
                span["items"] = 1

            if save_dir:
                np.save(
                    os.path.join(save_dir, os.path.basename(path).replace(".npz", "_pred.npy")),
                    pred.cpu().numpy()
                )

            results.append({
                "case":    os.path.basename(path),
                "mae":     mae(pred, y),
                "dice":    dice_from_sdf(pred, y),
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
    parser.add_argument("--base-ch",     type=int,   default=16)
    parser.add_argument("--shells-dir",  type=str,   default=None,
                        help="directory containing shell .npz files")
    parser.add_argument("--splits-json", type=str,   default=None,
                        help="path to splits.json")
    parser.add_argument("--profile-dir", type=str,   default=None,
                        help="directory to append profiling logs to")
    args = parser.parse_args()

    run_dir    = args.run_dir
    checkpoint = args.checkpoint or (
        os.path.join(run_dir, "checkpoints", "phase1_best.pth") if run_dir
        else os.path.join(OUT, "checkpoints", "phase1_best.pth")
    )
    save_dir     = os.path.join(run_dir, f"eval_{args.split}", "preds") if run_dir else None
    summary_path = os.path.join(run_dir, f"eval_{args.split}", "summary.txt") if run_dir else None

    if summary_path:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths, val_paths = get_splits(val_fraction=0.2, shells_dir=args.shells_dir, splits_json=args.splits_json)
    paths   = val_paths if args.split == "val" else train_paths
    dataset = ShellDataset(paths, augment=False)

    model = UNet3D(in_channels=1, base_channels=args.base_ch, out_activation="tanh").to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    print(f"Checkpoint: {checkpoint}")
    print(f"Split: {args.split}  Cases: {len(dataset)}  (interior = sdf < 0)\n")

    prof = Profiler("phase1_eval", config=vars(args), profile_dir=args.profile_dir)
    prof.config_update({"model": model_summary(model), "data": {"cases": len(dataset)}})

    with prof:
        results = evaluate(model, dataset, device, prof=prof, save_dir=save_dir)

    valid = [r for r in results if r["dice"] is not None]

    with open(summary_path, "w") if summary_path else open(os.devnull, "w") as f:
        print_and_write(f"{'Case':<30}  {'MAE':>6}  {'Dice':>6}", f)
        print_and_write("-" * 48, f)
        for r in results:
            dice_str = f"{r['dice']:.4f}" if r["dice"] is not None else "  skip"
            print_and_write(f"  {r['case']:<28}  {r['mae']:.4f}  {dice_str:>6}", f)
        print_and_write("-" * 48, f)

        mae_scores  = [r["mae"]  for r in results]
        dice_scores = [r["dice"] for r in valid]

        print_and_write(f"\n  Mean MAE  : {np.mean(mae_scores):.4f}", f)
        print_and_write(f"  Std  MAE  : {np.std(mae_scores):.4f}", f)

        if dice_scores:
            print_and_write(f"\n  Mean Dice : {np.mean(dice_scores):.4f}", f)
            print_and_write(f"  Std  Dice : {np.std(dice_scores):.4f}", f)
            print_and_write(f"  Min  Dice : {np.min(dice_scores):.4f}", f)
            print_and_write(f"  Max  Dice : {np.max(dice_scores):.4f}", f)

        skipped = len(results) - len(valid)
        if skipped:
            print_and_write(f"\n  Skipped: {skipped}", f)

    if summary_path:
        # Machine-readable twin of summary.txt, for tools.aggregate_results — avoids
        # regex-scraping the fixed-width text table.
        summary_json = {
            "checkpoint": checkpoint,
            "split":      args.split,
            "cases":      results,
            "aggregate": {
                "mae_mean":  float(np.mean(mae_scores)),
                "mae_std":   float(np.std(mae_scores)),
                "dice_mean": float(np.mean(dice_scores)) if dice_scores else None,
                "dice_std":  float(np.std(dice_scores))  if dice_scores else None,
                "dice_min":  float(np.min(dice_scores))  if dice_scores else None,
                "dice_max":  float(np.max(dice_scores))  if dice_scores else None,
                "skipped":   skipped,
            },
        }
        with open(os.path.join(os.path.dirname(summary_path), "summary.json"), "w") as f:
            json.dump(summary_json, f, indent=2)

        print(f"\nSummary saved to: {summary_path}")
    if save_dir:
        print(f"Predictions saved to: {save_dir}")


if __name__ == "__main__":
    main()
