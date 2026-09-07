"""
Runs the full pipeline from scratch: shells -> phase 1 train -> eval -> crops -> phase 2 train -> eval.

Usage:
python pipeline.py                            # full run, default settings
python pipeline.py --limit 3                  # quick test on 3 cases
python pipeline.py --epochs-p1 100 --epochs-p2 100
python pipeline.py --amp                      # mixed precision for phase 2
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

RUNS = "C:/Users/user/Desktop/boundary_first_then_refine/runs"


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")

    t0     = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nPipeline failed at: {label}")
        sys.exit(result.returncode)

    print(f"\n  Done in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit",     type=int,   default=0,          help="limit cases per preprocessing step; 0 = all")
    parser.add_argument("--epochs-p1", type=int,   default=50,         help="Phase 1 training epochs")
    parser.add_argument("--epochs-p2", type=int,   default=50,         help="Phase 2 training epochs")
    parser.add_argument("--loss-p1",   type=str,   default="mse_bce",  help="Phase 1 loss")
    parser.add_argument("--loss-p2",   type=str,   default="dice_bce", help="Phase 2 loss")
    parser.add_argument("--base-ch",   type=int,   default=16,         help="U-Net base channels (shared)")
    parser.add_argument("--amp",       action="store_true",             help="mixed precision for Phase 2")
    args = parser.parse_args()

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root    = os.path.join(RUNS, timestamp)
    shells_dir  = os.path.join(run_root, "shells")
    crops_dir   = os.path.join(run_root, "crops")
    splits_json = os.path.join(run_root, "splits.json")
    p1_dir      = os.path.join(run_root, "phase1")
    p2_dir      = os.path.join(run_root, "phase2")
    p1_ckpt     = os.path.join(p1_dir, "checkpoints", "phase1_best.pth")

    os.makedirs(shells_dir, exist_ok=True)
    os.makedirs(crops_dir,  exist_ok=True)
    os.makedirs(p1_dir,     exist_ok=True)
    os.makedirs(p2_dir,     exist_ok=True)

    print(f"\nRun: {run_root}")

    py    = sys.executable
    limit = ["--limit", str(args.limit)] if args.limit else []
    base  = ["--base-ch", str(args.base_ch)]

    total_start = time.time()

    run_step("Phase 1 — Build shells", [
        py, "make_shells.py",
        "--out-dir", shells_dir,
    ] + limit)

    run_step("Phase 1 — Train", [
        py, "train.py",
        "--epochs",      str(args.epochs_p1),
        "--loss",        args.loss_p1,
        "--run-dir",     p1_dir,
        "--shells-dir",  shells_dir,
        "--splits-json", splits_json,
    ] + base)

    run_step("Phase 1 — Evaluate", [
        py, "evaluate.py",
        "--run-dir",     p1_dir,
        "--split",       "val",
        "--shells-dir",  shells_dir,
        "--splits-json", splits_json,
    ] + base)

    run_step("Phase 2 — Build crops", [
        py, "make_crops.py",
        "--shells-dir", shells_dir,
        "--out-dir",    crops_dir,
        "--checkpoint", p1_ckpt,
    ] + limit + base)

    p2_cmd = [
        py, "train_phase2.py",
        "--epochs",      str(args.epochs_p2),
        "--loss",        args.loss_p2,
        "--run-dir",     p2_dir,
        "--crops-dir",   crops_dir,
        "--splits-json", splits_json,
    ] + base
    if args.amp:
        p2_cmd.append("--amp")

    run_step("Phase 2 — Train", p2_cmd)

    run_step("Phase 2 — Evaluate", [
        py, "evaluate_phase2.py",
        "--run-dir",     p2_dir,
        "--split",       "val",
        "--crops-dir",   crops_dir,
        "--splits-json", splits_json,
    ] + base)

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete — {total / 60:.1f} min")
    print(f"  Results in: {run_root}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()