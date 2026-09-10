"""
Runs the full pipeline from scratch: shells -> phase 1 train -> eval -> crops -> phase 2 train -> eval.

Usage:
python pipeline.py                            # full run, default settings
python pipeline.py --limit 3                  # quick test on 3 cases
python pipeline.py --epochs-p1 100 --epochs-p2 100
python pipeline.py --amp                      # mixed precision for phase 2

Phase 1 and phase 2 have very different GPU memory needs, so each can be run as its
own job (e.g. separate sbatch scripts with different --gres):
python pipeline.py --stage phase1 ...                        # shells + train p1 + eval p1
python pipeline.py --stage phase2 --run-dir runs/<timestamp>  # crops + train p2 + eval p2
Stage "phase1" writes the run directory it used to runs/.latest_run so a following
"phase2" job can pick it up automatically without hardcoding the timestamp.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE_DIR, "runs")

# Set once in main(); step wall times land next to the per-stage profiles.
PROFILE_DIR = None


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")

    t0     = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if PROFILE_DIR:
        with open(os.path.join(PROFILE_DIR, "events.jsonl"), "a") as f:
            f.write(json.dumps({
                "stage":      "pipeline",
                "kind":       "step",
                "span":       label,
                "duration_s": round(elapsed, 2),
                "returncode": result.returncode,
            }) + "\n")

    if result.returncode != 0:
        print(f"\nPipeline failed at: {label}")
        sys.exit(result.returncode)

    print(f"\n  Done in {elapsed:.0f}s")


def main():
    global PROFILE_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=str,  default=None,       help="directory containing input .nii.gz images; required unless --shells-dir reuses an already-built one")
    parser.add_argument("--labels-dir", type=str,  default=None,       help="directory containing label .nii.gz files; required unless --shells-dir reuses an already-built one")
    parser.add_argument("--limit",     type=int,   default=0,          help="limit cases per preprocessing step; 0 = all")
    parser.add_argument("--epochs-p1", type=int,   default=50,         help="Phase 1 training epochs")
    parser.add_argument("--epochs-p2", type=int,   default=50,         help="Phase 2 training epochs")
    parser.add_argument("--lr-p1",     type=float, default=1e-3,       help="Phase 1 learning rate")
    parser.add_argument("--lr-p2",     type=float, default=1e-3,       help="Phase 2 learning rate")
    parser.add_argument("--loss-p1",   type=str,   default="l1",       help="Phase 1 loss (signed distance regression)")
    parser.add_argument("--boundary-weight-floor-p1", type=float, default=1.0,
                        help="1.0 (default) = off. Below 1.0, Phase 1's loss adds an L1 "
                             "term weighting the saturated |sdf| region at this floor")
    parser.add_argument("--boundary-weight-thr-p1",   type=float, default=0.7,
                        help="normalised |sdf| threshold above which a voxel is 'saturated'")
    parser.add_argument("--boundary-weight-coef-p1",  type=float, default=1.0,
                        help="strength of the added Phase 1 boundary-weighted term")
    parser.add_argument("--gd-coef-p1", type=float, default=0.0,
                        help="0.0 (default) = off. Strength of an added Phase 1 "
                             "gradient-difference term on top of --loss-p1")
    parser.add_argument("--seed", type=int, default=None,
                        help="None (default) = unseeded. Fixes model init, augmentation/"
                             "patch RNG and DataLoader shuffle order for both phases")
    parser.add_argument("--loss-p2",   type=str,   default="dice_bce", help="Phase 2 loss")
    parser.add_argument("--base-ch",   type=int,   default=16,         help="U-Net base channels (shared)")
    parser.add_argument("--amp",       action="store_true",             help="mixed precision for Phase 2")
    parser.add_argument("--patch-size",       type=int, default=64,     help="Phase 2 patch size along the boundary band")
    parser.add_argument("--patches-per-case", type=int, default=8,      help="Phase 2 patches drawn per case per epoch")
    parser.add_argument("--trunc-mm",         type=float, default=10.0,
                        help="Phase 1 signed distance truncation in mm")
    parser.add_argument("--band-mm",          type=float, default=5.0,
                        help="half-thickness in mm of the band phase 2 refines")
    parser.add_argument("--boundary-weight-floor", type=float, default=1.0,
                        help="1.0 (default) = off. Below 1.0, Phase 2's loss adds a BCE "
                             "term weighting band voxels (from make_crops) at 1.0 and "
                             "everything else at this floor")
    parser.add_argument("--boundary-weight-coef",  type=float, default=1.0,
                        help="strength of the added boundary-weighted term")
    parser.add_argument("--stage",     type=str,   default="all",
                         choices=["phase1", "phase2", "all"],
                         help="phase1 = shells+train+eval (low GPU mem); "
                              "phase2 = crops+train+eval (high GPU mem); "
                              "all = both, in one job")
    parser.add_argument("--run-dir",   type=str,   default=None,
                         help="use this directory instead of an auto-generated timestamp "
                              "one; required for --stage phase2, optional otherwise. If it "
                              "already contains a splits.json, get_splits()/get_crop_splits() "
                              "load it as-is (e.g. a k-fold split pre-written by "
                              "tools.make_fold_splits) instead of deriving a fresh 72/18 split")
    parser.add_argument("--shells-dir", dest="shells_dir_override", type=str, default=None,
                         help="reuse an already-built shells/ directory (shells depend on "
                              "nothing fold/seed/hyperparameter-related) instead of "
                              "rebuilding it — skips the 'Phase 1 — Build shells' step")
    parser.add_argument("--no-save-preds", action="store_true",
                         help="skip writing Phase 2 eval's full-res prediction volumes "
                              "(~500MB/run) — off by default, exact fallback")
    args = parser.parse_args()

    if args.stage == "phase2" and not args.run_dir:
        parser.error("--run-dir is required when --stage phase2")
    # make_crops.py always reads native images/labels directly, regardless of whether
    # shells were reused — only a phase1-only run with a reused shells-dir can skip both.
    needs_raw_data = not (args.stage == "phase1" and args.shells_dir_override)
    if needs_raw_data and (not args.images_dir or not args.labels_dir):
        parser.error("--images-dir/--labels-dir are required (only --stage phase1 with "
                     "--shells-dir reusing an existing directory can omit them)")

    if args.run_dir:
        run_root = args.run_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root  = os.path.join(RUNS, timestamp)

    shells_dir  = args.shells_dir_override or os.path.join(run_root, "shells")
    crops_dir   = os.path.join(run_root, "crops")
    splits_json = os.path.join(run_root, "splits.json")
    p1_dir      = os.path.join(run_root, "phase1")
    p2_dir      = os.path.join(run_root, "phase2")
    p1_ckpt     = os.path.join(p1_dir, "checkpoints", "phase1_best.pth")
    profile_dir = os.path.join(run_root, "profile")

    os.makedirs(shells_dir,  exist_ok=True)
    os.makedirs(p1_dir,      exist_ok=True)
    os.makedirs(profile_dir, exist_ok=True)
    PROFILE_DIR = profile_dir
    if args.stage in ("phase2", "all"):
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(p2_dir,    exist_ok=True)

    print(f"\nRun: {run_root}")

    py      = sys.executable
    limit   = ["--limit", str(args.limit)] if args.limit else []
    base    = ["--base-ch", str(args.base_ch)]
    profile = ["--profile-dir", profile_dir]

    total_start = time.time()

    if args.stage in ("phase1", "all"):
        if args.shells_dir_override:
            print(f"\nReusing shells: {shells_dir} (skipping build — shells don't depend "
                  f"on fold/seed/hyperparameters)")
        else:
            run_step("Phase 1 — Build shells", [
                py, "make_shells.py",
                "--images-dir", args.images_dir,
                "--labels-dir", args.labels_dir,
                "--out-dir",  shells_dir,
                "--trunc-mm", str(args.trunc_mm),
            ] + limit + profile)

        run_step("Phase 1 — Train", [
            py, "train.py",
            "--epochs",      str(args.epochs_p1),
            "--lr",          str(args.lr_p1),
            "--loss",        args.loss_p1,
            "--run-dir",     p1_dir,
            "--shells-dir",  shells_dir,
            "--splits-json", splits_json,
            "--boundary-weight-floor", str(args.boundary_weight_floor_p1),
            "--boundary-weight-thr",   str(args.boundary_weight_thr_p1),
            "--boundary-weight-coef",  str(args.boundary_weight_coef_p1),
            "--gd-coef",               str(args.gd_coef_p1),
        ] + base + profile + (["--seed", str(args.seed)] if args.seed is not None else []))

        run_step("Phase 1 — Evaluate", [
            py, "evaluate.py",
            "--run-dir",     p1_dir,
            "--split",       "val",
            "--shells-dir",  shells_dir,
            "--splits-json", splits_json,
        ] + base + profile)

        with open(os.path.join(RUNS, ".latest_run"), "w") as f:
            f.write(run_root + "\n")

    if args.stage in ("phase2", "all"):
        run_step("Phase 2 — Build crops", [
            py, "make_crops.py",
            "--images-dir", args.images_dir,
            "--labels-dir", args.labels_dir,
            "--shells-dir", shells_dir,
            "--out-dir",    crops_dir,
            "--checkpoint", p1_ckpt,
            "--band-mm",    str(args.band_mm),
        ] + limit + base + profile)

        p2_cmd = [
            py, "train_phase2.py",
            "--epochs",           str(args.epochs_p2),
            "--lr",               str(args.lr_p2),
            "--loss",             args.loss_p2,
            "--run-dir",          p2_dir,
            "--crops-dir",        crops_dir,
            "--splits-json",      splits_json,
            "--patch-size",       str(args.patch_size),
            "--patches-per-case", str(args.patches_per_case),
            "--boundary-weight-floor", str(args.boundary_weight_floor),
            "--boundary-weight-coef",  str(args.boundary_weight_coef),
        ] + base + profile
        if args.amp:
            p2_cmd.append("--amp")
        if args.seed is not None:
            p2_cmd += ["--seed", str(args.seed)]

        run_step("Phase 2 — Train", p2_cmd)

        p2_eval_cmd = [
            py, "evaluate_phase2.py",
            "--run-dir",     p2_dir,
            "--split",       "val",
            "--crops-dir",   crops_dir,
            "--splits-json", splits_json,
            "--patch-size",  str(args.patch_size),
        ] + base + profile
        if args.no_save_preds:
            p2_eval_cmd.append("--no-save-preds")

        run_step("Phase 2 — Evaluate", p2_eval_cmd)

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete — {total / 60:.1f} min")
    print(f"  Results in: {run_root}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()