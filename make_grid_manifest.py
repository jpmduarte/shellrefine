"""
Writes one manifest.csv per Stage-A sub-stage (or Stage C), read by submit_grid.sh and
run_grid_task.sh. Each sub-stage after A1 takes the previous sub-stage's winner as CLI
overrides — read aggregate_results.py's table, then call the next sub-stage by hand. This
is deliberate (see plan): stages are cheap enough to eyeball between, not worth auto-
chaining.

Usage, in order:
  python make_grid_manifest.py a1    --out manifests/a1.csv    --shells-dir ... --splits-json ...
  python make_grid_manifest.py a1b   --out manifests/a1b.csv   --loss-p1 l1 --lr-p1 5e-4 ...
  python make_grid_manifest.py a1c   --out manifests/a1c.csv   --loss-p1 l1 --lr-p1 5e-4 ...
  python make_grid_manifest.py a1d   --out manifests/a1d.csv   --loss-p1 l1 --lr-p1 5e-4 \
                                        --bw-floor-p1 0.2 --bw-coef-p1 1.0 --gd-coef-p1 0.1 ...
  python make_grid_manifest.py crops-prep --out manifests/crops_prep.csv \
                                        --p1-checkpoint runs_or_experiment/.../phase1_best.pth ...
  python make_grid_manifest.py a2    --out manifests/a2.csv    --base-run experiment/stageA/crops_prep ...
  python make_grid_manifest.py a2b   --out manifests/a2b.csv   --base-run ... --loss-p2 dice_bce --lr-p2 3e-4
  python make_grid_manifest.py a3    --out manifests/a3.csv    --base-run ... --combos combos.json
  python make_grid_manifest.py stagec --out manifests/stagec.csv --finalists finalists.json \
                                        --images-dir ... --labels-dir ... --shells-dir ... \
                                        --folds-json folds/folds.json --seeds 42,43,44
"""

import argparse
import csv
import itertools
import json
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT  = os.path.join(BASE_DIR, "experiment")

FIELDS = [
    "run_id", "kind", "run_dir", "base_run", "images_dir", "labels_dir",
    "shells_dir", "splits_json", "p1_checkpoint", "band_mm",
    "epochs_p1", "lr_p1", "loss_p1", "bw_floor_p1", "bw_thr_p1", "bw_coef_p1", "gd_coef_p1",
    "epochs_p2", "lr_p2", "loss_p2", "bw_floor_p2", "bw_coef_p2",
    "seed",
]

DEFAULTS = {
    "epochs_p1": 250, "bw_floor_p1": 1.0, "bw_thr_p1": 0.7, "bw_coef_p1": 1.0, "gd_coef_p1": 0.0,
    "epochs_p2": 100, "bw_floor_p2": 1.0, "bw_coef_p2": 1.0,
    "band_mm": 5.0,
}


def write_manifest(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            full = dict(DEFAULTS)
            full.update(row)
            writer.writerow({k: full.get(k, "") for k in FIELDS})
    print(f"{len(rows)} rows -> {out_path}")


def row_dir(sub_stage: str, run_id: str) -> str:
    return os.path.join(EXPERIMENT, "stageA", sub_stage, run_id)


# ---------------------------------------------------------------- Stage A: phase 1

def cmd_a1(args):
    rows = []
    for loss, lr in itertools.product(["l1", "mse"], [1e-3, 5e-4, 2e-4]):
        rid = f"a1_{loss}_lr{lr}"
        rows.append({
            "run_id": rid, "kind": "phase1_only", "run_dir": row_dir("a1", rid),
            "shells_dir": args.shells_dir, "splits_json": args.splits_json,
            "loss_p1": loss, "lr_p1": lr,
        })
    write_manifest(rows, args.out)


def cmd_a1b(args):
    rows = []
    for floor, coef in itertools.product([0.1, 0.2], [0.5, 1.0]):
        rid = f"a1b_f{floor}_c{coef}"
        rows.append({
            "run_id": rid, "kind": "phase1_only", "run_dir": row_dir("a1b", rid),
            "shells_dir": args.shells_dir, "splits_json": args.splits_json,
            "loss_p1": args.loss_p1, "lr_p1": args.lr_p1,
            "bw_floor_p1": floor, "bw_coef_p1": coef,
        })
    write_manifest(rows, args.out)


def cmd_a1c(args):
    rows = []
    for gd in [0.05, 0.1, 0.2]:
        rid = f"a1c_gd{gd}"
        rows.append({
            "run_id": rid, "kind": "phase1_only", "run_dir": row_dir("a1c", rid),
            "shells_dir": args.shells_dir, "splits_json": args.splits_json,
            "loss_p1": args.loss_p1, "lr_p1": args.lr_p1, "gd_coef_p1": gd,
        })
    write_manifest(rows, args.out)


def cmd_a1d(args):
    rid = "a1d_combined"
    rows = [{
        "run_id": rid, "kind": "phase1_only", "run_dir": row_dir("a1d", rid),
        "shells_dir": args.shells_dir, "splits_json": args.splits_json,
        "loss_p1": args.loss_p1, "lr_p1": args.lr_p1,
        "bw_floor_p1": args.bw_floor_p1, "bw_coef_p1": args.bw_coef_p1,
        "gd_coef_p1": args.gd_coef_p1,
    }]
    write_manifest(rows, args.out)


# --------------------------------------------------------- crops for the phase-1 winner

def cmd_crops_prep(args):
    rid = "crops_prep"
    rows = [{
        "run_id": rid, "kind": "build_crops", "run_dir": row_dir("crops_prep", rid),
        "images_dir": args.images_dir, "labels_dir": args.labels_dir,
        "shells_dir": args.shells_dir, "splits_json": args.splits_json,
        "p1_checkpoint": args.p1_checkpoint, "band_mm": args.band_mm,
    }]
    write_manifest(rows, args.out)


# ---------------------------------------------------------------- Stage A: phase 2

def cmd_a2(args):
    rows = []
    losses = ["mse", "l1", "bce", "mse_bce", "dice", "dice_bce"]
    for loss, lr in itertools.product(losses, [1e-3, 5e-4, 3e-4]):
        rid = f"a2_{loss}_lr{lr}"
        rows.append({
            "run_id": rid, "kind": "phase2_only", "run_dir": row_dir("a2", rid),
            "base_run": args.base_run, "loss_p2": loss, "lr_p2": lr,
        })
    write_manifest(rows, args.out)


def cmd_a2b(args):
    rows = []
    for floor, coef in itertools.product([0.1, 0.2, 0.4], [0.5, 1.0]):
        rid = f"a2b_f{floor}_c{coef}"
        rows.append({
            "run_id": rid, "kind": "phase2_only", "run_dir": row_dir("a2b", rid),
            "base_run": args.base_run, "loss_p2": args.loss_p2, "lr_p2": args.lr_p2,
            "bw_floor_p2": floor, "bw_coef_p2": coef,
        })
    write_manifest(rows, args.out)


def cmd_a3(args):
    # --combos: JSON list of {"loss_p1","lr_p1","bw_floor_p1","bw_coef_p1","gd_coef_p1",
    #                         "loss_p2","lr_p2","bw_floor_p2","bw_coef_p2"} dicts —
    # the 3 not-yet-covered top-2-phase1 x top-2-phase2 combinations. Phase-1 differs
    # per combo here, so unlike a2/a2b this is full_pipeline, not phase2_only.
    combos = json.load(open(args.combos))
    rows = []
    for i, combo in enumerate(combos):
        rid = f"a3_{i}"
        rows.append({
            "run_id": rid, "kind": "full_pipeline", "run_dir": row_dir("a3", rid),
            "images_dir": args.images_dir, "labels_dir": args.labels_dir,
            "shells_dir": args.shells_dir, "splits_json": args.splits_json,
            **combo,
        })
    write_manifest(rows, args.out)


# --------------------------------------------------------------------------- Stage C

def cmd_stagec(args):
    # --finalists: JSON list of up to 3 {"name", loss/lr/bw/gd... for both phases} dicts
    # (the Stage-B picks). Each gets retrained from scratch per (fold, seed) -- required
    # for valid k-fold, since phase 1 is one of the swept axes too.
    finalists = json.load(open(args.finalists))
    folds     = json.load(open(args.folds_json))["folds"]
    seeds     = [int(s) for s in args.seeds.split(",")]

    rows = []
    for finalist in finalists:
        name = finalist["name"]
        cfg  = {k: v for k, v in finalist.items() if k != "name"}
        for fold_idx in range(len(folds)):
            for seed in seeds:
                rid = f"stagec_{name}_fold{fold_idx}_seed{seed}"
                run_dir = os.path.join(EXPERIMENT, "stageC", name, f"fold{fold_idx}", f"seed{seed}")
                rows.append({
                    "run_id": rid, "kind": "full_pipeline", "run_dir": run_dir,
                    "images_dir": args.images_dir, "labels_dir": args.labels_dir,
                    "shells_dir": args.shells_dir,
                    "splits_json": os.path.join(run_dir, "splits.json"),
                    "seed": seed,
                    **cfg,
                })
    write_manifest(rows, args.out)
    print(f"({len(finalists)} finalists x {len(folds)} folds x {len(seeds)} seeds)")
    print("NOTE: each run_dir's splits.json must be resolved first, e.g.:")
    print("  for row in manifest: make_fold_splits.py resolve --folds-json ... "
         "--fold-idx <i> --shells-dir ... --out <run_dir>/splits.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p, needs_base_run=False, needs_p1_winner=False, needs_raw=False):
        p.add_argument("--out", required=True)
        if needs_raw:
            p.add_argument("--images-dir", required=True)
            p.add_argument("--labels-dir", required=True)
        p.add_argument("--shells-dir", required=True)
        p.add_argument("--splits-json", required=True)
        if needs_p1_winner:
            p.add_argument("--loss-p1", required=True)
            p.add_argument("--lr-p1", type=float, required=True)
        if needs_base_run:
            p.add_argument("--base-run", required=True,
                           help="run_dir of the crops-prep step (has crops/ + splits.json)")

    common(sub.add_parser("a1"), needs_raw=False)

    p = sub.add_parser("a1b"); common(p, needs_p1_winner=True)
    p = sub.add_parser("a1c"); common(p, needs_p1_winner=True)

    p = sub.add_parser("a1d"); common(p, needs_p1_winner=True)
    p.add_argument("--bw-floor-p1", type=float, required=True)
    p.add_argument("--bw-coef-p1",  type=float, required=True)
    p.add_argument("--gd-coef-p1",  type=float, required=True)

    p = sub.add_parser("crops-prep")
    common(p, needs_raw=True)
    p.add_argument("--p1-checkpoint", required=True)
    p.add_argument("--band-mm", type=float, default=5.0)

    p = sub.add_parser("a2")
    p.add_argument("--out", required=True)
    p.add_argument("--base-run", required=True)

    p = sub.add_parser("a2b")
    p.add_argument("--out", required=True)
    p.add_argument("--base-run", required=True)
    p.add_argument("--loss-p2", required=True)
    p.add_argument("--lr-p2", type=float, required=True)

    p = sub.add_parser("a3")
    common(p, needs_raw=True)
    p.add_argument("--combos", required=True, help="JSON list of hyperparam-dict combos")

    p = sub.add_parser("stagec")
    p.add_argument("--out", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--shells-dir", required=True)
    p.add_argument("--folds-json", required=True)
    p.add_argument("--finalists", required=True, help="JSON list of {name, ...hyperparams}")
    p.add_argument("--seeds", required=True, help="comma-separated, e.g. 42,43,44")

    args = parser.parse_args()
    {
        "a1": cmd_a1, "a1b": cmd_a1b, "a1c": cmd_a1c, "a1d": cmd_a1d,
        "crops-prep": cmd_crops_prep, "a2": cmd_a2, "a2b": cmd_a2b, "a3": cmd_a3,
        "stagec": cmd_stagec,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
