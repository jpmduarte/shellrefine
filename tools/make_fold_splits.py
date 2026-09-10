"""
Two-step k-fold split generator for Stage C of the experiment.

Step 1 — generate the partition once (portable, no paths baked in):
    python -m tools.make_fold_splits generate --shells-dir runs/<base>/shells --k 5 --seed 12345 \
        --out folds/folds.json

Step 2 — resolve one fold into a splits.json in the exact format dataset.get_splits() /
dataset_phase2.get_crop_splits() already read without any code changes (both have a
"load if the file exists" branch):
    python -m tools.make_fold_splits resolve --folds-json folds/folds.json --fold-idx 0 \
        --shells-dir runs/<base>/shells --out runs/<run>/splits.json
"""

import argparse
import json
import os


def generate(shells_dir: str, k: int, seed: int, out_path: str) -> None:
    import glob

    import numpy as np

    # Shell files are already named exactly f"{case_id}.npz" by make_shells.py — plain
    # de-suffixing, not make_shells.case_id() (that one strips "_0000.nii.gz"/".nii.gz"
    # off the *original* image filenames, not ".npz").
    ids = sorted(
        os.path.basename(p)[:-len(".npz")]
        for p in glob.glob(os.path.join(shells_dir, "*.npz"))
    )
    if not ids:
        raise RuntimeError(f"No .npz files found in {shells_dir}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))

    # k roughly-equal groups; any remainder goes one-per-fold to the first folds.
    folds = [[] for _ in range(k)]
    for rank, idx in enumerate(order):
        folds[rank % k].append(ids[idx])

    payload = {"k": k, "seed": seed, "n_cases": len(ids), "folds": folds}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    sizes = [len(f) for f in folds]
    print(f"{len(ids)} cases -> {k} folds (sizes {sizes})  seed={seed}")
    print(f"Saved: {out_path}")


def resolve(folds_json: str, fold_idx: int, shells_dir: str, out_path: str) -> None:
    with open(folds_json) as f:
        payload = json.load(f)

    folds = payload["folds"]
    if not (0 <= fold_idx < len(folds)):
        raise ValueError(f"--fold-idx must be in [0, {len(folds) - 1}]")

    val_ids   = set(folds[fold_idx])
    train_ids = set(cid for i, fold in enumerate(folds) if i != fold_idx for cid in fold)

    def to_path(cid: str) -> str:
        path = os.path.join(shells_dir, f"{cid}.npz")
        if not os.path.exists(path):
            raise RuntimeError(f"Fold references case '{cid}' but {path} does not exist")
        return path

    splits = {
        "train": sorted(to_path(c) for c in train_ids),
        "val":   sorted(to_path(c) for c in val_ids),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Fold {fold_idx}/{len(folds)}: train={len(splits['train'])}  val={len(splits['val'])}")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="build the k-way case-ID partition once")
    gen.add_argument("--shells-dir", type=str, required=True)
    gen.add_argument("--k",          type=int, default=5)
    gen.add_argument("--seed",       type=int, default=12345,
                     help="seeds the fold PARTITION itself, unrelated to --seed on "
                          "train.py/train_phase2.py (that seeds a training run)")
    gen.add_argument("--out",        type=str, required=True)

    res = sub.add_parser("resolve", help="turn one fold into a splits.json")
    res.add_argument("--folds-json", type=str, required=True)
    res.add_argument("--fold-idx",   type=int, required=True)
    res.add_argument("--shells-dir", type=str, required=True)
    res.add_argument("--out",        type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "generate":
        generate(args.shells_dir, args.k, args.seed, args.out)
    else:
        resolve(args.folds_json, args.fold_idx, args.shells_dir, args.out)


if __name__ == "__main__":
    main()
