"""
Runs the refinement network only on tiles that contain boundary band, then stitches
the result over phase 1's coarse mask. Everything away from the band keeps phase 1's
answer, so no compute is spent deep inside or far outside the head.

Reports three numbers per case:
  coarse  — phase 1's upsampled mask alone (the baseline phase 2 has to beat)
  refined — after phase 2 rewrites the band
  band    — Dice restricted to the band, where the refinement actually acts

Usage:
python evaluate_phase2.py
python evaluate_phase2.py --split train
python evaluate_phase2.py --run-dir runs/20260907_143022/phase2
"""

import argparse
import itertools
import json
import os

import numpy as np
import torch

from dataset_phase2 import get_crop_splits, pad_to_patch
from metrics import asd, dice, hd95
from model import UNet3D
from profiler import Profiler, model_summary

OUT = os.path.dirname(os.path.abspath(__file__))


def axis_origins(dim: int, patch: int, stride: int) -> list[int]:
    last  = max(dim - patch, 0)
    stops = list(range(0, last + 1, stride))
    if stops[-1] != last:
        stops.append(last)
    return stops


def tile_origins(shape: tuple[int, ...], patch: int, stride: int):
    return itertools.product(*(axis_origins(d, patch, stride) for d in shape))


@torch.no_grad()
def predict_case(model, path: str, device, patch: int, stride: int, tile_batch: int):
    data = np.load(path)

    mask    = data["mask"].astype(np.float32)
    spacing = tuple(data["spacing"].tolist())
    native  = data["img"].shape

    # Same padding rules as training — the sdf pads with +1, not 0.
    arrays = pad_to_patch({
        "img":       data["img"].astype(np.float32),
        "sdf_prior": data["sdf_prior"].astype(np.float32),
        "band":        data["band"].astype(np.float32),
        "coarse_mask": data["coarse_mask"].astype(np.float32),
    }, patch)

    img_p, prior_p, coarse_p = arrays["img"], arrays["sdf_prior"], arrays["coarse_mask"]

    band  = arrays["band"] > 0.5
    accum = np.zeros(img_p.shape, dtype=np.float32)
    count = np.zeros(img_p.shape, dtype=np.float32)

    origins = [
        o for o in tile_origins(img_p.shape, patch, stride)
        if band[tuple(slice(s, s + patch) for s in o)].any()
    ]

    for start in range(0, len(origins), tile_batch):
        chunk  = origins[start:start + tile_batch]
        slices = [tuple(slice(s, s + patch) for s in o) for o in chunk]

        batch = np.stack([np.stack([img_p[sl], prior_p[sl]]) for sl in slices])
        preds = model(torch.from_numpy(batch).to(device)).squeeze(1).cpu().numpy()

        for sl, pred in zip(slices, preds):
            accum[sl] += pred
            count[sl] += 1.0

    refined = coarse_p.copy()
    touched = band & (count > 0)
    refined[touched] = accum[touched] / count[touched]

    unpad = tuple(slice(0, s) for s in native)
    return refined[unpad], coarse_p[unpad], band[unpad], mask, spacing, len(origins)


def evaluate(model, paths, device, threshold, patch, stride,
             tile_batch, prof, save_dir=None) -> list[dict]:
    results = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for path in paths:
        with prof.span("inference_case") as span:
            refined, coarse, band, mask, spacing, n_tiles = predict_case(
                model, path, device, patch, stride, tile_batch
            )
            span["tiles"]    = n_tiles
            span["items"]    = n_tiles
            span["band_pct"] = round(100 * float(band.mean()), 2)
            span["voxels"]   = int(mask.size)

        if save_dir:
            np.save(
                os.path.join(save_dir, os.path.basename(path).replace(".npz", "_pred.npy")),
                refined,
            )

        refined_bin = refined >= threshold
        coarse_bin  = coarse.astype(bool)
        mask_bin    = mask.astype(bool)

        refined_t = torch.from_numpy(refined)
        coarse_t  = torch.from_numpy(coarse)
        mask_t    = torch.from_numpy(mask)
        band_t    = torch.from_numpy(band)

        # Surface distances run on the same native-resolution masks as Dice, in mm
        # (via the crop's own spacing) — the numbers you'd actually report, not a
        # voxel-count proxy for them.
        results.append({
            "case":         os.path.basename(path),
            "coarse":       dice(coarse_t,  mask_t, threshold=threshold),
            "refined":      dice(refined_t, mask_t, threshold=threshold),
            "band":         dice(refined_t[band_t], mask_t[band_t], threshold=threshold),
            "hd95_coarse":  hd95(coarse_bin,  mask_bin, spacing),
            "hd95_refined": hd95(refined_bin, mask_bin, spacing),
            "asd_coarse":   asd(coarse_bin,  mask_bin, spacing),
            "asd_refined":  asd(refined_bin, mask_bin, spacing),
            "tiles":        n_tiles,
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
    parser.add_argument("--patch-size",  type=int,   default=64,
                        help="must match training")
    parser.add_argument("--stride",      type=int,   default=None,
                        help="tile stride; defaults to half the patch (50%% overlap)")
    parser.add_argument("--tile-batch",  type=int,   default=8,
                        help="tiles pushed through the network at once")
    parser.add_argument("--crops-dir",   type=str,   default=None,
                        help="directory containing crop .npz files")
    parser.add_argument("--splits-json", type=str,   default=None,
                        help="path to splits.json")
    parser.add_argument("--profile-dir", type=str,   default=None,
                        help="directory to append profiling logs to")
    parser.add_argument("--no-save-preds", action="store_true",
                        help="skip writing eval_*/preds/*.npy (~500MB/run of full-res "
                             "float32 volumes) — off by default, exact fallback")
    args = parser.parse_args()

    stride = args.stride or max(args.patch_size // 2, 1)

    run_dir    = args.run_dir
    checkpoint = args.checkpoint or (
        os.path.join(run_dir, "checkpoints", "phase2_best.pth") if run_dir
        else os.path.join(OUT, "checkpoints", "phase2_best.pth")
    )
    save_dir     = os.path.join(run_dir, f"eval_{args.split}", "preds") if run_dir and not args.no_save_preds else None
    summary_path = os.path.join(run_dir, f"eval_{args.split}", "summary.txt") if run_dir else None

    if summary_path:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths, val_paths = get_crop_splits(crops_dir=args.crops_dir, splits_json=args.splits_json)
    paths = val_paths if args.split == "val" else train_paths

    model = UNet3D(in_channels=2, base_channels=args.base_ch).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    print(f"Checkpoint: {checkpoint}")
    print(f"Split: {args.split}  Cases: {len(paths)}  Threshold: {args.threshold}")
    print(f"Tiles: {args.patch_size}^3  stride={stride}  (band read from the crops)\n")

    prof = Profiler("phase2_eval", config=vars(args), profile_dir=args.profile_dir)
    prof.config_update({"model": model_summary(model), "data": {"cases": len(paths)}})

    with prof:
        results = evaluate(
            model, paths, device,
            threshold=args.threshold,
            patch=args.patch_size,
            stride=stride,
            tile_batch=args.tile_batch,
            prof=prof,
            save_dir=save_dir,
        )

    def column(key):
        return [r[key] for r in results if r[key] is not None]

    with open(summary_path, "w") if summary_path else open(os.devnull, "w") as f:
        print_and_write(
            f"{'Case':<30}  {'coarse':>7}  {'refined':>7}  {'band':>7}"
            f"  {'hd95_c':>7}  {'hd95_r':>7}  {'asd_c':>6}  {'asd_r':>6}  {'tiles':>5}", f
        )
        print_and_write("-" * 100, f)
        for r in results:
            dice_cells = "  ".join(
                f"{r[k]:.4f}" if r[k] is not None else "   skip"
                for k in ("coarse", "refined", "band")
            )
            dist_cells = "  ".join(
                f"{r[k]:6.2f}" if r[k] is not None else "  skip"
                for k in ("hd95_coarse", "hd95_refined", "asd_coarse", "asd_refined")
            )
            print_and_write(f"  {r['case']:<28}  {dice_cells}  {dist_cells}  {r['tiles']:>5}", f)
        print_and_write("-" * 100, f)

        coarse, refined, band = column("coarse"), column("refined"), column("band")
        hd95_c, hd95_r = column("hd95_coarse"), column("hd95_refined")
        asd_c,  asd_r  = column("asd_coarse"),  column("asd_refined")

        if refined:
            print_and_write(f"\n  Mean Dice coarse  : {np.mean(coarse):.4f}", f)
            print_and_write(f"  Mean Dice refined : {np.mean(refined):.4f}"
                            f"  (delta {np.mean(refined) - np.mean(coarse):+.4f})", f)
            print_and_write(f"  Mean Dice in band : {np.mean(band):.4f}", f)
            print_and_write(f"  Std  Dice refined : {np.std(refined):.4f}", f)
            print_and_write(f"  Min  Dice refined : {np.min(refined):.4f}", f)
            print_and_write(f"  Max  Dice refined : {np.max(refined):.4f}", f)

        if hd95_r:
            print_and_write(f"\n  Mean HD95 coarse  : {np.mean(hd95_c):.2f} mm", f)
            print_and_write(f"  Mean HD95 refined : {np.mean(hd95_r):.2f} mm"
                            f"  (delta {np.mean(hd95_r) - np.mean(hd95_c):+.2f})", f)
            print_and_write(f"  Max  HD95 refined : {np.max(hd95_r):.2f} mm", f)

        if asd_r:
            print_and_write(f"\n  Mean ASD  coarse  : {np.mean(asd_c):.2f} mm", f)
            print_and_write(f"  Mean ASD  refined : {np.mean(asd_r):.2f} mm"
                            f"  (delta {np.mean(asd_r) - np.mean(asd_c):+.2f})", f)
            print_and_write(f"  Max  ASD  refined : {np.max(asd_r):.2f} mm", f)

        skipped = len(results) - len(refined)
        if skipped:
            print_and_write(f"\n  Skipped (empty mask): {skipped}", f)

    if summary_path:
        # Machine-readable twin of summary.txt, for tools.aggregate_results.
        def agg(vals):
            return {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
            } if vals else None

        summary_json = {
            "checkpoint": checkpoint,
            "split":      args.split,
            "cases":      results,
            "aggregate": {
                "dice_coarse": agg(coarse), "dice_refined": agg(refined), "dice_band": agg(band),
                "hd95_coarse": agg(hd95_c), "hd95_refined": agg(hd95_r),
                "asd_coarse":  agg(asd_c),  "asd_refined":  agg(asd_r),
                "skipped": skipped,
            },
        }
        with open(os.path.join(os.path.dirname(summary_path), "summary.json"), "w") as f:
            json.dump(summary_json, f, indent=2)

        print(f"\nSummary saved to: {summary_path}")
    if save_dir:
        print(f"Predictions saved to: {save_dir}")


if __name__ == "__main__":
    main()
