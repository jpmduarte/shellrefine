"""
Trains the refinement network on patches sampled along the phase 1 boundary band,
so the convolutions only ever see the region that still needs deciding.

Usage:
python train_phase2.py
python train_phase2.py --epochs 100 --lr 1e-3 --loss dice_bce
python train_phase2.py --patch-size 96 --patches-per-case 12
python train_phase2.py --run-dir runs/my_run/phase2
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset_phase2 import CropDataset, collate_patches, get_crop_splits
from loss import boundary_weighted_bce, get_loss
from metrics import dice
from model import UNet3D
from profiler import Profiler, model_summary

OUT = os.path.dirname(os.path.abspath(__file__))


def seed_worker(worker_id):
    # CropDataset keeps a persistent stream (self._seed_rng), not the legacy global numpy
    # RNG — a fork copies its state byte-for-byte into every worker, so without this every
    # worker would draw the identical per-item seed sequence. Same fix as train.py's
    # ShellDataset, applied to the differently-named attribute.
    import numpy as np
    info = torch.utils.data.get_worker_info()
    info.dataset._seed_rng = np.random.default_rng(info.seed % 2**32)


def train_epoch(model, loader, loss_fn, optimizer, device, scaler,
                 boundary_weight_coef: float = 0.0) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    data_wait  = 0.0
    patches    = 0
    steps      = 0

    iterator = iter(loader)
    while True:
        fetch_start = time.perf_counter()
        try:
            x, y, w = next(iterator)
        except StopIteration:
            break
        data_wait += time.perf_counter() - fetch_start

        patches += x.shape[0]
        steps   += 1
        x, y, w  = x.to(device), y.to(device), w.to(device)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            pred = model(x)
            loss = loss_fn(pred, y)
            if boundary_weight_coef > 0:
                loss = loss + boundary_weight_coef * boundary_weighted_bce(pred, y, w)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(steps, 1), data_wait, patches


def val_epoch(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss  = 0.0
    dice_scores = []

    with torch.no_grad():
        for x, y, _w in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            total_loss += loss_fn(pred, y).item()

            for p, t in zip(pred, y):
                score = dice(p, t)
                if score is not None:
                    dice_scores.append(score)

    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return total_loss / len(loader), mean_dice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--workers", type=int,   default=2)
    parser.add_argument("--base-ch", type=int,   default=16)
    parser.add_argument("--loss",        type=str,   default="dice_bce")
    parser.add_argument("--amp",         action="store_true")
    parser.add_argument("--patch-size",       type=int,   default=64,
                        help="cubic patch size sampled along the boundary band")
    parser.add_argument("--patches-per-case", type=int,   default=8,
                        help="patches drawn per case per epoch")
    parser.add_argument("--batch-cases",      type=int,   default=1,
                        help="cases per batch; effective batch = this x patches-per-case")
    parser.add_argument("--boundary-weight-floor", type=float, default=1.0,
                        help="1.0 (default) = off, exact fallback to the unweighted loss. "
                             "Below 1.0, adds a BCE term that weights voxels outside the "
                             "band (already computed by make_crops) at this weight instead "
                             "of the full 1.0 given to band voxels — e.g. try 0.2")
    parser.add_argument("--boundary-weight-coef",  type=float, default=1.0,
                        help="strength of the added boundary-weighted BCE term (ignored "
                             "when --boundary-weight-floor is 1.0)")
    parser.add_argument("--seed",        type=int,   default=None,
                        help="None (default) = unseeded, today's behaviour. Fixes model "
                             "init, patch-sampling/augmentation RNG and DataLoader shuffle order")
    parser.add_argument("--run-dir",     type=str,   default=None,
                        help="directory to save checkpoint and train log")
    parser.add_argument("--crops-dir",   type=str,   default=None,
                        help="directory containing crop .npz files")
    parser.add_argument("--splits-json", type=str,   default=None,
                        help="path to splits.json")
    parser.add_argument("--profile-dir", type=str,   default=None,
                        help="directory to append profiling logs to")
    args = parser.parse_args()

    run_dir = args.run_dir or OUT
    os.makedirs(run_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = get_loss(args.loss)
    scaler  = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None
    print(f"Device: {device}  Loss: {args.loss}  Seed: {args.seed}  AMP: {scaler is not None}  Run: {run_dir}")

    train_paths, val_paths = get_crop_splits(crops_dir=args.crops_dir, splits_json=args.splits_json)
    print(f"Train: {len(train_paths)}  Val: {len(val_paths)}")

    patch_args = dict(
        patch_size=args.patch_size,
        patches_per_case=args.patches_per_case,
        boundary_weight_floor=args.boundary_weight_floor,
    )
    print(
        f"Patches: {args.patch_size}^3 x {args.patches_per_case}/case"
        f"  Effective batch: {args.batch_cases * args.patches_per_case}"
    )
    if args.boundary_weight_floor < 1.0:
        print(f"Boundary-weighted loss: floor={args.boundary_weight_floor}"
              f"  coef={args.boundary_weight_coef}  (weight 1.0 inside the band, "
              f"floor outside)")

    train_set = CropDataset(train_paths, augment=True,  base_seed=args.seed, **patch_args)
    val_set   = CropDataset(val_paths,   augment=False, deterministic=True, **patch_args)

    loader_gen  = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    worker_init = seed_worker if args.seed is not None else None

    train_loader = DataLoader(train_set, batch_size=args.batch_cases, shuffle=True,  num_workers=args.workers,
                              pin_memory=True, collate_fn=collate_patches,
                              generator=loader_gen, worker_init_fn=worker_init)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_cases, shuffle=False, num_workers=args.workers,
                              pin_memory=True, collate_fn=collate_patches)

    model     = UNet3D(in_channels=2, base_channels=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir  = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "phase2_best.pth")
    log_path        = os.path.join(run_dir, "train_log.csv")

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_patch_dice\n")

    best_dice = -1.0

    prof = Profiler("phase2_train", config=vars(args), profile_dir=args.profile_dir)
    prof.config_update({
        "model": model_summary(model),
        "data":  {"train_cases": len(train_paths), "val_cases": len(val_paths)},
    })

    with prof:
        for epoch in range(1, args.epochs + 1):
            with prof.span("train_epoch") as span:
                train_loss, data_wait, patches = train_epoch(
                    model, train_loader, loss_fn, optimizer, device, scaler,
                    boundary_weight_coef=args.boundary_weight_coef if args.boundary_weight_floor < 1.0 else 0.0,
                )
                span["items"]       = patches
                span["data_wait_s"] = round(data_wait, 3)

            with prof.span("val_epoch"):
                val_loss, val_dice = val_epoch(model, val_loader, loss_fn, device)

            scheduler.step()

            improved = val_dice > best_dice
            if improved:
                best_dice = val_dice
                torch.save(model.state_dict(), checkpoint_path)

            line = (
                f"Epoch {epoch:03d}/{args.epochs}"
                f"  train_loss={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}"
                f"  val_patch_dice={val_dice:.4f}"
                + ("  *" if improved else "")
            )
            print(line, flush=True)

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f}\n")

    print(f"\nBest val patch Dice : {best_dice:.4f}  (band only — full-volume Dice comes from evaluate_phase2.py)")
    print(f"Checkpoint    : {checkpoint_path}")
    print(f"Log           : {log_path}")


if __name__ == "__main__":
    main()
