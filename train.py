"""
Usage:
python train.py
python train.py --epochs 100 --batch-size 8 --lr 1e-3 --loss mse_bce
python train.py --run-dir runs/my_run/phase1
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import ShellDataset, get_splits
from loss import boundary_weighted_l1, gradient_difference_loss, get_loss
from metrics import dice_from_sdf, mae
from model import UNet3D
from profiler import Profiler, model_summary

OUT = os.path.dirname(os.path.abspath(__file__))

# F.binary_cross_entropy requires inputs in [0,1]; phase 1's model outputs a tanh SDF
# in [-1,1]. Selecting one of these with a tanh model is a CUDA device-side assert, not
# a graceful error — reject it up front instead.
BCE_FAMILY_LOSSES = {"bce", "dice_bce", "mse_bce"}


def seed_worker(worker_id):
    # ShellDataset keeps its own np.random.Generator (self._rng), not the legacy global
    # numpy RNG — a fork copies that Generator's state byte-for-byte into every worker,
    # so without this every worker would draw the identical "random" sequence. Replace
    # each worker's copy with a fresh Generator seeded from the per-worker seed torch
    # already derives once `generator=` is set on the DataLoader.
    import numpy as np
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(info.seed % 2**32)


def train_epoch(model, loader, loss_fn, optimizer, device,
                 boundary_weight_coef: float = 0.0, gd_coef: float = 0.0) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    data_wait  = 0.0
    seen       = 0

    iterator = iter(loader)
    while True:
        fetch_start = time.perf_counter()
        try:
            x, y, w = next(iterator)
        except StopIteration:
            break
        data_wait += time.perf_counter() - fetch_start

        seen += x.size(0)
        x, y, w = x.to(device), y.to(device), w.to(device)
        pred    = model(x)
        loss    = loss_fn(pred, y)
        if boundary_weight_coef > 0:
            loss = loss + boundary_weight_coef * boundary_weighted_l1(pred, y, w)
        if gd_coef > 0:
            loss = loss + gd_coef * gradient_difference_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset), data_wait, seen


def val_epoch(model, loader, loss_fn, device) -> tuple[float, float, float]:
    model.eval()
    total_loss  = 0.0
    mae_scores  = []
    dice_scores = []

    with torch.no_grad():
        for x, y, _w in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            total_loss += loss_fn(pred, y).item() * x.size(0)

            for p, t in zip(pred, y):
                mae_scores.append(mae(p, t))
                score = dice_from_sdf(p, t)
                if score is not None:
                    dice_scores.append(score)

    mean_mae  = sum(mae_scores)  / len(mae_scores)  if mae_scores  else 0.0
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return total_loss / len(loader.dataset), mean_mae, mean_dice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=2)
    parser.add_argument("--base-ch",    type=int,   default=16)
    parser.add_argument("--loss",        type=str,   default="l1")
    parser.add_argument("--boundary-weight-floor", type=float, default=1.0,
                        help="1.0 (default) = off, exact fallback to the unweighted loss. "
                             "Below 1.0, adds an L1 term weighting the saturated region "
                             "(|sdf| >= --boundary-weight-thr) at this floor instead of 1.0")
    parser.add_argument("--boundary-weight-thr",   type=float, default=0.7,
                        help="normalised |sdf| threshold above which a voxel is 'saturated'")
    parser.add_argument("--boundary-weight-coef",  type=float, default=1.0,
                        help="strength of the added boundary-weighted L1 term (ignored "
                             "when --boundary-weight-floor is 1.0)")
    parser.add_argument("--gd-coef",     type=float, default=0.0,
                        help="0.0 (default) = off, exact fallback. Strength of an added "
                             "gradient-difference term (matches local slope of the SDF, "
                             "not just its value) on top of --loss")
    parser.add_argument("--seed",        type=int,   default=None,
                        help="None (default) = unseeded, today's behaviour. Fixes model "
                             "init, augmentation/patch RNG and DataLoader shuffle order")
    parser.add_argument("--run-dir",     type=str,   default=None,
                        help="directory to save checkpoint and train log")
    parser.add_argument("--shells-dir",  type=str,   default=None,
                        help="directory containing shell .npz files")
    parser.add_argument("--splits-json", type=str,   default=None,
                        help="path to splits.json")
    parser.add_argument("--profile-dir", type=str,   default=None,
                        help="directory to append profiling logs to")
    args = parser.parse_args()

    if args.loss in BCE_FAMILY_LOSSES:
        parser.error(
            f"--loss {args.loss} is incompatible with phase 1's tanh output "
            f"([-1,1]): F.binary_cross_entropy requires [0,1] and raises a CUDA "
            f"device-side assert on GPU. Use l1 or mse."
        )

    run_dir = args.run_dir or OUT
    os.makedirs(run_dir, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        # cudnn picks convolution algorithms by benchmarking at runtime, which is not
        # itself seeded — same-seed runs otherwise still drift measurably (confirmed:
        # val_mae differed by several % across two --seed 42 runs before this was added).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = get_loss(args.loss)
    print(f"Device: {device}  Loss: {args.loss}  Seed: {args.seed}  Run: {run_dir}")

    train_paths, val_paths = get_splits(val_fraction=0.2, shells_dir=args.shells_dir, splits_json=args.splits_json)
    print(f"Train: {len(train_paths)}  Val: {len(val_paths)}")

    weight_args = dict(boundary_weight_floor=args.boundary_weight_floor,
                       boundary_weight_thr=args.boundary_weight_thr)
    if args.boundary_weight_floor < 1.0:
        print(f"Boundary-weighted loss: floor={args.boundary_weight_floor}"
              f"  thr={args.boundary_weight_thr}  coef={args.boundary_weight_coef}")
    if args.gd_coef > 0:
        print(f"Gradient-difference loss: coef={args.gd_coef}")

    loader_gen = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    worker_init = seed_worker if args.seed is not None else None

    train_loader = DataLoader(ShellDataset(train_paths, augment=True,  seed=args.seed, **weight_args),
                              batch_size=args.batch_size, shuffle=True,  num_workers=args.workers,
                              pin_memory=True, generator=loader_gen, worker_init_fn=worker_init)
    val_loader   = DataLoader(ShellDataset(val_paths,   augment=False, seed=args.seed, **weight_args),
                              batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                              pin_memory=True, generator=loader_gen, worker_init_fn=worker_init)

    model     = UNet3D(in_channels=1, base_channels=args.base_ch, out_activation="tanh").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir  = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "phase1_best.pth")
    log_path        = os.path.join(run_dir, "train_log.csv")

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_mae,val_dice\n")

    best_mae = float("inf")

    prof = Profiler("phase1_train", config=vars(args), profile_dir=args.profile_dir)
    prof.config_update({
        "model": model_summary(model),
        "data":  {"train_cases": len(train_paths), "val_cases": len(val_paths)},
    })

    with prof:
        for epoch in range(1, args.epochs + 1):
            with prof.span("train_epoch") as span:
                train_loss, data_wait, seen = train_epoch(
                    model, train_loader, loss_fn, optimizer, device,
                    boundary_weight_coef=args.boundary_weight_coef if args.boundary_weight_floor < 1.0 else 0.0,
                    gd_coef=args.gd_coef,
                )
                span["items"]       = seen
                span["data_wait_s"] = round(data_wait, 3)

            with prof.span("val_epoch"):
                val_loss, val_mae, val_dice = val_epoch(model, val_loader, loss_fn, device)

            scheduler.step()

            improved = val_mae < best_mae
            if improved:
                best_mae = val_mae
                torch.save(model.state_dict(), checkpoint_path)

            line = (
                f"Epoch {epoch:03d}/{args.epochs}"
                f"  train_loss={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}"
                f"  val_mae={val_mae:.4f}"
                f"  val_dice={val_dice:.4f}"
                + ("  *" if improved else "")
            )
            print(line, flush=True)

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_mae:.4f},{val_dice:.4f}\n")

    print(f"\nBest val MAE : {best_mae:.4f}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Log          : {log_path}")


if __name__ == "__main__":
    main()
