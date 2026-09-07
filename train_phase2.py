"""
Usage:
python train_phase2.py
python train_phase2.py --epochs 100 --lr 1e-3 --loss dice_bce
python train_phase2.py --run-dir runs/my_run/phase2
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset_phase2 import CropDataset, get_crop_splits
from loss import get_loss
from metrics import dice
from model import UNet3D

OUT = "C:/Users/user/Desktop/boundary_first_then_refine"


def train_epoch(model, loader, loss_fn, optimizer, device, scaler) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            pred = model(x)
            loss = loss_fn(pred, y)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss  = 0.0
    dice_scores = []

    with torch.no_grad():
        for x, y in loader:
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
    parser.add_argument("--run-dir",     type=str,   default=None,
                        help="directory to save checkpoint and train log")
    parser.add_argument("--crops-dir",   type=str,   default=None,
                        help="directory containing crop .npz files")
    parser.add_argument("--splits-json", type=str,   default=None,
                        help="path to splits.json")
    args = parser.parse_args()

    run_dir = args.run_dir or OUT
    os.makedirs(run_dir, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = get_loss(args.loss)
    scaler  = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None
    print(f"Device: {device}  Loss: {args.loss}  AMP: {scaler is not None}  Run: {run_dir}")

    train_paths, val_paths = get_crop_splits(crops_dir=args.crops_dir, splits_json=args.splits_json)
    print(f"Train: {len(train_paths)}  Val: {len(val_paths)}")

    train_loader = DataLoader(CropDataset(train_paths, augment=True),  batch_size=1, shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(CropDataset(val_paths,   augment=False), batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    model     = UNet3D(in_channels=2, base_channels=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir  = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "phase2_best.pth")
    log_path        = os.path.join(run_dir, "train_log.csv")

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_dice\n")

    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss         = train_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
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
            f"  val_dice={val_dice:.4f}"
            + ("  *" if improved else "")
        )
        print(line, flush=True)

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f}\n")

    print(f"\nBest val Dice : {best_dice:.4f}")
    print(f"Checkpoint    : {checkpoint_path}")
    print(f"Log           : {log_path}")


if __name__ == "__main__":
    main()
