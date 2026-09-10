"""
Usage:
python -m tools.visualize_shell                      # picks first .npz in shells/
python -m tools.visualize_shell shells/case001.npz   # specific file
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHELLS = os.path.join(BASE_DIR, "shells")
PREDS  = os.path.join(BASE_DIR, "preds")


def load(path: str) -> dict:
    data = np.load(path)
    return {
        "img":    data["img"],
        "mask":   data["mask"].astype(bool),
        "target": data["sdf"],
    }


def load_pred(shell_path: str) -> np.ndarray | None:
    case = os.path.basename(shell_path).replace(".npz", "_pred.npy")
    pred_path = os.path.join(PREDS, case)
    if not os.path.exists(pred_path):
        return None
    pred = np.load(pred_path)
    # saved as (1, D, H, W) — squeeze channel dim
    if pred.ndim == 4:
        pred = pred[0]
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help=".npz file to visualize")
    args = parser.parse_args()

    if args.file:
        path = args.file
    else:
        files = sorted(glob.glob(os.path.join(SHELLS, "*.npz")))
        if not files:
            raise RuntimeError(f"No .npz files found in {SHELLS}")
        path = files[0]

    print(f"Loading: {path}")
    data = load(path)
    pred = load_pred(path)

    img    = data["img"]
    mask   = data["mask"]
    target = data["target"]

    has_pred = pred is not None
    n_panels = 4 if has_pred else 3
    n_slices = img.shape[0]
    mid      = n_slices // 2

    fig_w = 17 if has_pred else 13
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5))
    fig.suptitle(os.path.basename(path), fontsize=12)
    plt.subplots_adjust(bottom=0.18, wspace=0.05)

    titles = ["Image", "Mask overlay", "GT signed distance", "Predicted signed distance"]
    for ax, title in zip(axes, titles[:n_panels]):
        ax.set_title(title)
        ax.axis("off")

    im0 = axes[0].imshow(img[mid],    cmap="gray",    vmin=img.min(), vmax=img.max())
    im1 = axes[1].imshow(img[mid],    cmap="gray",    vmin=img.min(), vmax=img.max())
    ov  = axes[1].imshow(mask[mid],   cmap="Reds",    alpha=0.4,      vmin=0, vmax=1)
    im2 = axes[2].imshow(target[mid], cmap="coolwarm", vmin=-1,       vmax=1)
    im3 = axes[3].imshow(pred[mid],   cmap="coolwarm", vmin=-1,       vmax=1) if has_pred else None

    if not has_pred:
        print("No prediction found — run: python evaluate.py --save-preds")

    slice_label = fig.text(0.5, 0.11, f"Slice {mid}/{n_slices - 1}",
                           ha="center", fontsize=9)

    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Slice", 0, n_slices - 1,
                    valinit=mid, valstep=1)

    def update(val):
        s = int(slider.val)
        im0.set_data(img[s])
        im1.set_data(img[s])
        ov.set_data(mask[s])
        im2.set_data(target[s])
        if im3 is not None:
            im3.set_data(pred[s])
        slice_label.set_text(f"Slice {s}/{n_slices - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
