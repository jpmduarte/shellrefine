"""
Usage:
python visualize_shell.py                      # picks first .npz in shells/
python visualize_shell.py shells/case001.npz   # specific file
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

SHELLS = "C:/Users/user/Desktop/boundary_first_then_refine/shells"


def load(path: str) -> dict:
    data = np.load(path)
    return {
        "img":    data["img"],
        "mask":   data["mask"].astype(bool),
        "target": data["boundary_target"],
    }


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

    img    = data["img"]
    mask   = data["mask"]
    target = data["target"]

    n_slices = img.shape[0]
    mid = n_slices // 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(os.path.basename(path), fontsize=12)
    plt.subplots_adjust(bottom=0.18, wspace=0.05)

    titles = ["Image", "Mask overlay", "Boundary target"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis("off")

    # initial plots
    im0 = axes[0].imshow(img[mid],    cmap="gray",    vmin=img.min(),    vmax=img.max())
    im1 = axes[1].imshow(img[mid],    cmap="gray",    vmin=img.min(),    vmax=img.max())
    ov  = axes[1].imshow(mask[mid],   cmap="Reds",    alpha=0.4,         vmin=0, vmax=1)
    im2 = axes[2].imshow(target[mid], cmap="inferno", vmin=0,            vmax=1)

    slice_label = fig.text(0.5, 0.11, f"Slice {mid}/{n_slices - 1}",
                           ha="center", fontsize=9)

    # slider
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Slice", 0, n_slices - 1,
                    valinit=mid, valstep=1)

    def update(val):
        s = int(slider.val)
        im0.set_data(img[s])
        im1.set_data(img[s])
        ov.set_data(mask[s])
        im2.set_data(target[s])
        slice_label.set_text(f"Slice {s}/{n_slices - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
