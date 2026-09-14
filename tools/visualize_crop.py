"""
Usage:
python -m tools.visualize_crop                    # picks first .npz in crops/
python -m tools.visualize_crop crops/case001.npz  # specific file
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS = os.path.join(REPO_ROOT, "crops")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help=".npz crop file to visualize")
    args = parser.parse_args()

    if args.file:
        path = args.file
    else:
        files = sorted(glob.glob(os.path.join(CROPS, "*.npz")))
        if not files:
            raise RuntimeError(f"No .npz files found in {CROPS}")
        path = files[0]

    print(f"Loading: {path}")
    data = np.load(path)

    img    = data["img"]            # (D, H, W) normalized native-res crop
    prior  = data["sdf_prior"]      # (D, H, W) upsampled Phase 1 signed distance
    mask   = data["mask"].astype(bool)  # (D, H, W) ground truth

    crop_shape   = img.shape
    native_shape = tuple(data["native_shape"])
    bbox         = data["bbox"]

    print(f"  native shape : {native_shape}")
    print(f"  crop shape   : {crop_shape}")
    print(f"  bbox         : {bbox.tolist()}")
    print(f"  head voxels  : {mask.sum()} / {mask.size} ({100*mask.mean():.1f}%)")

    n_slices = img.shape[0]
    mid      = n_slices // 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(os.path.basename(path), fontsize=12)
    plt.subplots_adjust(bottom=0.18, wspace=0.05)

    for ax, title in zip(axes, ["Image (native crop)", "Signed distance prior", "Ground truth mask"]):
        ax.set_title(title)
        ax.axis("off")

    im0 = axes[0].imshow(img[mid],   cmap="gray",    vmin=img.min(),  vmax=img.max())
    im1 = axes[1].imshow(prior[mid], cmap="coolwarm", vmin=-1,        vmax=1)
    im2 = axes[2].imshow(mask[mid],  cmap="gray",    vmin=0,          vmax=1)

    slice_label = fig.text(0.5, 0.11, f"Slice {mid}/{n_slices - 1}", ha="center", fontsize=9)

    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Slice", 0, n_slices - 1, valinit=mid, valstep=1)

    def update(val):
        s = int(slider.val)
        im0.set_data(img[s])
        im1.set_data(prior[s])
        im2.set_data(mask[s])
        slice_label.set_text(f"Slice {s}/{n_slices - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
