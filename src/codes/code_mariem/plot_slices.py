"""Qualitative figure: slices the pretrained model missed and fine-tuning recovered.

Scans every test volume for slices where the pretrained baseline scored near zero
while the fine-tuned checkpoint scored high, then draws one randomly chosen
example per echo time (TE1/TE2/TE3).
"""
import argparse
import os
import random

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib

DATA_ROOT = "/home/mial/Documents/for_mariem"
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


RUN_TO_TE = {1: "TE1", 2: "TE1", 3: "TE1",
             4: "TE2", 5: "TE2", 6: "TE2",
             7: "TE3", 8: "TE3", 9: "TE3"}

FIELD_DIR = {"lowfield": "low_field", "highfield": "high_field"}


def _load(path):
    """Volume as float32 (X, Y, Z), dropping the trailing singleton some masks carry. Some masks are 4D"""
    arr = np.asanyarray(nib.load(path).dataobj).astype(np.float32)
    return arr[..., 0] if arr.ndim == 4 else arr


def _slice_dice(pred, gt):
    """Per-slice Dice along the last axis. Empty pred and empty gt scores 0."""
    inter = (pred * gt).sum(axis=(0, 1))
    return 2 * inter / (pred.sum(axis=(0, 1)) + gt.sum(axis=(0, 1)) +1e-8)


def _paths(field, volume):
    subject = volume.split("_")[0]
    field_dir = FIELD_DIR[field]
    return (os.path.join(DATA_ROOT, field_dir, subject, "ses-01", "anat", f"{volume}.nii.gz"),
            os.path.join(DATA_ROOT, field_dir, "derivatives", "mask", subject, "ses-01", "anat",
                         f"{volume}_mask.nii.gz"),
            os.path.join(RESULTS_ROOT, f"pretrained_{field}", field_dir, subject, "ses-01", "anat",
                         f"{volume}_pred.nii.gz"),
            os.path.join(RESULTS_ROOT, f"v2_ep48_{field}", field_dir, subject, "ses-01", "anat",
                         f"{volume}_pred.nii.gz"))


def find_candidates(fields=("lowfield", "highfield"), pre_max=0.1, ft_min=0.9, cache=None):
    """Every slice where pretrained Dice <= pre_max and fine-tuned Dice >= ft_min.

    Only slices carrying real brain are considered
    """
    rows = []
    for field in fields:
        csv_path = os.path.join(RESULTS_ROOT, f"v2_ep48_{field}",
                                f"{field}_AttUNet_checkpoint-48.csv")
        for volume in pd.read_csv(csv_path)["Volume"]:
            img_p, gt_p, pre_p, ft_p = _paths(field, volume)
            gt = _load(gt_p)
            d_pre = _slice_dice(_load(pre_p), gt)
            d_ft = _slice_dice(_load(ft_p), gt)

            gt_per_slice = gt.sum(axis=(0, 1))
            has_brain = gt_per_slice > 0.25 * gt_per_slice.max()

            run = int(volume.split("_")[3].replace("run-", ""))
            for k in np.where(has_brain & (d_pre <= pre_max) & (d_ft >= ft_min))[0]:
                rows.append(dict(field=field, volume=volume, subject=volume.split("_")[0],
                                 te=RUN_TO_TE[run], slice=int(k),
                                 dice_pre=float(d_pre[k]), dice_ft=float(d_ft[k])))

    found = pd.DataFrame(rows)

    return found


def _panel(ax, image, gt, pred, colour, title):
    ax.imshow(np.rot90(image), cmap="gray")
    if pred is not None:
        masked = np.ma.masked_where(np.rot90(pred) < 0.5, np.rot90(pred))
        ax.imshow(masked, alpha=0.45, cmap=matplotlib.colors.ListedColormap([colour]))
    ax.contour(np.rot90(gt), levels=[0.5], colors="lime", linewidths=1.0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_te_examples(out_path, seed=None, pre_max=0.1, ft_min=0.9,
                     fields=("lowfield", "highfield"), cache=None):
    """One randomly drawn example per TE, three panels each.

    Returns the DataFrame of the rows actually drawn, so the figure can be
    reproduced or reported alongside the numbers.
    """
    candidates = find_candidates(fields, pre_max, ft_min, cache)
    
    rng = random.Random(seed)
    picked = []
    for te in ("TE1", "TE2", "TE3"):
        pool = candidates[candidates.te == te]
 
        picked.append(pool.iloc[rng.randrange(len(pool))])

    fig, axs = plt.subplots(len(picked), 3, figsize=(9, 3.2 * len(picked)))
    axs = np.atleast_2d(axs)
    for row, pick in enumerate(picked):
        img_p, gt_p, pre_p, ft_p = _paths(pick.field, pick.volume)
        image, gt = _load(img_p), _load(gt_p)
        pre, ft = _load(pre_p), _load(ft_p)
        k = int(pick.slice)

        _panel(axs[row, 0], image[..., k], gt[..., k], None, "w",
               f"{pick.te}  {pick.volume.replace('_T2w', '')}\nslice {k}  ({pick.field})")
        _panel(axs[row, 1], image[..., k], gt[..., k], pre[..., k], "red",
               f"pretrained   Dice {pick.dice_pre:.3f}")
        _panel(axs[row, 2], image[..., k], gt[..., k], ft[..., k], "cyan",
               f"fine-tuned   Dice {pick.dice_ft:.3f}")

    fig.suptitle("Slices missed by the pretrained baseline and recovered by fine-tuning at each echo time"
                 "   (green = manual ground truth)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(picked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out",
                        type=str,
                        default="../results/figures/te_recovered_slices_2.png",
                        help="where to write the figure")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed; omit for a different draw each run")

    parser.add_argument("--pre_max",
                        type=float,
                        default=0.1,
                        help="pretrained per-slice dice must be at or below this")

    parser.add_argument("--ft_min",
                        type=float,
                        default=0.9,
                        help="fine-tuned per-slice dice must be at or above this")



    args = parser.parse_args()

    drawn = plot_te_examples(args.out, seed=args.seed, pre_max=args.pre_max,
                             ft_min=args.ft_min)
    print(f"\nwrote {args.out}\n")
