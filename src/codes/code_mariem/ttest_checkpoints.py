"""Compare checkpoint-48 (v2) and checkpoint-70 (v3) Dice on the RAW (non-denoised)
test sets with a paired t-test, and plot the result.

Both checkpoints are evaluated on exactly the same volumes, so the volumes are
paired: the paired t-test (and its non-parametric Wilcoxon counterpart) is the
correct test here.  Welch's t-test on the (mean, std) summaries is also reported
for reference, since it ignores the pairing and is therefore more conservative.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "runs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "stats")

# Raw (non-denoised) test sets only, present for both checkpoints.
DATASETS = [
    ("High-field", "v2/v2_ep48_highfield/highfield_AttUNet_checkpoint-48.csv",
                   "v3/v3_ep70_highfield/highfield_AttUNet_checkpoint-70.csv"),
    ("Low-field", "v2/v2_ep48_lowfield/lowfield_AttUNet_checkpoint-48.csv",
                  "v3/v3_ep70_lowfield/lowfield_AttUNet_checkpoint-70.csv"),
    ("CHUV TE0", "v2/chuv_te0_v2_ep48/otherscanners_AttUNet_checkpoint-48.csv",
                 "v3/chuv_te0_v3_ep70/otherscanners_AttUNet_checkpoint-70.csv"),
    ("CHUV all echoes", "v2/chuv_all_v2_ep48/otherscanners_AttUNet_checkpoint-48.csv",
                        "v3/chuv_all_v3_ep70/otherscanners_AttUNet_checkpoint-70.csv"),
]

CKPT_A, CKPT_B = "ckpt-48", "ckpt-70"
COL_A, COL_B = "#2a78d6", "#eb6834"          # categorical slots 1 & 2
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"


def load(path):
    df = pd.read_csv(os.path.join(RUNS_DIR, path))
    return df.set_index("Volume")["Dice"].astype(float)


def stars(p):
    return "n.s." if p >= 0.05 else "*" if p >= 0.01 else "**" if p >= 0.001 else "***"


def analyse():
    rows, paired = [], {}
    for name, path_a, path_b in DATASETS:
        a, b = load(path_a), load(path_b)
        common = a.index.intersection(b.index)
        assert len(common) == len(a) == len(b), f"{name}: volume mismatch"
        a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()
        d = a - b                                     # ckpt48 - ckpt70
        n = len(d)

        t_rel, p_rel = stats.ttest_rel(a, b)
        # Welch on the summary stats (mean/std), i.e. treating the runs as independent
        t_ind, p_ind = stats.ttest_ind(a, b, equal_var=False)
        try:
            w_stat, p_w = stats.wilcoxon(a, b)
        except ValueError:                            # all differences zero
            w_stat, p_w = np.nan, 1.0

        se = d.std(ddof=1) / np.sqrt(n)
        ci = stats.t.ppf(0.975, n - 1) * se
        cohen_dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan

        rows.append(dict(
            dataset=name, n=n,
            mean_48=a.mean(), std_48=a.std(ddof=1),
            mean_70=b.mean(), std_70=b.std(ddof=1),
            mean_diff=d.mean(), ci95_lo=d.mean() - ci, ci95_hi=d.mean() + ci,
            t_paired=t_rel, p_paired=p_rel, cohen_dz=cohen_dz,
            t_welch=t_ind, p_welch=p_ind,
            wilcoxon_W=w_stat, p_wilcoxon=p_w,
            n_better_48=int((d > 0).sum()), n_better_70=int((d < 0).sum()),
        ))
        paired[name] = (a, b, d)

    return pd.DataFrame(rows), paired




if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    stats_df, paired = analyse()
    stats_df.to_csv(os.path.join(OUT_DIR, "ttest_ckpt48_vs_ckpt70_raw.csv"), index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(stats_df.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    print("\nwrote", os.path.abspath(OUT_DIR))
