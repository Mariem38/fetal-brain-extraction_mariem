#!/usr/bin/env python
"""Paired t-tests on SUBJECT-level mean Dice, high field and low field.

Volumes are averaged within subject first, so the unit of analysis is the
subject and the paired assumption holds (repeated volumes per subject are not
independent observations).

usage:  python paired_dice_stats.py [runs_dir] [out_dir] [fig_dir]
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO = "/home/mial/code_mariem/fetal-brain-extraction_mariem"
RUNS = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/src/results/runs"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{REPO}/src/results/stats"
FIG = sys.argv[3] if len(sys.argv) > 3 else f"{REPO}/src/figures"

SOURCES = {
    ("highfield", "pretrained"): "pretrained/pretrained_highfield/highfield_AttUNet.csv",
    ("highfield", "v2"):         "v2/v2_ep48_highfield/highfield_AttUNet_checkpoint-48.csv",
    ("highfield", "v3"):         "v3/v3_ep70_highfield/highfield_AttUNet_checkpoint-70.csv",
    ("lowfield", "pretrained"):  "pretrained/pretrained_lowfield/lowfield_AttUNet.csv",
    ("lowfield", "v2"):          "v2/v2_ep48_lowfield/lowfield_AttUNet_checkpoint-48.csv",
    ("lowfield", "v3"):          "v3/v3_ep70_lowfield/lowfield_AttUNet_checkpoint-70.csv",
}
MODELS = ["pretrained", "v2", "v3"]
FIELDS = ["highfield", "lowfield"]
PAIRS = [("pretrained", "v3"), ("pretrained", "v2"), ("v2", "v3")]

# reference palette (pre-validated), light surface
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, RED = "#2a78d6", "#e34948"


def subject_means():
    """Subject x model mean Dice, per field. Volumes collapsed within subject."""
    out = {}
    for field in FIELDS:
        cols = {}
        for model in MODELS:
            d = pd.read_csv(os.path.join(RUNS, SOURCES[(field, model)]))
            cols[model] = d.groupby("Subject")["Dice"].mean()
        df = pd.DataFrame(cols).sort_index()
        assert not df.isna().any().any(), f"{field}: subjects do not align across models"
        out[field] = df
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def test_pair(a, b):
    """Paired stats for arrays a (reference) and b (comparison); diff = b - a."""
    d = b - a
    n = len(d)
    t, p = stats.ttest_rel(b, a)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    try:
        _, wp = stats.wilcoxon(b, a)
    except ValueError:
        wp = np.nan
    return {
        "n": n, "mean_a": a.mean(), "mean_b": b.mean(),
        "mean_diff": d.mean(), "sd_diff": sd,
        "ci_lo": d.mean() - crit * se, "ci_hi": d.mean() + crit * se,
        "t": t, "df": n - 1, "p": p,
        "cohen_dz": d.mean() / sd if sd else np.nan,
        "wilcoxon_p": wp,
        "shapiro_p": stats.shapiro(d).pvalue if n >= 3 else np.nan,
        "n_improved": int((d > 0).sum()),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    sm = subject_means()

    rows = []
    for field in FIELDS:
        df = sm[field]
        res = [test_pair(df[a].values, df[b].values) for a, b in PAIRS]
        for (a, b), r in zip(PAIRS, res):
            r.update({"field": field, "baseline": a, "model": b})
        for r, adj in zip(res, holm([r["p"] for r in res])):
            r["p_holm"] = adj
        rows.extend(res)

    tab = pd.DataFrame(rows)[[
        "field", "baseline", "model", "n", "mean_a", "mean_b", "mean_diff",
        "sd_diff", "ci_lo", "ci_hi", "t", "df", "p", "p_holm", "cohen_dz",
        "wilcoxon_p", "shapiro_p", "n_improved"]]
    tab.to_csv(f"{OUT}/paired_ttest_subject_level.csv", index=False)
    for field in FIELDS:
        sm[field].to_csv(f"{OUT}/subject_means_{field}.csv")

    # ------------------------------- table ---------------------------------- #
    lines = []
    lines.append("PAIRED t-TEST ON SUBJECT-LEVEL MEAN DICE")
    lines.append("=" * 100)
    for field in FIELDS:
        df = sm[field]
        lines.append(f"\n{field.upper()}  (n = {len(df)} subjects, "
                     f"{', '.join(df.index)})")
        lines.append("  subject-level mean Dice:")
        for m in MODELS:
            lines.append(f"    {m:12s} {df[m].mean():.4f} +/- {df[m].std(ddof=1):.4f}"
                         f"   [{df[m].min():.4f}, {df[m].max():.4f}]")
        lines.append("")
        lines.append(f"  {'comparison':<24}{'diff':>9}{'95% CI':>20}{'t':>9}"
                     f"{'p':>10}{'p_holm':>9}{'dz':>8}{'wilcox':>9}{'up':>5}")
        lines.append("  " + "-" * 101)
        for r in rows:
            if r["field"] != field:
                continue
            ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
            lines.append(
                f"  {r['baseline'] + ' -> ' + r['model']:<24}"
                f"{r['mean_diff']:>+9.4f}{ci:>22}"
                f"{r['t']:>9.3f}{r['p']:>10.4f}{r['p_holm']:>9.4f}"
                f"{r['cohen_dz']:>8.2f}{r['wilcoxon_p']:>9.4f}"
                f"{r['n_improved']:>4d}/{r['n']}")
    lines.append("\n" + "=" * 100)
    lines.append("diff = model - baseline, in Dice. dz = Cohen's d for paired samples.")
    lines.append("p_holm = Holm-Bonferroni across the 3 comparisons within each field.")
    lines.append("'up' = subjects improving. wilcox = Wilcoxon signed-rank (exact).")
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/paired_ttest_subject_level.txt", "w").write(txt + "\n")

    # ------------------------------- figure --------------------------------- #
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), facecolor=SURFACE)
    fig.subplots_adjust(top=0.84, hspace=0.50, wspace=0.28)

    for col, field in enumerate(FIELDS):
        df = sm[field]
        n = len(df)

        # --- row 1: paired slopes, one line per subject ---------------------- #
        ax = axes[0, col]
        ax.set_facecolor(SURFACE)
        x = np.arange(len(MODELS))
        for subj in df.index:
            ax.plot(x, df.loc[subj, MODELS].values, color=MUTED, lw=1.0,
                    marker="o", ms=4.5, alpha=0.75, zorder=2,
                    label="subject" if subj == df.index[0] else None)
        ax.plot(x, df[MODELS].mean().values, color=BLUE, lw=2.0, marker="o",
                ms=8, zorder=3, label="mean", markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        for xi, m in zip(x, MODELS):
            ax.annotate(f"{df[m].mean():.3f}", (xi, df[m].mean()),
                        textcoords="offset points", xytext=(0, 13),
                        ha="center", fontsize=9, color=BLUE, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE,
                                  ec="none", alpha=0.85))
        ax.set_xticks(x); ax.set_xticklabels(MODELS, color=INK2)
        ax.set_xlim(-0.35, len(MODELS) - 0.65)
        ax.set_ylabel("subject mean Dice", color=INK2, fontsize=10)
        ax.set_title(f"{field}  (n = {n} subjects)", color=INK, fontsize=12,
                     fontweight="bold", pad=26, loc="left")
        if col == 0:
            handles, labels = ax.get_legend_handles_labels()

        # --- row 2: per-subject change, sorted, with mean and CI ------------- #
        ax = axes[1, col]
        ax.set_facecolor(SURFACE)
        d = (df["v3"] - df["pretrained"]).sort_values()
        ypos = np.arange(len(d))
        for yi, (subj, val) in zip(ypos, d.items()):
            c = BLUE if val > 0 else RED
            ax.plot([0, val], [yi, yi], color=c, lw=1.5, alpha=0.55, zorder=2)
            ax.plot(val, yi, "o", color=c, ms=8, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        r = next(r for r in rows if r["field"] == field
                 and r["baseline"] == "pretrained" and r["model"] == "v3")
        ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
        ax.axvspan(r["ci_lo"], r["ci_hi"], color=BLUE, alpha=0.10, zorder=0)
        ax.axvline(r["mean_diff"], color=BLUE, lw=2.0, ls="--", zorder=4)
        ax.set_yticks(ypos)
        ax.set_yticklabels([s.replace("sub-", "") for s in d.index], color=INK2)
        ax.set_ylim(-0.7, len(d) - 0.3)
        ax.set_xlabel("Dice change,  v3 - pretrained", color=INK2, fontsize=10)
        ax.set_ylabel("subject", color=INK2, fontsize=10)
        sig = "n.s." if r["p"] >= 0.05 else f"p = {r['p']:.3f}"
        ax.set_title(f"mean {r['mean_diff']:+.4f}   95% CI "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]   {sig}",
                     color=INK, fontsize=10.5, pad=10, loc="left")
        pad = max(abs(d).max(), abs(r["ci_hi"]), abs(r["ci_lo"])) * 1.35
        ax.set_xlim(-pad, pad)
        ax.annotate("worse", xy=(0.02, 0.03), xycoords="axes fraction",
                    color=RED, fontsize=9)
        ax.annotate("better", xy=(0.90, 0.03), xycoords="axes fraction",
                    color=BLUE, fontsize=9, ha="right")

    for ax in axes.ravel():
        ax.grid(axis="both", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle("Subject-level Dice: retrained models vs released Fetal-BET",
                 color=INK, fontsize=14, fontweight="bold", x=0.125, ha="left",
                 y=0.985)
    fig.text(0.125, 0.952,
             "y-scales differ between fields; low-field Dice is far lower "
             "throughout. Volumes averaged within subject before testing.",
             color=MUTED, fontsize=9, ha="left", va="top")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.125, 0.925),
               ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/paired_dice_by_field.{ext}", dpi=200,
                    facecolor=SURFACE, bbox_inches="tight")
    print(f"\nwrote {FIG}/paired_dice_by_field.pdf / .png")
    print(f"wrote {OUT}/paired_ttest_subject_level.csv / .txt")
    print(f"wrote {OUT}/subject_means_{{highfield,lowfield}}.csv")


if __name__ == "__main__":
    main()
