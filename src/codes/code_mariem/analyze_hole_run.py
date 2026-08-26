#!/usr/bin/env python
"""Analysis of the lambda=0 hole-instrumentation run (20 epochs from v3 ep100)."""
import os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RUN = sys.argv[1] if len(sys.argv) > 1 else "/media/mial/HDD2TB/fetbet_train/v4_hole_lam0_ep100"
FIG = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/mial/code_mariem/fetal-brain-extraction_mariem/src/figures"
OUT = "/home/mial/code_mariem/fetal-brain-extraction_mariem/src/results/stats"
BASELINE, KILL = 0.942063, 0.932063
BINS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33+"]

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE, RED = "#e1e0d9", "#c3c2b7", "#2a78d6", "#e34948"
# ordinal ramp, light -> dark with increasing TE (never lighter than step 250)
TE_RAMP = ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]

d = pd.read_csv(f"{RUN}/val_metrics.csv")
g = d.dice_by_group.map(lambda s: {k: float(v) for k, v in
                                   (p.split("=") for p in s.split(";"))}).apply(pd.Series)
echoes = sorted(g.columns)
TE = {"echo0": 90, "echo1": 180, "echo2": 250, "echo3": 300,
      "echo4": 400, "echo5": 500, "echo6": 600}

rows, epoch = [], None
sre = re.compile(r"epoch (\d+) hole stats:")
hre = re.compile(r"^\s*([\d+\-]+)vox\s+n=\s*(\d+)\s*\([\d. ]+%\)\s+px=\s*(\d+)\s+frac_correct=([\d.]+)")
for line in open(f"{RUN}/log_train.txt"):
    b = line.split("] ", 1)[-1]
    m = sre.search(b)
    if m: epoch = int(m.group(1)); continue
    m = hre.match(b)
    if m and epoch: rows.append({"epoch": epoch, "bin": m.group(1),
                                 "n": int(m.group(2)), "px": int(m.group(3)),
                                 "fc": float(m.group(4))})
h = pd.DataFrame(rows); h["correct"] = h.px * h.fc
agg = h.groupby("bin").agg(n=("n", "sum"), px=("px", "sum"), correct=("correct", "sum"))
agg["fc"] = agg.correct / agg.px
agg["se"] = np.sqrt(agg.fc * (1 - agg.fc) / agg.px)

fig, ax = plt.subplots(2, 2, figsize=(12, 8.8), facecolor=SURFACE)
fig.subplots_adjust(top=0.86, hspace=0.42, wspace=0.26)

# 1 -- pooled val dice vs the two reference lines
a = ax[0, 0]
a.axhspan(KILL, BASELINE, color=BLUE, alpha=0.07, zorder=0)
a.axhline(BASELINE, color=AXIS, lw=1.4, ls="-", zorder=1)
a.axhline(KILL, color=RED, lw=1.4, ls="--", zorder=1)
a.plot(d.epoch, d.mean_dice, color=BLUE, lw=2, marker="o", ms=6,
       markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
bad = d[d.mean_dice < KILL]
a.plot(bad.epoch, bad.mean_dice, "o", color=RED, ms=10, zorder=4,
       markeredgecolor=SURFACE, markeredgewidth=1.5)
a.annotate("ep100 baseline 0.9421", (20.4, BASELINE), fontsize=8, color=INK2,
           va="bottom", xytext=(0, 3), textcoords="offset points")
a.annotate("kill line 0.9321", (20.4, KILL), fontsize=8, color=RED,
           va="bottom", xytext=(0, 3), textcoords="offset points")
a.annotate("ep13", (bad.epoch.iloc[0], bad.mean_dice.iloc[0]), fontsize=8.5,
           color=RED, xytext=(0, -14), textcoords="offset points", ha="center")
a.set_xlim(0.3, 24.5); a.set_xlabel("epoch", color=INK2); a.set_ylabel("pooled val Dice", color=INK2)
a.set_title("Validation held steady; one excursion, recovered", color=INK,
            fontsize=11.5, fontweight="bold", loc="left", pad=10)

# 2 -- per-echo trajectories, ordinal ramp by TE
a = ax[0, 1]
for c, col in zip(echoes, TE_RAMP):
    a.plot(d.epoch, g[c], color=col, lw=1.8, zorder=2, label=f"{TE[c]} ms")
a.legend(ncol=4, loc="lower left", fontsize=7.5, frameon=True, facecolor=SURFACE,
         edgecolor=GRID, framealpha=0.92, labelcolor=INK2, columnspacing=1.1,
         handlelength=1.4, title="TE (light to dark)", title_fontsize=7.5)
a.get_legend().get_title().set_color(MUTED)
a.set_ylim(0.911, 0.9575)
a.set_xlim(0.5, 20.5); a.set_xlabel("epoch", color=INK2)
a.set_ylabel("val Dice by echo", color=INK2)
sp = g.max(axis=1) - g.min(axis=1)
r = stats.linregress(d.epoch, sp)
a.set_title(f"Long-TE gap closed: spread {sp.iloc[0]:.4f} -> {sp.iloc[-1]:.4f}  "
            f"(p = {r.pvalue:.4f})", color=INK, fontsize=11.5, fontweight="bold",
            loc="left", pad=10)

# 3 -- frac_correct by size bin, pooled over the run
a = ax[1, 0]
x = np.arange(len(BINS))
vals = [agg.loc[b, "fc"] for b in BINS]
errs = [1.96 * agg.loc[b, "se"] for b in BINS]
cols = [BLUE] * 6 + [RED]
a.bar(x, vals, yerr=errs, color=cols, width=0.66, zorder=3,
      error_kw=dict(ecolor=INK2, lw=1.2, capsize=3))
a.axhline(0.5, color=RED, lw=1.4, ls="--", zorder=4)
a.annotate("noise floor 0.5", (len(BINS) - 0.4, 0.5), fontsize=8, color=RED,
           va="bottom", ha="right", xytext=(0, 4), textcoords="offset points")
for xi, v, b in zip(x, vals, BINS):
    a.annotate(f"{v:.3f}", (xi, v), xytext=(0, 9), textcoords="offset points",
               ha="center", fontsize=8.5, color=INK2)
    a.annotate(f"{100*agg.loc[b,'px']/agg.px.sum():.0f}% px", (xi, 0.02),
               ha="center", fontsize=7.5, color=SURFACE, zorder=5)
a.set_xticks(x); a.set_xticklabels(BINS, color=INK2)
a.set_ylim(0, 0.85); a.set_xlabel("hole size (voxels)", color=INK2)
a.set_ylabel("frac_correct", color=INK2)
a.set_title("Flat at ~0.66 to 32 vox, then collapses", color=INK,
            fontsize=11.5, fontweight="bold", loc="left", pad=10)

# 4 -- cumulative ceiling
a = ax[1, 1]
cp = cc = 0; cx, cy, cpx = [], [], []
for i, b in enumerate(BINS):
    cp += agg.loc[b, "px"]; cc += agg.loc[b, "correct"]
    cx.append(i); cy.append(cc / cp); cpx.append(100 * cp / agg.px.sum())
a.plot(cx[:-1], cy[:-1], color=BLUE, lw=2, marker="o", ms=7,
       markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
a.plot(cx[-2:], cy[-2:], color=RED, lw=2, ls=":", marker="o", ms=7,
       markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
a.axhline(0.5, color=RED, lw=1.4, ls="--", zorder=1)
for xi, yi, p in zip(cx, cy, cpx):
    a.annotate(f"{p:.0f}%", (xi, yi), xytext=(0, -16), textcoords="offset points",
               ha="center", fontsize=7.5, color=MUTED)
a.annotate("keep <=32:\n0.661 at 50% of px", (5, cy[5]), xytext=(-14, 26),
           textcoords="offset points", fontsize=9, color=BLUE, fontweight="bold",
           ha="center")
a.set_xticks(cx); a.set_xticklabels(["<=1", "<=2", "<=4", "<=8", "<=16", "<=32", "all"],
                                    color=INK2)
a.set_ylim(0.3, 0.85); a.set_xlabel("size ceiling  (labels: % of selected px kept)",
                                    color=INK2)
a.set_ylabel("frac_correct of kept px", color=INK2)
a.set_title("The cap is the only decision that matters", color=INK,
            fontsize=11.5, fontweight="bold", loc="left", pad=10)

for a in ax.ravel():
    a.set_facecolor(SURFACE); a.grid(color=GRID, lw=0.8, zorder=0); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(AXIS)
    a.tick_params(colors=MUTED, labelsize=8.5)

fig.suptitle("lambda = 0 hole instrumentation: 20 epochs from v3 epoch 100",
             color=INK, fontsize=14.5, fontweight="bold", x=0.09, ha="left", y=0.975)
fig.text(0.09, 0.925, "264,385 selected pixels over 20 epochs. The loss was never "
         "touched (lambda = 0) - this run only measures the selector.",
         color=MUTED, fontsize=9.5, ha="left", va="top")
os.makedirs(FIG, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/hole_instrumentation_lam0.{ext}", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
os.makedirs(OUT, exist_ok=True)
agg.to_csv(f"{OUT}/hole_size_bins_pooled.csv")
print(f"wrote {FIG}/hole_instrumentation_lam0.pdf/.png and {OUT}/hole_size_bins_pooled.csv")
