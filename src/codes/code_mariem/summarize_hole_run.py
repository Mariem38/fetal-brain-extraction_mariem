#!/usr/bin/env python
"""Aggregate the lambda=0 instrumentation run: per-size-bin frac_correct pooled
across every epoch, and the cutoff that follows from it."""
import re, sys
import numpy as np
import pandas as pd

LOG = sys.argv[1] if len(sys.argv) > 1 else \
    "/media/mial/HDD2TB/fetbet_train/v4_hole_lam0_ep100/log_train.txt"
BINS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33+"]

rows, epoch = [], None
stat = re.compile(r"epoch (\d+) hole stats: fire_rate=([\d.]+)\s+px_per_slice=([\d.]+)\s+"
                  r"frac_correct=([\d.]+)\s+leak_rate=([\d.]+)\s+penalty_value=([\d.]+)")
hist = re.compile(r"^\s*([\d+\-]+)vox\s+n=\s*(\d+)\s*\([\d. ]+%\)\s+px=\s*(\d+)\s+frac_correct=([\d.]+)")
stats_by_epoch = {}
for line in open(LOG):
    body = line.split("] ", 1)[-1]
    m = stat.search(body)
    if m:
        epoch = int(m.group(1))
        stats_by_epoch[epoch] = dict(zip(
            ["fire_rate", "px_per_slice", "frac_correct", "leak_rate", "penalty_value"],
            [float(m.group(i)) for i in range(2, 7)]))
        continue
    m = hist.match(body)
    if m and epoch is not None:
        rows.append({"epoch": epoch, "bin": m.group(1), "n": int(m.group(2)),
                     "px": int(m.group(3)), "fc": float(m.group(4))})

d = pd.DataFrame(rows)
d["correct"] = d.px * d.fc
n_ep = d.epoch.nunique()
print(f"parsed {n_ep} epochs, {len(d)} bin-rows, {d.px.sum():,} selected px total\n")

print("PER-SIZE-BIN, POOLED OVER ALL EPOCHS (px-weighted)")
print(f"  {'bin':>7} {'components':>11} {'px':>9} {'% px':>7} {'frac_correct':>13} {'95% CI':>18}")
agg = d.groupby("bin").agg(n=("n", "sum"), px=("px", "sum"), correct=("correct", "sum"))
agg["fc"] = agg.correct / agg.px
for b in BINS:
    r = agg.loc[b]
    se = np.sqrt(r.fc * (1 - r.fc) / r.px)          # binomial se on pooled pixels
    print(f"  {b:>7} {int(r.n):>11,} {int(r.px):>9,} {100*r.px/agg.px.sum():>6.1f}% "
          f"{r.fc:>13.4f}  [{r.fc-1.96*se:.4f}, {r.fc+1.96*se:.4f}]")

print("\nCUMULATIVE CEILING (keep components <= X vox)")
print(f"  {'ceiling':>9} {'px kept':>9} {'% px':>7} {'frac_correct':>13}")
cum_px = cum_c = 0
for b in BINS:
    cum_px += agg.loc[b, "px"]; cum_c += agg.loc[b, "correct"]
    lab = "all" if b == "33+" else "<=" + b.split("-")[-1]
    print(f"  {lab:>9} {int(cum_px):>9,} {100*cum_px/agg.px.sum():>6.1f}% {cum_c/cum_px:>13.4f}")

per_ep = (d[d["bin"] != "33+"].groupby("epoch")
          .apply(lambda g: g.correct.sum() / g.px.sum(), include_groups=False))
from scipy import stats as st
ci = st.t.ppf(.975, len(per_ep)-1) * per_ep.std(ddof=1) / np.sqrt(len(per_ep))
print(f"\n<=32 per-epoch: mean {per_ep.mean():.4f}  sd {per_ep.std(ddof=1):.4f}  "
      f"95% CI [{per_ep.mean()-ci:.4f}, {per_ep.mean()+ci:.4f}]  n={len(per_ep)}")
print(f"  one-sided t vs 0.5: p = {st.ttest_1samp(per_ep, 0.5, alternative='greater').pvalue:.2e}")

s = pd.DataFrame(stats_by_epoch).T
print(f"\nRUN-LEVEL MEANS: fire_rate {s.fire_rate.mean():.4f}  px_per_slice "
      f"{s.px_per_slice.mean():.2f}  leak_rate {s.leak_rate.mean():.4f}  "
      f"penalty_value {s.penalty_value.mean():.4f} (uncapped)")

frac_kept = agg.loc[BINS[:-1], "px"].sum() / agg.px.sum()
pen_capped = s.penalty_value.mean() * frac_kept
print(f"\nLAMBDA, from the loss ratio under --hole_max_abs 32:")
print(f"  penalty_value scales by the kept-pixel fraction {frac_kept:.3f} "
      f"-> ~{pen_capped:.4f}")
dice_ce = 0.00925
for rho in (0.02, 0.05, 0.10):
    print(f"    rho={rho:<5.2f}  lambda ~ {rho*dice_ce/pen_capped:.3e}")
print("  NOTE: loss-ratio only. The gradient ratio needs calibrate_hole_lambda.py.")
