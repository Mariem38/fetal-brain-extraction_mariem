#!/usr/bin/env python
"""Calibrate --hole_lambda against the gradient scale of the main loss.

Runs the v3 epoch-70 checkpoint over ~200 batches of the TRAIN loader and, per
batch, backprops dice_ce and hole_penalty separately to compare the global
gradient L2 norm each one produces. lambda is then chosen so the hole term
contributes a target fraction rho of the main loss's gradient.

Read-only: eval mode, no optimizer step, weights never updated. The optimizer
exists only to zero grads between the two backward passes.

usage:  python calibrate_hole_lambda.py [n_batches] [out_csv] [checkpoint] [max_abs] [eval|train]
"""

import os
import sys
import time

import numpy as np
import torch
from scipy import ndimage

REPO = os.path.expanduser("~/code_mariem/fetal-brain-extraction_mariem")
CODES = os.path.join(REPO, "src", "codes")
CFG = os.path.join(REPO, "src", "configs", "train_configs.yml")
CKPT = "/media/mial/HDD2TB/fetbet_train/baseline_sim_v3/AttUNet_checkpoint-70.pth"

# same kwargs the training loop uses
THR, MAX_ABS, MAX_REL, CLOSE_ITERS = 0.5, 200, 0.05, 0
RHOS = (0.02, 0.05, 0.10)

CONN4 = ndimage.generate_binary_structure(2, 1)
CONN8 = ndimage.generate_binary_structure(2, 2)


# --------------------------------------------------------------------------- #
# fast selector: background components that do not touch the border are holes.
# Identical output to binary_fill_holes(m) & ~m, ~3.7x cheaper.
# --------------------------------------------------------------------------- #

def fast_holes(m):
    """Enclosed background of one 2D bool mask, before size filtering."""
    lab, n = ndimage.label(~m, CONN4)
    if n == 0:
        return np.zeros(m.shape, dtype=bool)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False                       # foreground label
    keep[border[border > 0]] = False      # reaches the image edge -> not a hole
    return keep[lab]


def fast_hole_selector_2d(prob, thr=THR, max_hole_abs=MAX_ABS,
                          max_hole_rel=MAX_REL, close_iters=CLOSE_ITERS):
    """Drop-in for losses.hole_reg.hole_selector_2d at close_iters=0."""
    if close_iters:
        raise ValueError("fast selector only implements close_iters=0")
    hard = (prob > thr).cpu().numpy()[:, 0]
    out = np.zeros(hard.shape, dtype=np.float32)
    for b, m in enumerate(hard):
        if m.sum() == 0:
            continue
        holes = fast_holes(m)
        if not holes.any():
            continue
        lab, n = ndimage.label(holes, CONN8)
        sizes = ndimage.sum(holes, lab, range(1, n + 1))
        cutoff = min(max_hole_abs, max_hole_rel * m.sum())
        keep = [i + 1 for i in range(n) if sizes[i] <= cutoff]
        if keep:
            out[b] = np.isin(lab, keep)
    return torch.from_numpy(out).unsqueeze(1).to(prob.device)


def hole_penalty_from(fg_logit, selector):
    """Mean (1 - p) over the selected voxels; 0-with-graph when none fired."""
    prob = torch.sigmoid(fg_logit)
    if selector.sum() == 0:
        return prob.sum() * 0.0
    return ((1.0 - prob) * selector).sum() / selector.sum()


def grad_norm(model):
    """Global L2 norm over every parameter gradient."""
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return tot ** 0.5


def histogram(values, bins=10, width=44):
    """Text histogram over [0, 1]."""
    if not len(values):
        return ["  (no firing batches)"]
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    top = max(counts.max(), 1)
    return [f"  [{edges[i]:.1f},{edges[i+1]:.1f})  {'#' * int(width * c / top):<{width}} {c}"
            for i, c in enumerate(counts)]


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "calibrate_hole_lambda.csv"
    ckpt = sys.argv[3] if len(sys.argv) > 3 else CKPT
    max_abs = int(sys.argv[4]) if len(sys.argv) > 4 else MAX_ABS
    mode = sys.argv[5] if len(sys.argv) > 5 else "eval"
    assert mode in ("eval", "train")
    # bound explicitly: fast_hole_selector_2d's defaults were captured at def time
    kw = dict(thr=THR, max_hole_abs=max_abs, max_hole_rel=MAX_REL,
              close_iters=CLOSE_ITERS)

    sys.path.insert(0, CODES)
    import yaml
    from monai.losses import DiceCELoss
    from data_generator_monai import FetalTrainData
    from model_zoo import get_network
    from losses.hole_reg import hole_selector_2d

    print("command:", " ".join([sys.executable] + sys.argv), flush=True)
    print(f"checkpoint : {ckpt}\nconfig     : {CFG}\nbatches    : {n_batches}\n"
          f"out        : {os.path.abspath(out_csv)}\n"
          f"hole kwargs: {kw}", flush=True)

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = yaml.safe_load(open(CFG))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_network(cfg).to(device)
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    # eval() = the deployed model. train() = the regime the regularizer actually
    # runs in (dropout 0.15 on), which fires far more often.
    getattr(model, mode)()
    print(f"loaded {len(sd)} tensors onto {device}, {mode} mode", flush=True)

    # identical to the training loss
    dice_ce = DiceCELoss(include_background=cfg["include_background"], to_onehot_y=True,
                         softmax=True, squared_pred=True, batch=True,
                         smooth_nr=0.00001, smooth_dr=0.00001,
                         lambda_dice=0.6, lambda_ce=0.4)
    # never stepped; only used to zero grads between the two backward passes
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    train_dl, _ = FetalTrainData(cfg).load_data()

    rows = []
    slice_px = []          # selected pixels per firing slice
    checked = False
    t0 = time.time()

    for i, batch in enumerate(train_dl):
        if i >= n_batches:
            break
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        fg_logit = logits[:, 1:2] - logits[:, 0:1]     # sigmoid(z1-z0) == softmax(z)[1]

        with torch.no_grad():
            prob = torch.sigmoid(fg_logit)
            sel = fast_hole_selector_2d(prob, **kw)
            if not checked:                            # the fast path must agree
                ref = hole_selector_2d(prob, **kw)
                same = bool(torch.equal(sel, ref))
                print(f"fast selector == hole_selector_2d on batch 0: {same}", flush=True)
                if not same:
                    sys.exit("fast selector disagrees with the training selector")
                checked = True

            gt = (y > 0.5)
            n_px = int(sel.sum().item())
            fired = n_px > 0
            frac_correct = (float((sel.bool() & gt).sum().item()) / n_px
                            if fired else float("nan"))
            per_slice = sel.sum(dim=(1, 2, 3)).cpu().numpy()
            slice_px.extend(int(v) for v in per_slice if v > 0)

        base = dice_ce(logits, y)
        opt.zero_grad(set_to_none=True)
        base.backward(retain_graph=True)
        g_base = grad_norm(model)

        pen = hole_penalty_from(fg_logit, sel)
        opt.zero_grad(set_to_none=True)
        pen.backward()
        g_hole = grad_norm(model)

        rows.append((i, int(fired), n_px, int((per_slice > 0).sum()),
                     float(pen.item()), frac_correct, g_base, g_hole,
                     g_base / g_hole if g_hole > 0 else float("nan")))

        if (i + 1) % 25 == 0:
            f = np.mean([r[1] for r in rows])
            print(f"  {i+1}/{n_batches}  fire_rate={f:.3f}  "
                  f"{(time.time()-t0)/(i+1):.2f}s/batch", flush=True)

    opt.zero_grad(set_to_none=True)

    hdr = ("batch,fired,n_selected_px,n_firing_slices,penalty_value,"
           "frac_correct,g_base,g_hole,ratio")
    with open(out_csv, "w") as f:
        f.write(hdr + "\n")
        for r in rows:
            f.write("%d,%d,%d,%d,%.8f,%.8f,%.8e,%.8e,%.8e\n" % r)

    # ----------------------------------------------------------------- report #
    n = len(rows)
    fire = [r for r in rows if r[1]]
    fire_rate = len(fire) / n if n else 0.0

    print("\n" + "=" * 72)
    print(f"{n} batches, fire_rate = {fire_rate:.4f}  ({len(fire)} firing)")
    print("=" * 72)

    if not fire:
        print("\nno batch fired -- nothing to calibrate.")
    else:
        gb = np.array([r[6] for r in fire])
        gh = np.array([r[7] for r in fire])
        ratio = np.array([r[8] for r in fire])
        print("\nover firing batches only:")
        print(f"  {'':14s}{'median':>14s}{'p90':>14s}")
        for name, v in (("g_base", gb), ("g_hole", gh), ("g_base/g_hole", ratio)):
            print(f"  {name:14s}{pct(v, 50):14.6e}{pct(v, 90):14.6e}")

        base_med, hole_med = pct(gb, 50), pct(gh, 50)
        print(f"\nsuggested lambda = rho * median(g_base)/median(g_hole)"
              f"  [= rho * {base_med/hole_med:.4g}]")
        for rho in RHOS:
            print(f"  rho={rho:<5.2f}  lambda = {rho * base_med / hole_med:.6g}")

        fc = np.array([r[5] for r in fire], dtype=float)
        fc = fc[~np.isnan(fc)]
        print(f"\nfrac_correct over firing batches: mean = {fc.mean():.4f}  "
              f"median = {np.median(fc):.4f}")
        for line in histogram(fc):
            print(line)

        print(f"\nmedian selected px per firing slice: "
              f"{np.median(slice_px):.1f}  (n={len(slice_px)} firing slices)")
        print(f"median penalty_value: {np.median([r[4] for r in fire]):.6f}")

    if fire_rate < 0.02:
        print("\n" + "!" * 72)
        print("!! WARNING  fire_rate = %.4f < 0.02" % fire_rate)
        print("!! The selector almost never fires on this data distribution.")
        print("!! The hole term will be NEAR-INERT during training: it will")
        print("!! contribute nothing on >98%% of steps, and any lambda derived")
        print("!! from the medians above rests on a tiny sample. Re-check the")
        print("!! threshold and size cutoffs before spending GPU time on it.")
        print("!" * 72)

    print(f"\nwrote {n} rows to {os.path.abspath(out_csv)}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
