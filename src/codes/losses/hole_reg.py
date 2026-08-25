"""Hole-filling regularizer for 2D fetal brain masks.

Prior: a fetal brain mask has no interior holes. The model occasionally predicts
small enclosed background pockets inside the brain. We penalise them with a
differentiable term whose *targets* are picked by a non-differentiable flood fill
(``scipy.ndimage.binary_fill_holes``) run under ``no_grad`` on detached
probabilities. Gradient flows only through the probabilities; the selector is a
constant 0/1 mask. Same pattern as hard-negative mining.

Sign: the penalty pushes probabilities UP on the selected voxels
(``(1 - p) * s``), the opposite of a false-positive suppression term.
"""

import numpy as np
import torch
from scipy import ndimage

CONN8 = ndimage.generate_binary_structure(2, 2)


def _holes_for_mask(m, max_hole_abs=200, max_hole_rel=0.05, close_iters=0,
                    min_depth_vox=2.0):
    """Enclosed-hole mask for one 2D hard mask ``m`` (bool HxW).

    Returns a bool array of the same shape holding the hole components that are
    small enough to be treated as speckle. Empty array (all False) when there is
    nothing to fill. Large enclosed regions are dropped: they may be
    ventricles / CSF that the GT legitimately excludes, so the cutoff is a
    MAXIMUM, not a minimum.

    With ``close_iters > 0`` the mask is first sealed by a binary closing so that
    pockets whose outer wall has a thin break are still found; the depth filter
    then drops pixels that sit less than ``min_depth_vox`` inside the blob.
    Note that in that mode the selection is NOT guaranteed to stay inside
    ``binary_fill_holes(m)`` -- sealing a broken wall is exactly the case where
    a genuine pocket lies outside the mask's own filled hull.
    """
    if m.sum() == 0:
        return np.zeros(m.shape, dtype=bool)

    if close_iters:
        sealed = ndimage.binary_closing(m, iterations=close_iters)
        holes = ndimage.binary_fill_holes(sealed) & ~m
        depth = ndimage.distance_transform_edt(
            ndimage.binary_fill_holes(m) | holes)
        holes &= (depth >= min_depth_vox)
    else:
        holes = ndimage.binary_fill_holes(m) & ~m

    if not holes.any():
        return np.zeros(m.shape, dtype=bool)

    lab, n = ndimage.label(holes, CONN8)
    sizes = ndimage.sum(holes, lab, range(1, n + 1))
    cutoff = min(max_hole_abs, max_hole_rel * m.sum())
    keep = [i + 1 for i in range(n) if sizes[i] <= cutoff]
    if not keep:
        return np.zeros(m.shape, dtype=bool)
    return np.isin(lab, keep)


@torch.no_grad()
def hole_selector_2d(prob, thr=0.5, max_hole_abs=200, max_hole_rel=0.05,
                     close_iters=0, min_depth_vox=2.0):
    """0/1 selector over the enclosed holes of ``prob > thr``, per 2D slice.

    ``prob``: (B, 1, H, W) foreground probability. Returns (B, 1, H, W) float on
    ``prob.device``. Non-differentiable by construction -- always call it on a
    detached tensor.
    """
    hard = (prob > thr).cpu().numpy()[:, 0]
    out = np.zeros(hard.shape, dtype=np.float32)
    for b, m in enumerate(hard):
        holes = _holes_for_mask(m, max_hole_abs=max_hole_abs,
                                max_hole_rel=max_hole_rel,
                                close_iters=close_iters,
                                min_depth_vox=min_depth_vox)
        if holes.any():
            out[b] = holes
    return torch.from_numpy(out).unsqueeze(1).to(prob.device)


def hole_penalty_with_selector(logits, **kw):
    """``(penalty, selector)`` -- lets the caller reuse the selector for stats."""
    prob = torch.sigmoid(logits)
    s = hole_selector_2d(prob.detach(), **kw)
    if s.sum() == 0:
        return prob.sum() * 0.0, s          # keeps graph, zero grad
    return ((1.0 - prob) * s).sum() / s.sum(), s


def hole_penalty(logits, **kw):
    """Mean ``1 - p`` over the selected hole voxels; 0 (with graph) if none."""
    return hole_penalty_with_selector(logits, **kw)[0]


def hole_lambda_at(epoch, lam, warmup_start, warmup_end):
    """0 before ``warmup_start``, linear ramp to ``lam`` over the warmup window,
    constant ``lam`` after. ``epoch`` is 1-based, matching the training log."""
    if lam <= 0.0 or epoch < warmup_start:
        return 0.0
    if epoch >= warmup_end or warmup_end <= warmup_start:
        return float(lam)
    return float(lam) * (epoch - warmup_start) / (warmup_end - warmup_start)


class HoleStats:
    """Per-epoch accumulator for the hole diagnostics (see PART A).

    Cheap, ``no_grad`` only, never touches the loss. ``update`` reuses the
    selector the loss already built, so the only extra work is the second
    hole pass needed by ``leak_rate``.
    """

    LEAK_CLOSE_ITERS = 3

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0            # steps on which the selector was evaluated
        self.fire_steps = 0       # ... of which the selector was non-empty
        self.slices = 0           # slices with a non-empty hard mask
        self.fire_slices = 0      # ... of which the selector was non-empty
        self.sel_px = 0.0         # total selected pixels
        self.correct_px = 0.0     # ... of which GT says brain
        self.leak_slices = 0      # holes only visible after closing(iters=3)
        self.penalty_sum = 0.0
        self.penalty_n = 0

    @torch.no_grad()
    def update(self, prob, label, selector, penalty_value=None,
               thr=0.5, max_hole_abs=200, max_hole_rel=0.05, close_iters=0,
               min_depth_vox=2.0):
        """``prob``/``label``/``selector``: (B, 1, H, W). ``selector`` must have
        been built from ``prob`` with the same kwargs."""
        sel = selector.detach().cpu().numpy()[:, 0].astype(bool)
        hard = (prob.detach() > thr).cpu().numpy()[:, 0]
        gt = (label.detach() > 0.5).cpu().numpy()
        gt = gt[:, 0] if gt.ndim == 4 else gt

        self.steps += 1
        if sel.any():
            self.fire_steps += 1

        kw = dict(max_hole_abs=max_hole_abs, max_hole_rel=max_hole_rel,
                  min_depth_vox=min_depth_vox)
        for b, m in enumerate(hard):
            if m.sum() == 0:
                continue
            self.slices += 1
            n_sel = int(sel[b].sum())
            if n_sel:
                self.fire_slices += 1
                self.sel_px += n_sel
                self.correct_px += float((sel[b] & gt[b]).sum())

            # leak: nothing enclosed as-is, but a closing seals a broken wall
            holes0 = sel[b] if close_iters == 0 else _holes_for_mask(
                m, close_iters=0, **kw)
            if holes0.any():
                continue
            holes3 = sel[b] if close_iters == self.LEAK_CLOSE_ITERS else \
                _holes_for_mask(m, close_iters=self.LEAK_CLOSE_ITERS, **kw)
            if holes3.any():
                self.leak_slices += 1

        if penalty_value is not None:
            self.penalty_sum += float(penalty_value)
            self.penalty_n += 1

    def summary(self):
        """Epoch aggregates, ready to hand to the logger. Empty dict if the
        selector never ran this epoch."""
        if self.steps == 0:
            return {}
        return {
            "hole/fire_rate": self.fire_steps / self.steps,
            "hole/px_per_slice": (self.sel_px / self.fire_slices
                                  if self.fire_slices else 0.0),
            "hole/frac_correct": (self.correct_px / self.sel_px
                                  if self.sel_px else float("nan")),
            "hole/leak_rate": (self.leak_slices / self.slices
                               if self.slices else 0.0),
            "hole/penalty_value": (self.penalty_sum / self.penalty_n
                                   if self.penalty_n else 0.0),
        }
