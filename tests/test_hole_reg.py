"""Unit tests for the hole-filling regularizer.

Runs under pytest, or standalone: ``python tests/test_hole_reg.py``.
"""

import os
import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "codes"))

from losses.hole_reg import hole_penalty, hole_selector_2d  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #

def _disc(size, radius, centre=None):
    centre = centre or (size // 2, size // 2)
    yy, xx = np.ogrid[:size, :size]
    return ((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) <= radius ** 2


def disc_with_hole(size=128, radius=30, hole=3):
    """Filled disc with a ``hole x hole`` square punched out of the centre."""
    m = _disc(size, radius)
    c = size // 2
    lo, hi = c - hole // 2, c - hole // 2 + hole
    m[lo:hi, lo:hi] = False
    holes = np.zeros_like(m)
    holes[lo:hi, lo:hi] = True
    return m, holes


def c_shape(size=128, radius=40, thickness=12, mouth=24):
    """Annulus with a wide opening cut to the right -- the concavity reaches the
    image border, so it is NOT enclosed and must never be filled. The mouth is
    far wider than a closing with iterations=3 can bridge (~6 px)."""
    m = _disc(size, radius) & ~_disc(size, radius - thickness)
    c = size // 2
    m[c - mouth // 2:c + mouth // 2, c:] = False
    return m



def leaky_pocket(size=160, radius=50, pocket=8, gap=2):
    """Disc with a small pocket whose wall is broken by a thin channel running
    all the way out. ``close_iters=0`` sees nothing (the flood fill leaks);
    ``close_iters=3`` seals the channel and finds the pocket."""
    m = _disc(size, radius).copy()
    c = size // 2
    lo = c - pocket // 2
    m[lo:lo + pocket, lo:lo + pocket] = False
    m[c - gap // 2:c - gap // 2 + gap, lo + pocket:] = False
    return m

def _selector(mask, **kw):
    """Selector for one hard mask, as a bool HxW array."""
    prob = torch.from_numpy(np.where(mask, 0.95, 0.05).astype(np.float32))
    prob = prob[None, None]
    return hole_selector_2d(prob, **kw).numpy()[0, 0].astype(bool)


def all_cases():
    """(name, hard mask) for every geometry the guarantee test sweeps."""
    return [
        ("disc_small_hole", disc_with_hole(128, 30, 3)[0]),
        ("disc_big_hole", disc_with_hole(256, 100, 60)[0]),
        ("c_shape", c_shape()),
        ("empty", np.zeros((64, 64), dtype=bool)),
        ("all_ones", np.ones((64, 64), dtype=bool)),
        ("multi_hole", _multi_hole()),
    ]


def _multi_hole():
    """Disc holding one small (fillable) and one large (kept) hole."""
    m = _disc(256, 100)
    m[60:63, 60:63] = False            # 9 px  -> selected
    m[140:200, 140:200] = False        # 3600 px -> above the cutoff
    return m


# --------------------------------------------------------------------------- #
# 1. small enclosed hole -> fires on exactly those pixels                      #
# --------------------------------------------------------------------------- #

def test_small_hole_selects_exactly_the_hole():
    mask, holes = disc_with_hole(128, 30, hole=3)
    sel = _selector(mask)
    assert sel.sum() == 9, sel.sum()
    assert np.array_equal(sel, holes)


# --------------------------------------------------------------------------- #
# 2. hole above max_hole_abs -> nothing selected                               #
# --------------------------------------------------------------------------- #

def test_large_hole_is_left_alone():
    mask, _ = disc_with_hole(256, 100, hole=60)
    sel = _selector(mask, max_hole_abs=200)
    assert sel.sum() == 0, sel.sum()


def test_multi_hole_keeps_only_the_small_one():
    sel = _selector(_multi_hole())
    assert sel.sum() == 9, sel.sum()
    assert sel[60:63, 60:63].all()


# --------------------------------------------------------------------------- #
# 3. concavity open to the border -> never filled                              #
# --------------------------------------------------------------------------- #

def test_c_shape_is_not_filled():
    sel = _selector(c_shape(), close_iters=0)
    assert sel.sum() == 0, sel.sum()


# --------------------------------------------------------------------------- #
# 4. degenerate slices                                                         #
# --------------------------------------------------------------------------- #

def test_empty_and_full_slices():
    for mask in (np.zeros((64, 64), dtype=bool), np.ones((64, 64), dtype=bool)):
        for close_iters in (0, 3):
            sel = _selector(mask, close_iters=close_iters)
            assert sel.sum() == 0
            assert np.isfinite(sel).all()


def test_batch_mixes_empty_full_and_holed_without_nan():
    mask, _ = disc_with_hole(64, 20, hole=3)
    batch = np.stack([np.zeros((64, 64), dtype=bool),
                      np.ones((64, 64), dtype=bool),
                      mask,
                      _disc(64, 20)])                    # no holes at all
    prob = torch.from_numpy(np.where(batch, 0.95, 0.05).astype(np.float32))
    prob = prob[:, None]
    sel = hole_selector_2d(prob)
    assert sel.shape == prob.shape
    assert torch.isfinite(sel).all()
    assert sel[0].sum() == 0 and sel[1].sum() == 0 and sel[3].sum() == 0
    assert sel[2].sum() == 9

    logits = torch.logit(prob.clamp(1e-4, 1 - 1e-4)).requires_grad_(True)
    loss = hole_penalty(logits)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

    # a batch with no holes anywhere still gives a finite, zero, backwardable loss
    flat = torch.full((2, 1, 32, 32), 3.0, requires_grad=True)
    zero = hole_penalty(flat)
    assert zero.item() == 0.0
    zero.backward()
    assert torch.count_nonzero(flat.grad) == 0


# --------------------------------------------------------------------------- #
# 5. guarantee: the term can never grow the mask past its own perimeter        #
# --------------------------------------------------------------------------- #

def test_selector_never_grows_past_the_filled_hull():
    for close_iters in (0, 3):
        for name, mask in all_cases():
            sel = _selector(mask, close_iters=close_iters)
            grown = mask | sel
            hull = ndimage.binary_fill_holes(mask)
            assert not (grown & ~hull).any(), (name, close_iters,
                                               int((grown & ~hull).sum()))


# --------------------------------------------------------------------------- #
# 5b. the one place the hull bound does not hold, pinned on purpose            #
# --------------------------------------------------------------------------- #

def test_close_iters_3_reaches_past_the_hull_on_a_broken_wall():
    """Documented limit of the guarantee above. When the outer wall is broken,
    the pocket is by definition outside ``binary_fill_holes(hard)`` -- sealing it
    is exactly what ``close_iters > 0`` is for, so at close_iters=3 the selector
    can and does step past the mask's own filled hull. ``close_iters=0`` (the
    default) never does."""
    mask = leaky_pocket()
    hull = ndimage.binary_fill_holes(mask)

    assert _selector(mask, close_iters=0).sum() == 0

    sel3 = _selector(mask, close_iters=3)
    assert sel3.sum() > 0
    assert (sel3 & ~hull).any()


# --------------------------------------------------------------------------- #
# 6. gradient sign: selected voxels are pushed UP                              #
# --------------------------------------------------------------------------- #

def test_gradient_pushes_selected_probabilities_up():
    mask, holes = disc_with_hole(128, 30, hole=3)
    logits = np.full(mask.shape, -4.0, dtype=np.float32)
    logits[mask] = 4.0
    logits[holes] = float(np.log(0.2 / 0.8))          # prob 0.2 on the hole
    logits = torch.from_numpy(logits)[None, None].requires_grad_(True)

    prob = torch.sigmoid(logits)
    assert np.allclose(prob.detach().numpy()[0, 0][holes], 0.2, atol=1e-6)

    loss = hole_penalty(logits)
    assert abs(loss.item() - 0.8) < 1e-5              # mean of (1 - 0.2)
    loss.backward()

    g = logits.grad[0, 0].numpy()
    # loss decreases as the logit rises there -> negative gradient
    assert (g[holes] < 0).all(), g[holes]
    assert np.count_nonzero(g[~holes]) == 0

    # and a step of gradient descent really does raise those probabilities
    with torch.no_grad():
        stepped = logits - 10.0 * logits.grad
    assert (torch.sigmoid(stepped)[0, 0].numpy()[holes] > 0.2).all()


# --------------------------------------------------------------------------- #
# 7. no gradient flows through the scipy path                                  #
# --------------------------------------------------------------------------- #

def test_selector_carries_no_gradient():
    mask, _ = disc_with_hole(128, 30, hole=3)
    prob = torch.from_numpy(np.where(mask, 0.95, 0.05).astype(np.float32))
    prob = prob[None, None].requires_grad_(True)

    sel = hole_selector_2d(prob.detach())
    assert not sel.requires_grad
    assert sel.grad_fn is None

    # the selector is a plain constant: values are exactly 0.0 / 1.0
    assert set(np.unique(sel.numpy()).tolist()) <= {0.0, 1.0}


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
