"""Regression tests for boundary refinement defects (test_defect1 and test_defect5)."""

import numpy as np
import pytest

from mtah.v2 import fallback


def test_refine_considers_frame_hi():
    # Reuses logic of agyJ/test_defect1_refine_offbyone.py:
    # Frame hi emission must affect refinement, and frame hi must not be forced into state b.
    e = np.zeros((100, 3))
    cut = 50
    lo, hi = 48, 52

    # Case A: Frame hi has strong evidence for state a (0)
    e[hi, 0] = 500.0
    res_a = fallback._refine(e, cut, 0, 2, lo, hi)

    # Case B: Frame hi has strong evidence for state b (2)
    e[hi, 0] = 0.0
    e[hi, 2] = 500.0
    res_b = fallback._refine(e, cut, 0, 2, lo, hi)

    assert res_a != res_b, "Refinement should differ when frame hi evidence changes"
    assert res_a == hi + 1, f"Frame hi should be assigned to state a when favored (got cut={res_a})"

    # When state a is favored everywhere, frame hi must be included in state a
    e[:, 0] = 1000.0
    e[:, 2] = 0.0
    res_max = fallback._refine(e, cut, 0, 2, lo, hi)
    assert res_max > hi, f"Frame hi was forced into state b: res_max={res_max} <= hi={hi}"


def test_refine_flat_evidence_tiebreak():
    # Reuses logic of agyJ/test_defect5_refine_flat_bias.py:
    # Under flat evidence, ties must resolve deterministically to the DP cut without shift.
    e = np.zeros((100, 3))
    cuts = [20, 35, 50, 70]
    for c in cuts:
        lo = c - fallback.CELL // 2
        hi = c + fallback.CELL // 2
        res = fallback._refine(e, c, 0, 1, lo, hi)
        shift = res - c
        assert shift == 0, f"Expected 0 shift on flat evidence for cut={c}, got {shift} (res={res})"
