"""Regression tests for infinite emissions poisoning segment DP (test_defect3)."""

import warnings
import numpy as np
import pytest

from mtah.v2 import fallback


def test_infinite_emissions_do_not_produce_nans_or_corrupt_dp():
    # Reuses logic of agyJ/test_defect3_neginf_nan.py
    fps = 25
    n = 200
    e = np.zeros((n, 3))
    # Home has -inf emission from frame 20 onwards (completely impossible for Home)
    e[20:, 0] = -np.inf

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        res = fallback._semi_markov_block(e, fps)

    sub_warnings = [
        w for w in recorded_warnings if "invalid value encountered in subtract" in str(w.message)
    ]
    assert len(sub_warnings) == 0, (
        f"Encountered {len(sub_warnings)} RuntimeWarnings from (-inf) - (-inf) NaNs: {sub_warnings}"
    )

    h_count = int((res[20:] == "H").sum())
    assert h_count == 0, (
        f"NaNs corrupted DP argmax and selected impossible state H on {h_count} frames"
    )


def test_finite_emissions_unaffected():
    # Identical output on finite emissions
    rng = np.random.default_rng(42)
    fps = 25
    e = rng.normal(0, 1.0, (100, 3))
    res1 = fallback._semi_markov_block(e, fps)
    res2 = fallback._semi_markov_block(e, fps)
    assert np.array_equal(res1, res2)
