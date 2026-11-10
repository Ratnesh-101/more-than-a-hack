"""Regression tests for pre-kickoff warmup touches bleeding into post-kickoff emissions (test_defect6)."""

import numpy as np
import pytest

from mtah.v2 import fallback, touch, tracking


def _make_touch_trk(n=300, blocks=None, kickoffs=None):
    """Build a synthetic trk dict."""
    if blocks is None:
        blocks = [(1, 0, n - 1)]
    px = np.zeros((n, 2))
    py = np.zeros((n, 2))
    ps = np.full((n, 2), 2.0, dtype=np.float32)
    team = np.array([0, 1], dtype=np.int8)
    bx = np.full(n, 50.0, dtype=np.float32)
    by = np.full(n, 30.0, dtype=np.float32)
    trk = tracking.build(px, py, ps, team, bx, by, (105.0, 68.0), blocks, kickoffs=kickoffs)
    return trk


def test_warmup_touch_ignored_after_kickoff_fallback(monkeypatch):
    # Reuses logic of agyJ/test_defect6_prekickoff_bleed.py
    n = 300
    kick = 100
    blocks = [(2, 0, n - 1)]
    kickoffs = {2: kick}

    trk = _make_touch_trk(n=n, blocks=blocks, kickoffs=kickoffs)

    # Away touches during pre-kickoff warmup at frame 95; Home kicks off at frame 100
    t_h = np.zeros(n, bool)
    t_a = np.zeros(n, bool)
    t_a[95] = True
    t_h[100] = True
    kick_mask = np.zeros(n, bool)
    kick_mask[100] = True
    still = np.zeros(n, bool)

    monkeypatch.setattr(fallback, "_touches", lambda t: (t_h.copy(), t_a.copy(), kick_mask.copy(), still.copy()))
    e_dirty = fallback.emissions(trk)

    # Clean scenario: no pre-kickoff touch by Away
    t_a_clean = np.zeros(n, bool)
    monkeypatch.setattr(fallback, "_touches", lambda t: (t_h.copy(), t_a_clean.copy(), kick_mask.copy(), still.copy()))
    e_clean = fallback.emissions(trk)

    # Post-kickoff emissions at and after frame 100 must not be contaminated by Away's warmup touch
    assert np.allclose(e_dirty[kick:], e_clean[kick:]), (
        f"Warm-up touch at frame 95 leaked into post-kickoff emissions at frame {kick}"
    )


def test_warmup_touch_ignored_after_kickoff_touch_module(monkeypatch):
    # Test touch.team_logodds directly
    n = 300
    kick = 100
    blocks = [(2, 0, n - 1)]
    kickoffs = {2: kick}

    trk = _make_touch_trk(n=n, blocks=blocks, kickoffs=kickoffs)

    t_h = np.zeros(n, bool)
    t_a = np.zeros(n, bool)
    t_a[95] = True
    t_h[100] = True

    monkeypatch.setattr(touch, "_touches", lambda t: (t_h.copy(), t_a.copy()))
    r_dirty = touch.team_logodds(trk)

    t_a_clean = np.zeros(n, bool)
    monkeypatch.setattr(touch, "_touches", lambda t: (t_h.copy(), t_a_clean.copy()))
    r_clean = touch.team_logodds(trk)

    assert np.allclose(r_dirty[kick:], r_clean[kick:]), (
        f"Warm-up touch leaked into post-kickoff team_logodds at frame {kick}"
    )


def test_warmup_touch_noop_when_kickoff_before_block():
    # Kickoff before block (like Half 1) must be an exact no-op
    n = 300
    blocks = [(1, 50, n - 1)]
    kickoffs = {1: 20}  # kickoff at frame 20 <= lo (50)

    trk_with = _make_touch_trk(n=n, blocks=blocks, kickoffs=kickoffs)
    trk_none = _make_touch_trk(n=n, blocks=blocks, kickoffs=None)

    e_with = fallback.emissions(trk_with)
    e_none = fallback.emissions(trk_none)
    assert np.array_equal(e_with, e_none), "Kickoff <= lo should be a no-op"
