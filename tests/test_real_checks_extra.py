"""Unit tests for the label-free diagnostic helper functions in analysis/real_checks.py.

Tests:
  (1) second_differences_1d and ball_roughness_stats on synthetic ball arrays.
  (2) kick_enrichment on small synthetic Trk dictionaries.
  (3) team_tag_to_side, check_half_kickoff_alternation, and check_goal_kickoffs
      on synthetic Smart Tagging events and label arrays.
"""

import numpy as np
import pytest

from analysis import real_checks
from mtah.v2 import touch


def test_second_differences_constant_and_linear():
    # Constant: d2 should be exactly 0
    c = np.full(20, 5.0)
    d2 = real_checks.second_differences_1d(c)
    assert len(d2) == 18
    np.testing.assert_allclose(d2, 0.0)

    # Linear: x[t] = 2.5 * t + 1.0 -> d2 should be 0
    t = np.arange(20, dtype=np.float64)
    lin = 2.5 * t + 1.0
    d2_lin = real_checks.second_differences_1d(lin)
    assert len(d2_lin) == 18
    np.testing.assert_allclose(d2_lin, 0.0, atol=1e-12)


def test_second_differences_quadratic():
    # Quadratic: x[t] = t^2 -> d2 = (t+2)^2 - 2(t+1)^2 + t^2 = 2
    t = np.arange(15, dtype=np.float64)
    quad = t ** 2
    d2 = real_checks.second_differences_1d(quad)
    assert len(d2) == 13
    np.testing.assert_allclose(d2, 2.0)


def test_second_differences_nan_handling():
    # Isolated NaNs: triples touching NaN are excluded
    # Indices: 0, 1, 2, 3, 4, 5, 6, 7
    # Array:  [0, 1, nan, 3, 4, 5, 6, 7]
    # Triples:
    # 0,1,nan -> invalid
    # 1,nan,3 -> invalid
    # nan,3,4 -> invalid
    # 3,4,5 -> valid (diff = 5 - 8 + 3 = 0)
    # 4,5,6 -> valid (diff = 6 - 10 + 4 = 0)
    # 5,6,7 -> valid (diff = 7 - 12 + 5 = 0)
    arr = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    d2 = real_checks.second_differences_1d(arr)
    assert len(d2) == 3
    np.testing.assert_allclose(d2, 0.0)


def test_second_differences_blocks():
    # Ensure second differences do not span across block boundaries
    arr = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    # Block 1: 0..2 (diff at 0,1,2 = 0)
    # Block 2: 3..5 (diff at 3,4,5 = 0)
    blocks = [(1, 0, 2), (2, 3, 5)]
    d2 = real_checks.second_differences_1d(arr, blocks=blocks)
    assert len(d2) == 2
    np.testing.assert_allclose(d2, 0.0)


def test_ball_roughness_stats_empty_and_short():
    short = np.array([1.0, 2.0])
    stats = real_checks.ball_roughness_stats(short, short)
    assert stats["x"] == (0.0, 0.0)
    assert stats["y"] == (0.0, 0.0)


def test_ball_roughness_stats_basic():
    bx = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    by = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])  # quadratic, second diff is 2.0
    stats = real_checks.ball_roughness_stats(bx, by)
    assert stats["x"] == (0.0, 0.0)
    assert stats["y"] == (2.0, 2.0)


def _make_synth_kick_trk(n=60, kick_t=30, player_near=True):
    """Build a synthetic Trk with a sharp ball velocity reversal at kick_t."""
    fps = 25
    # Ball moves right at 5 m/s, then reverses to move left at 5 m/s
    # Velocity change = 10 m/s
    bx = np.empty(n, dtype=np.float32)
    by = np.zeros(n, dtype=np.float32)
    for t in range(n):
        if t < kick_t:
            bx[t] = 50.0 + (t - kick_t) * (5.0 / fps)
        else:
            bx[t] = 50.0 - (t - kick_t) * (5.0 / fps)

    # Players: one home (0) and one away (1)
    if player_near:
        px = np.repeat(bx[:, None], 2, axis=1)
    else:
        px = np.full((n, 2), 1000.0, dtype=np.float32)
    py = np.zeros((n, 2), dtype=np.float32)
    ps = np.full((n, 2), 1.0, dtype=np.float32)
    team = np.array([0, 1], dtype=np.int8)
    blocks = [(1, 0, n - 1)]
    return {
        "bx": bx,
        "by": by,
        "px": px,
        "py": py,
        "ps": ps,
        "team": team,
        "fps": fps,
        "blocks": blocks,
    }


def test_kick_enrichment_no_kicks():
    # Constant ball position: dv = 0, no kicks
    n = 50
    trk = {
        "bx": np.full(n, 50.0, dtype=np.float32),
        "by": np.full(n, 30.0, dtype=np.float32),
        "px": np.full((n, 2), 50.0, dtype=np.float32),
        "py": np.full((n, 2), 30.0, dtype=np.float32),
        "ps": np.full((n, 2), 1.0, dtype=np.float32),
        "team": np.array([0, 1], dtype=np.int8),
        "fps": 25,
        "blocks": [(1, 0, n - 1)],
    }
    res = real_checks.kick_enrichment(trk, thresholds=(3.0, 5.0, 8.0))
    assert res["usable_minutes"] > 0.0
    assert res["baseline_share"] == 1.0  # player is on top of ball
    for row in res["rows"]:
        assert row["count"] == 0
        assert row["per_min"] == 0.0
        assert row["near_share"] == 0.0


def test_kick_enrichment_detected_kick():
    # Kick with player near
    trk_near = _make_synth_kick_trk(n=60, kick_t=30, player_near=True)
    res_near = real_checks.kick_enrichment(trk_near, thresholds=(3.0, 5.0, 8.0))
    assert res_near["baseline_share"] == 1.0
    for row in res_near["rows"]:
        assert row["count"] == 1
        assert row["near_share"] == 1.0

    # Kick with player far (player at x=10m, ball at x=50m)
    trk_far = _make_synth_kick_trk(n=60, kick_t=30, player_near=False)
    res_far = real_checks.kick_enrichment(trk_far, thresholds=(3.0, 5.0, 8.0))
    assert res_far["baseline_share"] == 0.0
    for row in res_far["rows"]:
        assert row["count"] == 1
        assert row["near_share"] == 0.0


def test_team_tag_to_side():
    assert real_checks.team_tag_to_side("Barcelona") == "H"
    assert real_checks.team_tag_to_side("FC Barcelona") == "H"
    assert real_checks.team_tag_to_side("h") == "H"
    assert real_checks.team_tag_to_side("Home") == "H"

    assert real_checks.team_tag_to_side("Atletic Club") == "A"
    assert real_checks.team_tag_to_side("Athletic Club") == "A"
    assert real_checks.team_tag_to_side("a") == "A"
    assert real_checks.team_tag_to_side("Away") == "A"

    assert real_checks.team_tag_to_side("N/A") is None
    assert real_checks.team_tag_to_side("") is None
    assert real_checks.team_tag_to_side(None) is None


def test_check_half_kickoff_alternation():
    # Half 1 kicks off at frame 10 -> first live is A
    # Half 2 kicks off at frame 50 -> first live is H
    n = 100
    labels = np.full(n, "D", dtype="<U1")
    labels[15:30] = "A"
    labels[55:70] = "H"

    events = [
        {"code": "SET PIECES", "tags": {"Type": "Kick Off (Start)", "Half": "1st Half"}, "start_frame": 10},
        {"code": "SET PIECES", "tags": {"Type": "Kick Off (Start)", "Half": "2nd Half"}, "start_frame": 50},
    ]

    letters, alternates = real_checks.check_half_kickoff_alternation(labels, events)
    assert letters == ["A", "H"]
    assert alternates is True

    # Same team kicks off both halves -> no alternation
    labels_same = np.full(n, "D", dtype="<U1")
    labels_same[15:30] = "H"
    labels_same[55:70] = "H"
    letters_same, alternates_same = real_checks.check_half_kickoff_alternation(labels_same, events)
    assert letters_same == ["H", "H"]
    assert alternates_same is False


def test_check_goal_kickoffs():
    n = 200
    labels = np.full(n, "D", dtype="<U1")
    # Goal 1 by Barcelona (H) at 20..30; kickoff at 35; restart label is A (opposite)
    labels[40:60] = "A"
    # Goal 2 by Athletic Club (A) at 80..90; kickoff at 95; restart label is H (opposite)
    labels[100:120] = "H"
    # Goal 3 by Barcelona (H) at 140..150; kickoff at 155; restart label is H (same - fails!)
    labels[160:180] = "H"

    events = [
        {"code": "GOALS", "tags": {"Team": "Barcelona"}, "start_frame": 20, "end_frame": 30},
        {"code": "SET PIECES", "tags": {"Type": "Kick Off (After Goal)"}, "start_frame": 35, "end_frame": 45},
        {"code": "GOALS", "tags": {"Team": "Atletic Club"}, "start_frame": 80, "end_frame": 90},
        {"code": "SET PIECES", "tags": {"Type": "Kick Off (After Goal)"}, "start_frame": 95, "end_frame": 105},
        {"code": "GOALS", "tags": {"Team": "Barcelona"}, "start_frame": 140, "end_frame": 150},
        {"code": "SET PIECES", "tags": {"Type": "Kick Off (After Goal)"}, "start_frame": 155, "end_frame": 165},
    ]

    holds, total = real_checks.check_goal_kickoffs(labels, events)
    assert total == 3
    assert holds == 2
