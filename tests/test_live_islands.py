"""Unit tests for labelling evidence-free live islands dead in mtah/v2.

Tests:
  - synthetic island -> D
  - island with one sighting -> unchanged
  - run at a half edge unchanged
  - run next to X unchanged
  - no-op when there are no islands
"""

import numpy as np
import pytest

from mtah.v2 import decode, fallback, tracking


def _make_synth_trk(n=100, blocks=None, kickoffs=None, bx=None):
    """Minimal synthetic Trk dict for testing."""
    if blocks is None:
        blocks = [(1, 0, n - 1)]
    px = np.zeros((n, 2), dtype=np.float32)
    py = np.zeros((n, 2), dtype=np.float32)
    ps = np.full((n, 2), 2.0, dtype=np.float32)
    team = np.array([0, 1], dtype=np.int8)
    if bx is None:
        bx = np.full(n, 50.0, dtype=np.float32)
    by = np.full(n, 30.0, dtype=np.float32)
    return tracking.build(px, py, ps, team, bx, by, (105.0, 68.0), blocks, kickoffs=kickoffs)


# 1. Synthetic island -> D
@pytest.mark.parametrize("letter", ["H", "A"])
def test_synthetic_island_becomes_dead(letter):
    # A maximal H or A run bounded by D on both sides with all-NaN ball becomes D
    labels = np.array(["D", "D", letter, letter, letter, "D", "D"], dtype="<U1")
    bx = np.array([50.0, 50.0, np.nan, np.nan, np.nan, 50.0, 50.0], dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    expected = np.array(["D", "D", "D", "D", "D", "D", "D"], dtype="<U1")
    assert np.array_equal(res, expected)


def test_synthetic_island_multiple():
    # Both H and A islands in the same half become D
    labels = np.array(["D", "H", "H", "D", "A", "A", "D"], dtype="<U1")
    bx = np.full(len(labels), np.nan, dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    expected = np.array(["D", "D", "D", "D", "D", "D", "D"], dtype="<U1")
    assert np.array_equal(res, expected)


def test_synthetic_island_via_decode_half():
    # Verify through decode_half
    n = 10
    bx = np.full(n, 50.0, dtype=np.float32)
    bx[3:7] = np.nan  # frames 3..6 have no ball
    trk = _make_synth_trk(n=n, blocks=[(1, 0, n - 1)], bx=bx)

    def segment(s, h):
        # returns D D D H H H H D D D
        seg = np.full(h - s + 1, "D", dtype="<U1")
        seg[3:7] = "H"
        return seg

    res = decode.decode_half(trk, 1, 0, n - 1, segment)
    assert np.all(res == "D")


# 2. Island with one sighting -> unchanged
@pytest.mark.parametrize("sighting_offset", [0, 1, 2])
@pytest.mark.parametrize("letter", ["H", "A"])
def test_island_with_one_sighting_unchanged(sighting_offset, letter):
    # A run bounded by D on both sides with at least one ball sighting is preserved
    labels = np.array(["D", "D", letter, letter, letter, "D", "D"], dtype="<U1")
    bx = np.array([50.0, 50.0, np.nan, np.nan, np.nan, 50.0, 50.0], dtype=np.float32)
    bx[2 + sighting_offset] = 50.0  # one sighting in the run
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


# 3. Run at a half edge unchanged
def test_run_at_left_half_edge_unchanged():
    # Run at frame 0 (left edge of half) is not bounded by D on the left
    labels = np.array(["H", "H", "H", "D", "D"], dtype="<U1")
    bx = np.full(len(labels), np.nan, dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


def test_run_at_right_half_edge_unchanged():
    # Run at the end of the half is not bounded by D on the right
    labels = np.array(["D", "D", "A", "A", "A"], dtype="<U1")
    bx = np.full(len(labels), np.nan, dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


def test_run_spanning_entire_half_unchanged():
    # Run spanning the entire half
    labels = np.array(["H", "H", "H", "H"], dtype="<U1")
    bx = np.full(len(labels), np.nan, dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


# 4. Run next to X unchanged
def test_run_next_to_x_left_unchanged():
    # Left neighbor is X
    labels = np.array(["X", "H", "H", "D"], dtype="<U1")
    bx = np.full(len(labels), np.nan, dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


def test_run_next_to_x_right_unchanged():
    # Right neighbor is X
    labels = np.array(["D", "A", "A", "X"], dtype="<U1")
    bx = np.full(len(labels), np.nan, dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


def test_run_between_x_unchanged():
    # Both neighbors are X
    labels = np.array(["X", "H", "H", "X"], dtype="<U1")
    bx = np.full(len(labels), np.nan, dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


# 5. No-op when there are no islands
def test_no_op_when_no_islands():
    # Visible ball throughout: no change
    labels = np.array(["D", "H", "H", "D", "A", "A", "D"], dtype="<U1")
    bx = np.full(len(labels), 50.0, dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


def test_no_op_direct_team_handover():
    # H directly transitioning to A without D in between: neither is bounded by D on both sides
    labels = np.array(["D", "H", "H", "A", "A", "D"], dtype="<U1")
    bx = np.full(len(labels), np.nan, dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


def test_no_op_all_dead():
    labels = np.array(["D", "D", "D", "D"], dtype="<U1")
    bx = np.full(len(labels), np.nan, dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


def test_no_op_empty():
    labels = np.array([], dtype="<U1")
    bx = np.array([], dtype=np.float32)
    res = decode._label_evidence_free_islands_dead(labels.copy(), bx)
    assert np.array_equal(res, labels)


# 6. Consistency across all three variants
@pytest.mark.parametrize("variant", ["champion", "touch_only", "semi_markov"])
def test_all_variants_use_decode_half_island_pass(variant):
    n = 20
    bx = np.full(n, 50.0, dtype=np.float32)
    # Put an evidence-free island at frames 5..9
    bx[5:10] = np.nan
    trk = _make_synth_trk(n=n, blocks=[(1, 0, n - 1)], bx=bx)

    dummy_half = np.array(["D"] * 5 + ["H"] * 5 + ["D"] * 10, dtype="<U1")
    res = decode.decode_half(trk, 1, 0, n - 1, lambda s, h: dummy_half[s:h + 1])
    assert np.all(res == "D"), f"variant {variant} decode_half failed to label island dead"
