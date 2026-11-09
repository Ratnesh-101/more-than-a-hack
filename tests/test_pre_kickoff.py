"""Unit tests for the pre-kickoff dead ball rule in mtah/v2/.

Tests the champion decoder and fallbacks (touch_only, semi_markov) on:
  (1) synthetic case where head looks live but a kickoff tag exists later -> head is D
  (2) no tag -> output unchanged
  (3) a half whose kickoff tag starts at the first graded frame -> unchanged
"""

import numpy as np
import pytest

from mtah.v2 import decode, fallback, tracking


def _make_synth_trk(n=100, blocks=None, kickoffs=None):
    """Minimal valid synthetic Trk dict for decoder testing."""
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


def _live_emissions(n, team=0):
    """Emissions (n, 3) strongly favoring `team` (0=H, 1=A) and strongly live."""
    e = np.full((n, 3), -50.0)
    e[:, team] = 0.0      # log P(team | live) = 0
    e[:, 2] = -100.0      # log P(dead) = -100 (almost 0 probability dead)
    return e


# (1) synthetic case where the head looks live but a kickoff tag exists later -> head is D
@pytest.mark.parametrize("variant", ["champion", "touch_only", "semi_markov"])
def test_pre_kickoff_head_becomes_dead(variant):
    n = 100
    lo, hi = 0, n - 1
    kick_frame = 35
    blocks = [(1, lo, hi)]

    # Head looks strongly live (Home team possession)
    e = _live_emissions(n, team=0)

    # 1. Without kickoff tag: the head is decoded as live (H)
    trk_no_tag = _make_synth_trk(n=n, blocks=blocks, kickoffs=None)
    if variant == "champion":
        labels_no_tag = decode.decode(trk_no_tag, e)
    else:
        # Patch emissions for fallback variant test
        labels_no_tag = (
            decode.decode_half(
                trk_no_tag, 1, lo, hi,
                lambda s, h: decode.LETTERS[fallback._viterbi(e[s:h + 1], fallback.SWITCH_COST,
                                                              fallback.DEAD_COST, fallback.MIN_DEAD_FRAMES)])
            if variant == "touch_only"
            else decode.decode_half(
                trk_no_tag, 1, lo, hi,
                lambda s, h: fallback._semi_markov_block(e[s:h + 1], trk_no_tag["fps"]))
        )
    assert (labels_no_tag[lo:kick_frame] == "H").all(), "baseline without tag should be live"

    # 2. With kickoff tag at kick_frame: every frame before kickoff is labelled D
    trk_with_tag = _make_synth_trk(n=n, blocks=blocks, kickoffs={1: kick_frame})
    if variant == "champion":
        labels_with_tag = decode.decode(trk_with_tag, e)
    else:
        labels_with_tag = (
            decode.decode_half(
                trk_with_tag, 1, lo, hi,
                lambda s, h: decode.LETTERS[fallback._viterbi(e[s:h + 1], fallback.SWITCH_COST,
                                                              fallback.DEAD_COST, fallback.MIN_DEAD_FRAMES)])
            if variant == "touch_only"
            else decode.decode_half(
                trk_with_tag, 1, lo, hi,
                lambda s, h: fallback._semi_markov_block(e[s:h + 1], trk_with_tag["fps"]))
        )

    # All frames before kickoff are D
    assert (labels_with_tag[lo:kick_frame] == "D").all(), "pre-kickoff head must be D"
    # From kickoff onward, decoder continues normally (live H)
    assert (labels_with_tag[kick_frame:hi + 1] == "H").all(), "post-kickoff must continue normally"


# (2) no tag -> output unchanged
@pytest.mark.parametrize("variant", ["champion", "touch_only", "semi_markov"])
def test_no_tag_output_unchanged(variant):
    n = 120
    lo, hi = 10, 100
    blocks = [(1, lo, hi)]
    e = _live_emissions(n, team=1)  # live Away team

    trk_none = _make_synth_trk(n=n, blocks=blocks, kickoffs=None)
    trk_empty = _make_synth_trk(n=n, blocks=blocks, kickoffs={})
    trk_other_half = _make_synth_trk(n=n, blocks=blocks, kickoffs={2: 50})

    if variant == "champion":
        res_none = decode.decode(trk_none, e)
        res_empty = decode.decode(trk_empty, e)
        res_other = decode.decode(trk_other_half, e)
    elif variant == "touch_only":
        res_none = decode.decode_half(
            trk_none, 1, lo, hi,
            lambda s, h: decode.LETTERS[fallback._viterbi(e[s:h + 1], fallback.SWITCH_COST,
                                                          fallback.DEAD_COST, fallback.MIN_DEAD_FRAMES)])
        res_empty = decode.decode_half(
            trk_empty, 1, lo, hi,
            lambda s, h: decode.LETTERS[fallback._viterbi(e[s:h + 1], fallback.SWITCH_COST,
                                                          fallback.DEAD_COST, fallback.MIN_DEAD_FRAMES)])
        res_other = decode.decode_half(
            trk_other_half, 1, lo, hi,
            lambda s, h: decode.LETTERS[fallback._viterbi(e[s:h + 1], fallback.SWITCH_COST,
                                                          fallback.DEAD_COST, fallback.MIN_DEAD_FRAMES)])
    else:
        res_none = decode.decode_half(
            trk_none, 1, lo, hi,
            lambda s, h: fallback._semi_markov_block(e[s:h + 1], trk_none["fps"]))
        res_empty = decode.decode_half(
            trk_empty, 1, lo, hi,
            lambda s, h: fallback._semi_markov_block(e[s:h + 1], trk_empty["fps"]))
        res_other = decode.decode_half(
            trk_other_half, 1, lo, hi,
            lambda s, h: fallback._semi_markov_block(e[s:h + 1], trk_other_half["fps"]))

    assert np.array_equal(res_none, res_empty), "None vs empty kickoffs must be identical"
    assert np.array_equal(res_none, res_other), "kickoff for unrelated half must leave half unchanged"


# (3) a half whose kickoff tag starts at the first graded frame -> unchanged
@pytest.mark.parametrize("variant", ["champion", "touch_only", "semi_markov"])
def test_kickoff_at_first_graded_frame_unchanged(variant):
    n = 120
    lo, hi = 15, 110
    blocks = [(1, lo, hi)]
    e = _live_emissions(n, team=0)

    trk_no_tag = _make_synth_trk(n=n, blocks=blocks, kickoffs=None)
    trk_at_lo = _make_synth_trk(n=n, blocks=blocks, kickoffs={1: lo})
    trk_before_lo = _make_synth_trk(n=n, blocks=blocks, kickoffs={1: lo - 5})

    if variant == "champion":
        base = decode.decode(trk_no_tag, e)
        at_lo = decode.decode(trk_at_lo, e)
        before_lo = decode.decode(trk_before_lo, e)
    elif variant == "touch_only":
        dec = lambda t, s, h: decode.LETTERS[fallback._viterbi(  # noqa: E731
            e[s:h + 1], fallback.SWITCH_COST, fallback.DEAD_COST, fallback.MIN_DEAD_FRAMES)]
        base = decode.decode_half(trk_no_tag, 1, lo, hi, lambda s, h: dec(trk_no_tag, s, h))
        at_lo = decode.decode_half(trk_at_lo, 1, lo, hi, lambda s, h: dec(trk_at_lo, s, h))
        before_lo = decode.decode_half(trk_before_lo, 1, lo, hi, lambda s, h: dec(trk_before_lo, s, h))
    else:
        dec = lambda t, s, h: fallback._semi_markov_block(e[s:h + 1], t["fps"])  # noqa: E731
        base = decode.decode_half(trk_no_tag, 1, lo, hi, lambda s, h: dec(trk_no_tag, s, h))
        at_lo = decode.decode_half(trk_at_lo, 1, lo, hi, lambda s, h: dec(trk_at_lo, s, h))
        before_lo = decode.decode_half(trk_before_lo, 1, lo, hi, lambda s, h: dec(trk_before_lo, s, h))

    assert np.array_equal(at_lo, base), "tag starting at first graded frame must be unchanged"
    assert np.array_equal(before_lo, base), "tag starting before first graded frame must be unchanged"


def test_parse_kickoffs_from_events():
    """Test extracting kickoff frames from Smart Tagging event dictionaries."""
    events = [
        {"code": "SET PIECES", "start_frame": 100, "tags": {"Type": "Corner", "Half": "1st Half"}},
        {"code": "SET PIECES", "start_frame": 250, "tags": {"Type": "Kick Off (Start)", "Half": "1st Half"}},
        {"code": "SET PIECES", "start_frame": 5000, "tags": {"Type": "Kick Off (Start)", "Half": "2nd Half"}},
        {"code": "SET PIECES", "start_frame": 300, "tags": {"Type": "Free Kick", "Half": "1st Half"}},
    ]
    parsed = tracking.parse_kickoffs(events)
    assert parsed == {1: 250, 2: 5000}


def test_parse_kickoffs_case_and_seconds():
    """Test case-insensitivity and start_s conversion."""
    events = [
        {"code": "set pieces", "start_s": 10.0, "tags": {"Type": "kick off (start)", "Half": "First Half"}},
    ]
    parsed = tracking.parse_kickoffs(events, fps=25)
    assert parsed == {1: 250}
