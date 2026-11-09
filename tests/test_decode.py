"""Unit tests for mtah/decode.py and features.fill_ball_gaps. Synthetic arrays only -- no match
data, no ground truth.

    python -m pytest tests -q -p no:cacheprovider
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, source_frames  # noqa: E402
from mtah import decode, features  # noqa: E402
from mtah.decode import A, D, H  # noqa: E402

FIELDS = ("ball_x", "ball_y", "d_home", "d_away", "n_home", "n_away", "mean_speed", "ball_out")
K = decode.PARAMS["min_dead_frames"]


def params(**kw):
    p = dict(decode.PARAMS)
    p.update(kw)
    return p


def feats(n):
    """All-NaN features (no evidence anywhere) of length n."""
    f = {k: np.full(n, np.nan, dtype=np.float32) for k in FIELDS}
    f["ball_vis"] = np.zeros(n, dtype=np.float32)
    return f


def set_team(f, sl, team):
    """Strong, live evidence for `team` on slice sl: ball seen, that team 0.5 m away."""
    near, far = (f["d_home"], f["d_away"]) if team == H else (f["d_away"], f["d_home"])
    f["ball_vis"][sl] = 1
    near[sl] = 0.5
    far[sl] = 10.0
    f["mean_speed"][sl] = 3.0     # play is moving
    f["ball_out"][sl] = 0.0


def path_of(f, p=decode.PARAMS):
    return decode.viterbi(decode.emissions(f, p), p)


def runs(path, state):
    """[(start, end_exclusive)] of maximal runs of `state`."""
    x = np.concatenate(([0], (np.asarray(path) == state).astype(np.int8), [0]))
    d = np.diff(x)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def viterbi_3state_v1(e, p):
    """Previous round's 3-state decoder (with stay tie-break), kept as a regression reference."""
    trans = np.array([[0.0, -p["switch_cost"], -p["dead_cost"]],
                      [-p["switch_cost"], 0.0, -p["dead_cost"]],
                      [-p["dead_cost"], -p["dead_cost"], 0.0]])
    n, states = len(e), np.arange(3)
    back = np.zeros((n, 3), dtype=np.int8)
    score = e[0].copy()
    for t in range(1, n):
        cand = score[:, None] + trans
        best = np.argmax(cand, axis=0)
        stay = cand[states, states] >= cand[best, states]
        back[t] = np.where(stay, states, best)
        score = cand[back[t], states] + e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def random_emissions(rng, n):
    """Blocky random evidence: segments favouring H, A or D, plus per-frame noise."""
    e = rng.normal(0, 1.0, (n, 3))
    t = 0
    while t < n:
        dur = int(rng.integers(1, 60))
        e[t:t + dur, int(rng.integers(0, 3))] += rng.uniform(0.5, 4.0)
        t += dur
    return e


# a. hand-built emissions -> obvious path (all segments >= K, so the dead chain allows them)
def test_viterbi_obvious_path():
    truth = np.array([H] * 50 + [D] * 50 + [A] * 50 + [H] * 50)
    e = np.full((len(truth), 3), -100.0)
    e[np.arange(len(truth)), truth] = 0.0
    assert np.array_equal(decode.viterbi(e), truth)


# b. a single A blip inside H does not flip (switch_cost)
def test_single_frame_blip_does_not_flip():
    f = feats(200)
    set_team(f, slice(None), H)
    set_team(f, slice(100, 101), A)
    assert np.all(path_of(f) == H)       # also no H-D-H detour around the blip


# c. sustained A after H does flip -- exactly at the seam now (no 1-frame D detour)
def test_sustained_evidence_flips():
    f = feats(300)
    set_team(f, slice(0, 200), H)
    set_team(f, slice(200, 300), A)
    p = path_of(f)
    assert np.all(p[:200] == H)
    assert np.all(p[200:] == A)


# d. flat evidence carries the previous live state, never defaults to H
@pytest.mark.parametrize("team", [H, A])
def test_flat_stretch_carries_previous_state(team):
    f = feats(500)
    set_team(f, slice(0, 200), team)          # 300 trailing frames: all NaN
    assert np.all(path_of(f) == team)


def test_flat_gap_between_same_team_is_bridged():
    f = feats(600)
    set_team(f, slice(0, 200), A)
    set_team(f, slice(400, 600), A)           # 200 NaN frames in the middle
    assert np.all(path_of(f) == A)


def test_flat_half_start_is_X_until_first_evidence():
    f = feats(300)
    set_team(f, slice(100, 300), A)
    out = decode.decode_block(decode.emissions(f), decode.informative(f))
    assert np.all(out[:100] == "X")
    assert np.all(out[100:] == "A")


def test_all_flat_block_is_all_X():
    f = feats(300)
    out = decode.decode_block(decode.emissions(f), decode.informative(f))
    assert np.all(out == "X")


def test_viterbi_ties_prefer_staying():
    # Frame 0 favours H by exactly switch_cost, frame 1 forces A. Arriving in A at t=1 from H
    # (0 - 30) and from A (-30 + 0) score the SAME. Old argmax picked index 0 (H) and invented
    # an H->A switch; the stay tie-break keeps A for both frames.
    sc = decode.PARAMS["switch_cost"]
    e = np.array([[0.0, -sc, -100.0],
                  [-100.0, 0.0, -100.0]])
    assert decode.viterbi(e).tolist() == [A, A]


# f. a 1-frame dead blip between H and A can no longer appear
def test_one_frame_dead_blip_cannot_appear():
    f = feats(401)
    set_team(f, slice(0, 200), H)
    set_team(f, slice(201, 401), A)
    set_team(f, slice(200, 201), H)
    f["ball_out"][200] = 1.0                  # one frame of strong dead evidence at the seam
    f["mean_speed"][200] = 0.0
    old = path_of(f, params(min_dead_frames=1))
    assert np.sum(old == D) == 1              # the loophole existed with K = 1 ...
    new = path_of(f)
    assert not np.any(new == D)               # ... and is gone with K = 25
    assert len(runs(new, H)) == 1 and len(runs(new, A)) == 1


# g. no D run shorter than K strictly inside a block (clipped-at-boundary runs exempt)
@pytest.mark.parametrize("seed", range(5))
def test_no_short_dead_runs_inside(seed):
    rng = np.random.default_rng(seed)
    n = 4000
    path = decode.viterbi(random_emissions(rng, n))
    dead = runs(path, D)
    assert dead, "test is vacuous without any D runs"
    for s, e in dead:
        if s > 0 and e < n:
            assert e - s >= K, (s, e)


def test_no_short_dead_runs_in_full_decode():
    rng = np.random.default_rng(7)
    n = source_frames("tactical", GAME)
    f = feats(n)
    f["ball_vis"][:] = 1
    f["d_home"][:] = rng.uniform(0, 8, n)
    f["d_away"][:] = rng.uniform(0, 8, n)
    f["mean_speed"][:] = np.repeat(rng.uniform(0, 3, n // 50 + 1), 50)[:n]
    f["ball_out"][:] = 0
    labels = decode.decode(f)
    for _, lo, hi in half_video_blocks(GAME):
        block = labels[lo:hi + 1]
        dead = runs(block == "D", True)
        assert dead
        for s, e in dead:
            if s > 0 and e < len(block):
                assert e - s >= K


# h. the fast chain decoder is exact: matches the literal (K+2)-state Viterbi
@pytest.mark.parametrize("k", [1, 2, 5, 25])
@pytest.mark.parametrize("seed", range(3))
def test_fast_matches_dense_reference(k, seed):
    rng = np.random.default_rng(100 + seed)
    e = random_emissions(rng, 400)
    p = params(min_dead_frames=k)
    assert np.array_equal(decode.viterbi(e, p), decode.viterbi_dense(e, p))


@pytest.mark.parametrize("seed", range(3))
def test_k1_is_previous_3state_model(seed):
    e = random_emissions(np.random.default_rng(200 + seed), 3000)
    p = params(min_dead_frames=1)
    assert np.array_equal(decode.viterbi(e, p), viterbi_3state_v1(e, p))


# e. X outside the graded halves, H/A/D inside
def test_full_decode_x_outside_halves():
    n = source_frames("tactical", GAME)
    f = feats(n)
    graded = np.zeros(n, dtype=bool)
    for _, lo, hi in half_video_blocks(GAME):
        graded[lo:hi + 1] = True
        mid = (lo + hi) // 2
        set_team(f, slice(lo, mid), H)
        set_team(f, slice(mid, hi + 1), A)
    labels = decode.decode(f)
    assert len(labels) == n
    assert np.all(labels[~graded] == "X")
    assert np.all(np.isin(labels[graded], ["H", "A", "D"]))


# --- ball-gap interpolation ---------------------------------------------------------------------

def gap_feats(x_after=16.0, gap=(5, 10), n=20):
    """Ball seen at x=10 before the gap and x=x_after after it (y=10). Home player fixed at
    (13, 10) in slot 0, away player at (50, 50) in slot 1."""
    f = feats(n)
    f["pos"] = np.full((n, 2, 2), np.nan, dtype=np.float16)
    f["pos"][:, 0] = (13.0, 10.0)
    f["pos"][:, 1] = (50.0, 50.0)
    f["slot_team"] = np.array([0, 1], dtype=np.int8)
    s, e = gap
    for sl, x in ((slice(0, s), 10.0), (slice(e, n), x_after)):
        f["ball_vis"][sl] = 1
        f["ball_x"][sl] = x
        f["ball_y"][sl] = 10.0
        f["d_home"][sl] = abs(x - 13.0)
        f["d_away"][sl] = np.hypot(x - 50.0, 40.0)
        f["ball_out"][sl] = 0
        f["mean_speed"][sl] = 3.0
    return f


def test_interp_fills_short_plausible_gap():
    f = gap_feats()
    g = features.fill_ball_gaps(f, 12)
    gap = slice(5, 10)
    np.testing.assert_allclose(g["ball_x"][gap], [11, 12, 13, 14, 15], atol=1e-4)
    np.testing.assert_allclose(g["ball_y"][gap], 10.0, atol=1e-4)
    np.testing.assert_allclose(g["d_home"][gap], [2, 1, 0, 1, 2], atol=1e-2)
    assert np.all(np.isfinite(g["d_away"][gap]))
    assert np.all(g["ball_out"][gap] == 0)
    assert g["ball_interp"].tolist() == [0] * 5 + [1] * 5 + [0] * 10
    assert np.all(g["ball_vis"][gap] == 0)              # raw tracking flag untouched
    assert np.all(decode.informative(g)[gap])           # but the frames are informative
    assert np.all(np.isnan(f["ball_x"][gap]))           # input not mutated


def test_interp_skips_gap_longer_than_limit():
    g = features.fill_ball_gaps(gap_feats(), 4)         # gap is 5 frames
    assert np.all(np.isnan(g["ball_x"][5:10]))
    assert not g["ball_interp"].any()


def test_interp_skips_implausible_jump():
    g = features.fill_ball_gaps(gap_feats(x_after=70.0), 12)   # 60 m in 6 frames = 250 m/s
    assert np.all(np.isnan(g["ball_x"][5:10]))
    assert not g["ball_interp"].any()


def test_interp_skips_gap_at_block_edge():
    f = gap_feats(gap=(0, 5))                           # no ball seen before the gap
    g = features.fill_ball_gaps(f, 12)
    assert not g["ball_interp"].any()


def test_interp_off_is_byte_identical():
    rng = np.random.default_rng(3)
    n = source_frames("tactical", GAME)
    f = feats(n)
    seen = rng.random(n) < 0.6
    # slow ball (random walk, <0.3 m/frame) so short gaps are plausible and do get filled
    bx = 50 + np.cumsum(rng.uniform(-0.3, 0.3, n))
    f["ball_vis"][:] = seen
    f["ball_x"][seen] = bx[seen]
    f["ball_y"][seen] = 34.0
    f["d_home"][seen] = rng.uniform(0, 8, seen.sum())
    f["d_away"][seen] = rng.uniform(0, 8, seen.sum())
    f["mean_speed"][:] = 2.0
    f["ball_out"][seen] = 0
    f["pos"] = np.full((n, 2, 2), np.nan, dtype=np.float16)
    f["pos"][:, 0, 0], f["pos"][:, 0, 1] = bx, 34.0          # home player on the ball
    f["pos"][:, 1, 0], f["pos"][:, 1, 1] = bx + 20, 34.0     # away player 20 m off
    f["slot_team"] = np.array([0, 1], dtype=np.int8)

    assert features.fill_ball_gaps(f, 0) is f
    off = decode.decode(f, p=params(interp_gap_frames=0))
    bare = {k: v for k, v in f.items() if k not in ("pos", "slot_team")}
    assert np.array_equal(off, decode.decode(bare, p=params(interp_gap_frames=0)))

    # flag on actually changes the evidence on filled frames (labels may or may not move)
    g = features.fill_ball_gaps(f, 12)
    assert g["ball_interp"].sum() > 1000
    assert not np.array_equal(decode.emissions(g), decode.emissions(f))
