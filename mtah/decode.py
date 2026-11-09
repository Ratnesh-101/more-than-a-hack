"""v1 model: per-frame evidence -> HMM (H, A, D1..DK) -> Viterbi path.

Why an HMM: spellF1 rewards whole, correctly-bounded possessions. Per-frame classifiers flicker
and invent possessions; a switch penalty in the decoder makes the path change state only when the
evidence for the new state outweighs the cost. Loose balls (nobody near) give flat evidence, so the
path keeps the last team -- exactly the "contested ball stays with the last possessor" rule in
CHALLENGE.md.

Dead ball is a chain of K = min_dead_frames states. Entering dead ball goes to D1, then
D1 -> D2 -> ... -> DK is forced at no cost, and only DK may stay or exit to H/A. Every D_i emits
with the dead-ball emission and all collapse to "D" in the output. Without the chain, two
dead_cost hops (H -> D -> A) were cheaper than one switch_cost, so the decoder could route a
hand-over through a 1-frame "stoppage". A real stoppage lasts at least ~1 s. K = 1 is exactly
the old 3-state model.

Every constant in PARAMS is a football prior, NOT tuned on the public 5 minutes (RULES.md forbids
using it as a tuning set). Change them for a reason you can write in the technical doc.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from mtah import features  # noqa: E402

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

PARAMS = {
    "team_scale_m": 2.0,      # softness of nearest-team evidence (m)
    "control_radius_m": 3.0,  # beyond this nobody controls the ball -> evidence fades to flat
    "slow_speed": 1.5,        # mean player speed (m/s) below which play looks stopped
    "slow_width": 0.35,
    "dead_bias": -2.5,        # prior log-odds of dead ball on a normal frame
    "dead_out_bonus": 5.0,    # ball outside the lines -> strongly dead
    "dead_slow_bonus": 3.0,   # everyone slow -> likely stoppage
    "switch_cost": 30.0,      # log-penalty for H <-> A (higher = fewer possessions)
    "dead_cost": 12.0,        # log-penalty for live <-> dead
    "min_dead_frames": 25,    # a real stoppage (out, foul, restart) lasts >= ~1 s at 25 fps
    "interp_gap_frames": 12,  # fill ball gaps up to ~0.5 s (features.fill_ball_gaps); 0 = off
}

EPS = 1e-6


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _seen(f):
    """Ball position known on this frame: tracked, or (if enabled) interpolated."""
    seen = f["ball_vis"] > 0.5
    if "ball_interp" in f:
        seen = seen | (f["ball_interp"] > 0.5)
    return seen


def emissions(f, p=PARAMS):
    """(n, 3) log-likelihoods for H, A, D."""
    n = len(f["ball_vis"])
    vis = _seen(f)
    d_h = np.where(np.isnan(f["d_home"]), 99.0, f["d_home"])
    d_a = np.where(np.isnan(f["d_away"]), 99.0, f["d_away"])

    # team evidence: softmax over -distance, faded toward 50/50 when nobody is close
    s = p["team_scale_m"]
    z_h, z_a = -d_h / s, -d_a / s
    m = np.maximum(z_h, z_a)
    log_norm = m + np.log(np.exp(z_h - m) + np.exp(z_a - m))
    conf = np.exp(-np.minimum(d_h, d_a) / p["control_radius_m"])
    conf = np.where(vis, conf, 0.0)
    lh = conf * (z_h - log_norm) + (1 - conf) * np.log(0.5)
    la = conf * (z_a - log_norm) + (1 - conf) * np.log(0.5)

    # dead-ball evidence
    speed = np.where(np.isnan(f["mean_speed"]), p["slow_speed"], f["mean_speed"])
    slow = _sigmoid((p["slow_speed"] - speed) / p["slow_width"])
    out = np.where(np.isnan(f["ball_out"]), 0.0, f["ball_out"])
    p_dead = _sigmoid(p["dead_bias"] + p["dead_out_bonus"] * out + p["dead_slow_bonus"] * slow)
    p_dead = np.clip(p_dead, EPS, 1 - EPS)

    e = np.empty((n, 3), dtype=np.float64)
    e[:, H] = lh + np.log(1 - p_dead)
    e[:, A] = la + np.log(1 - p_dead)
    e[:, D] = np.log(p_dead)
    return e


def viterbi(e, p=PARAMS):
    """Most likely H/A/D path for one contiguous block, with dead ball as a K-state chain.

    Exact Viterbi over the states (H, A, D1..DK), computed in O(1) per frame instead of O(K):
    the chain is deterministic, so "in D_i at t" just means "entered dead ball at t-i+1", and
    arriving in DK from the chain at t is "entered at t-K+1" plus the summed D emissions
    (a prefix sum). The path may start in dead ball (any phase) and may end mid-chain, so a D run
    clipped by the block boundary can be shorter than K; any D run strictly inside is >= K.

    Ties are broken towards STAYING in the current state (never towards index 0 = H), so a
    stretch of flat evidence carries the previous state instead of drifting to home.
    viterbi_dense() is the literal (K+2)-state reference; tests check the two agree.
    """
    sw, dc, K = float(p["switch_cost"]), float(p["dead_cost"]), int(p["min_dead_frames"])
    if K < 1:
        raise ValueError("min_dead_frames must be >= 1")
    n = len(e)
    eH, eA, eD = (e[:, j].tolist() for j in (H, A, D))
    cD = np.concatenate(([0.0], np.cumsum(e[:, D]))).tolist()   # cD[t+1] = sum eD[0..t]
    ninf = -np.inf

    enter = [ninf] * n            # score of being in D1 at t (just entered dead ball)
    enter_from = [H] * n          # H or A at t-1
    bH = bytearray(n)             # predecessor of H at t: 0=H 1=A 2=DK
    bA = bytearray(n)             # predecessor of A at t: 0=H 1=A 2=DK
    bK = bytearray(n)             # DK at t: 0 = stayed in DK, 1 = arrived from the chain

    # t = 0: may start live, or in dead ball. Starting in DK dominates starting at any earlier
    # phase (same emissions, freer exit), so dead-ball start is modelled as DK.
    sH, sA, sK = eH[0], eA[0], eD[0]
    for t in range(1, n):
        # enter D1 at t from a live state at t-1
        if sA > sH:
            enter[t], enter_from[t] = sA - dc + eD[t], A
        else:
            enter[t], enter_from[t] = sH - dc + eD[t], H
        # DK at t: stay, or finish the chain entered at t-K+1 (only for K >= 2 that is not t)
        kst = sK
        t0 = t - K + 1
        if K == 1:
            kch = enter[t] - eD[t]            # D1 == DK: arriving now, emission added below
        elif t0 >= 1:
            kch = enter[t0] + (cD[t] - cD[t0 + 1])   # chain emissions t0+1 .. t-1
        else:
            kch = ninf
        # H at t
        a, b, c = sH, sA - sw, sK - dc
        if a >= b and a >= c:
            nH, bH[t] = a, 0
        elif b >= c:
            nH, bH[t] = b, 1
        else:
            nH, bH[t] = c, 2
        # A at t
        a, b, c = sA, sH - sw, sK - dc
        if a >= b and a >= c:
            nA, bA[t] = a, 1
        elif b >= c:
            nA, bA[t] = b, 0
        else:
            nA, bA[t] = c, 2
        if kst >= kch:
            nK, bK[t] = kst, 0
        else:
            nK, bK[t] = kch, 1
        sH, sA, sK = nH + eH[t], nA + eA[t], nK + eD[t]

    # end: H, A, DK, or mid-chain (entered at t0 > n-K, still in D_{n-t0})
    best, state, t0_end = sH, H, -1
    if sA > best:
        best, state = sA, A
    if sK > best:
        best, state = sK, 2
    for t0 in range(max(1, n - K + 1), n):
        if K == 1:
            break
        v = enter[t0] + (cD[n] - cD[t0 + 1])
        if v > best:
            best, state, t0_end = v, 3, t0

    path = np.empty(n, dtype=np.int8)
    t = n - 1
    if state == 3:                      # ended mid-chain
        path[t0_end:] = D
        t, state = t0_end - 1, enter_from[t0_end]
    while t >= 0:
        if state == H:
            path[t] = H
            prev = bH[t]
        elif state == A:
            path[t] = A
            prev = bA[t]
        else:                           # DK
            if t == 0:
                path[0] = D
                break
            if bK[t] == 0:
                path[t] = D
                prev = 2
            else:                       # chain entered at t0 = t-K+1 covers t0..t
                t0 = t - K + 1
                path[t0:t + 1] = D
                t, state = t0 - 1, enter_from[t0]
                continue
        if t == 0:
            break
        state = (H, A, 2)[prev]
        t -= 1
    return path


def viterbi_dense(e, p=PARAMS):
    """Reference implementation: literal (K+2)-state Viterbi with a full transition matrix.

    States 0=H, 1=A, 2..K+1 = D1..DK. O(n K^2) -- for tests only. Same start/end and tie rules
    as viterbi(): start live or in DK, end anywhere; ties prefer staying.
    """
    sw, dc, K = p["switch_cost"], p["dead_cost"], int(p["min_dead_frames"])
    S = K + 2
    DK = S - 1
    T = np.full((S, S), -np.inf)
    T[H, H] = T[A, A] = 0.0
    T[H, A] = T[A, H] = -sw
    T[H, 2] = T[A, 2] = -dc
    for i in range(2, DK):
        T[i, i + 1] = 0.0
    T[DK, DK] = 0.0
    T[DK, H] = T[DK, A] = -dc
    if K == 1:                          # D1 == DK
        T[2, 2] = 0.0
    em = np.concatenate([e[:, :2], np.repeat(e[:, 2:3], K, axis=1)], axis=1)
    n = len(e)
    states = np.arange(S)
    score = np.full(S, -np.inf)
    score[[H, A, DK]] = em[0, [H, A, DK]]
    back = np.zeros((n, S), dtype=np.int64)
    for t in range(1, n):
        cand = score[:, None] + T
        best = np.argmax(cand, axis=0)
        stay = cand[states, states] >= cand[best, states]
        back[t] = np.where(stay, states, best)
        score = cand[back[t], states] + em[t]
    s = int(np.argmax(score))
    path = np.empty(n, dtype=np.int64)
    path[-1] = s
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.minimum(path, D).astype(np.int8)


def informative(f):
    """Frames with real team evidence: ball known (tracked or interpolated) and at least one
    team's distance known."""
    return _seen(f) & (np.isfinite(f["d_home"]) | np.isfinite(f["d_away"]))


def decode_block(e, info, p=PARAMS):
    """Letters for one half. X before the first informative frame (no evidence = no guess;
    the scorer back-fills it from the first real label), Viterbi from there on."""
    out = np.full(len(e), "X", dtype="<U1")
    idx = np.flatnonzero(info)
    if len(idx):
        out[idx[0]:] = LETTERS[viterbi(e[idx[0]:], p)]
    return out


def decode(f, game=GAME, p=PARAMS):
    """Full-length label array: H/A/D inside the graded halves (X only on a leading stretch
    with no evidence at all), X outside."""
    n = len(f["ball_vis"])
    labels = np.full(n, "X", dtype="<U1")
    f = features.fill_ball_gaps(f, int(p.get("interp_gap_frames", 0)))
    e = emissions(f, p)
    info = informative(f)
    for _, lo, hi in half_video_blocks(game):
        labels[lo:hi + 1] = decode_block(e[lo:hi + 1], info[lo:hi + 1], p)
    return labels
