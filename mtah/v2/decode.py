"""Emissions and decoder of the champion (stack_2).

Emissions: team log-probabilities from touch events (touch.py) given "live", multiplied by the
live probability from the dead-ball logit (deadball.py). Decoder: Viterbi over H, A, D where every
dead run lasts at least MIN_DEAD_S, solved exactly by a semi-Markov D chain in O(n) scalar steps.
"""

import numpy as np

from mtah.v2 import deadball, touch

SWITCH_COST = 30.0    # ~1.5 s of new-team touching flips; penalizes deflections, though loose-ball evidence can still exceed 60 nats
DEAD_COST = 12.0      # v1's live <-> dead penalty, untouched
MIN_DEAD_S = 3.0      # regularization floor to prevent brief D flickers from splitting possessions; real quick restarts can be shorter
EPS = 1e-6            # keeps log-probabilities finite
LETTERS = np.array(["H", "A", "D"])


def emissions(trk):
    """(n, 3) float64 log-likelihoods for H, A, D on every frame."""
    r = touch.team_logodds(trk)
    team_h, team_a = -np.logaddexp(0.0, -r), -np.logaddexp(0.0, r)
    p_dead = np.clip(1.0 / (1.0 + np.exp(-deadball.dead_logit(trk))), EPS, 1 - EPS)
    return np.stack([team_h + np.log1p(-p_dead), team_a + np.log1p(-p_dead), np.log(p_dead)], axis=1)


def _viterbi_min_dead(e, k, sw, dc):
    """Exact Viterbi over H, A, D where every D run after the first frame lasts >= k frames.
    D1 = first dead frame, DK = k-th dead frame or later; the stages between are forced, so the
    chain reaching DK at t scores D1(t-k+1) + dead emissions in between (prefix sums). The block
    may start in DK and may end in H, A, DK or inside a not-yet-complete chain.
    Ties keep the current state."""
    n = len(e)
    eh, ea, ed = e[:, 0].tolist(), e[:, 1].tolist(), e[:, 2].tolist()
    cd = np.concatenate([[0.0], np.cumsum(e[:, 2])]).tolist()
    H, A, DK = eh[0], ea[0], ed[0]
    d1 = [-np.inf] * n
    bh, ba, bdk, b1 = [0] * n, [1] * n, [2] * n, [0] * n   # 0 H, 1 A, 2 DK, 3 chain
    for t in range(1, n):
        ch = max((H, 1, 0), (A - sw, 0, 1), (DK - dc, 0, 2))   # (score, stay-tiebreak, state)
        ca = max((A, 1, 1), (H - sw, 0, 0), (DK - dc, 0, 2))
        j = t - k + 1
        chain = d1[j] + cd[t] - cd[j + 1] if j >= 1 else -np.inf
        d1[t], b1[t] = (H - dc + ed[t], 0) if H >= A else (A - dc + ed[t], 1)
        bdk[t] = 2 if DK >= chain else 3
        H, A, DK = ch[0] + eh[t], ca[0] + ea[t], max(DK, chain) + ed[t]
        bh[t], ba[t] = ch[2], ca[2]
    # End: H, A, DK, or still inside a dead chain entered at t0 > n-k (a block may end mid-stoppage).
    best, s, t0_end = H, 0, -1
    if A > best:
        best, s = A, 1
    if DK > best:
        best, s = DK, 2
    for t0 in range(max(1, n - k + 1), n):
        v = d1[t0] + cd[n] - cd[t0 + 1]
        if v > best:
            best, s, t0_end = v, 3, t0
    path = np.empty(n, dtype=np.int8)
    t = n - 1
    if s == 3:                                     # ended mid-chain: frames t0_end .. n-1 all dead
        path[t0_end:] = 2
        t, s = t0_end - 1, b1[t0_end]
    while t >= 0:
        path[t] = s
        if s == 2 and bdk[t] == 3:                 # forced chain: frames t-k+1 .. t all dead
            start = t - k + 1
            path[start:t] = 2
            s, t = b1[start], start - 1
        else:
            s, t = (bh, ba, bdk)[s][t], t - 1
    return path


def kickoff_frame(trk, half):
    """Start frame of half's Kick Off (Start) tag if present in trk, else None."""
    return (trk.get("kickoffs") or {}).get(half)


def _label_evidence_free_islands_dead(labels, bx):
    """Maximal H or A runs bounded by D on both sides with zero ball sightings become D."""
    if len(labels) == 0 or bx is None or len(bx) != len(labels):
        return labels
    cut = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1, [len(labels)]))
    for s, e in zip(cut[:-1], cut[1:]):
        if labels[s] in ("H", "A") and s > 0 and e < len(labels):
            if labels[s - 1] == "D" and labels[e] == "D" and np.isnan(bx[s:e]).all():
                labels[s:e] = "D"
    return labels


def decode_half(trk, half, lo, hi, decode_segment):
    """Labels for one graded half [lo, hi]: D before that half's kickoff, decoded after."""
    kick = kickoff_frame(trk, half)
    start = max(lo, kick) if kick is not None else lo
    labels = np.empty(hi - lo + 1, dtype="<U1")
    if start > lo:
        labels[:start - lo] = "D"
    if start <= hi:
        labels[start - lo:] = decode_segment(start, hi)
    bx = trk.get("bx")
    if bx is not None:
        half_bx = bx[lo:hi + 1] if len(bx) >= hi + 1 else (bx if len(bx) == len(labels) else None)
        _label_evidence_free_islands_dead(labels, half_bx)
    return labels


def decode(trk, e):
    """H/A/D inside each graded half with the minimum-stoppage decoder, X outside."""
    labels = np.full(len(e), "X", dtype="<U1")
    k = int(round(MIN_DEAD_S * trk["fps"]))
    for half, lo, hi in trk["blocks"]:
        labels[lo:hi + 1] = decode_half(
            trk, half, lo, hi,
            lambda s, h: LETTERS[_viterbi_min_dead(e[s:h + 1], k, SWITCH_COST, DEAD_COST)])
    return labels
