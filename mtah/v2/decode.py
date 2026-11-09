"""Emissions and decoder of the champion (stack_2).

Emissions: team log-probabilities from touch events (touch.py) given "live", multiplied by the
live probability from the dead-ball logit (deadball.py). Decoder: Viterbi over H, A, D where every
dead run lasts at least MIN_DEAD_S, solved exactly by a semi-Markov D chain in O(n) scalar steps.
"""

import numpy as np

from mtah.v2 import deadball, touch

SWITCH_COST = 30.0    # ~1.5 s of new-team touching flips possession; a deflection (flip + back) costs 60
DEAD_COST = 12.0      # v1's live <-> dead penalty, untouched
MIN_DEAD_S = 3.0      # fastest real restarts (quick throw-in / free kick) take about 3 s
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
    chain reaching DK at t scores D1(t-k+1) + dead emissions in between (prefix sums).
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
    path = np.empty(n, dtype=np.int8)
    s, t = int(np.argmax([H, A, DK])), n - 1
    while t >= 0:
        path[t] = s
        if s == 2 and bdk[t] == 3:                 # forced chain: frames t-k+1 .. t all dead
            start = t - k + 1
            path[start:t] = 2
            s, t = b1[start], start - 1
        else:
            s, t = (bh, ba, bdk)[s][t], t - 1
    return path


def decode(trk, e):
    """H/A/D inside each graded half with the minimum-stoppage decoder, X outside."""
    labels = np.full(len(e), "X", dtype="<U1")
    k = int(round(MIN_DEAD_S * trk["fps"]))
    for _, lo, hi in trk["blocks"]:
        labels[lo:hi + 1] = LETTERS[_viterbi_min_dead(e[lo:hi + 1], k, SWITCH_COST, DEAD_COST)]
    return labels
