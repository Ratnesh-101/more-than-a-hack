"""Segment-level (semi-Markov) decoding: decode in possession units, not frames.

spellF1 scores whole possessions, so the decoder chooses a SEGMENTATION of each block into
alternating H / A / D segments, each with an explicit duration prior and a hard minimum length:

    score = sum over segments [ sum of frame emissions + log p_dur(d | state) - transition cost ]

solved exactly by dynamic programming over segment boundaries (hidden semi-Markov model). A frame
HMM can only express a geometric duration (most likely length = 1 frame); here a 1-s "possession"
is impossible by construction instead of merely penalised.

Speed: the DP runs on a coarse grid of CELL frames (segment boundaries on cell edges), then every
boundary is refined to the exact frame by a local search. Per block that is O(cells x max segment
cells) in vectorised numpy, so a full real match (2 x ~80k frames) decodes in seconds.

Emissions are the incumbent's unchanged (nearest-player softmax + slowness + ball out), so this
candidate isolates the effect of the decoder alone.
"""

import numpy as np

from mtah.decode import PARAMS as V1
from mtah.experiments.court import incumbent

NAME = "semi_markov"
PARAMS = {}
LETTERS = np.array(["H", "A", "D"])

CELL = 5                 # DP grid = 0.2 s at 25 fps; boundaries are refined to the frame after
SWITCH_COST = V1["switch_cost"]   # H<->A transition, the incumbent's own prior (not re-chosen here)
DEAD_COST = V1["dead_cost"]       # live<->dead transition, likewise the incumbent's prior
# Duration priors (seconds), lognormal(median, sigma), truncated to [min, max]:
LIVE_MEDIAN_S = 10.0     # possessions in elite football are typically 5-25 s; geometric middle ~10 s
LIVE_SIGMA = 0.8         # ln(25/10) ~ 0.9 and ln(10/5) ~ 0.7: the 5-25 s band is about +/- 1 sigma
LIVE_MIN_S = 2.0         # "controlled" possession needs a few seconds (CHALLENGE.md proxy); a shorter
                         # run is a touch/deflection, which keeps the previous team's possession
LIVE_MAX_S = 240.0       # a 4-min unbroken possession does not occur in a match; bounds the DP
DEAD_MEDIAN_S = 12.0     # throw-ins ~8 s, goal kicks / free kicks ~20-30 s; geometric middle ~12 s
DEAD_SIGMA = 1.0         # stoppages range from a quick throw-in to a goal celebration (5 s-90 s)
DEAD_MIN_S = 1.0         # same floor as the incumbent's min_dead_frames: a restart takes >= ~1 s
DEAD_MAX_S = 600.0       # treatment / VAR stoppages can run minutes; 10 min bounds the DP
DUR_WEIGHT = CELL        # emissions are summed over 25-fps frames that are near-duplicates within
                         # ~0.2 s; weighting the prior by CELL counts it as one look per cell, not
                         # one per frame (it is shape-only: 0 at the mode, so no extra segment price)


def emissions(trk):
    """(n, 3) log-likelihoods for H, A, D: the incumbent's emissions, unchanged."""
    return incumbent.emissions(trk)


def _log_dur(median_s, sigma, min_s, max_s, cell_s):
    """Shape-only log duration prior for k = 0..kmax cells: weighted lognormal log-mass minus its
    maximum (0 at the mode, negative elsewhere), -inf below the minimum."""
    kmax = int(round(max_s / cell_s))
    d = (np.arange(kmax + 1) + 0.5) * cell_s
    z = (np.log(d) - np.log(median_s)) / sigma
    logp = -0.5 * z * z - np.log(d * sigma * np.sqrt(2 * np.pi)) + np.log(cell_s)
    logp = DUR_WEIGHT * (logp - logp.max())
    logp[:int(np.ceil(min_s / cell_s))] = -np.inf
    return logp


def _segment_dp(cum, logdur, trans):
    """Exact semi-Markov Viterbi on the cell grid.

    cum: (3, m+1) prefix sums of cell emissions; logdur: per-state arrays log P(k cells);
    trans: (3, 3) log transition scores (row = from, column = to). The first and last segments are
    clipped by the block edge, so their duration prior is dropped but their minimum is kept.
    Returns a list of (state, start_cell, end_cell).
    """
    m = cum.shape[1] - 1
    ninf = -np.inf
    F = np.full((3, m + 1), ninf)        # best score of [0, t) with a segment of state s ending at t
    arg = np.zeros((3, m + 1), dtype=np.int64)    # its start cell
    prev = np.full((3, m + 1), -1, dtype=np.int64)  # state of the segment before it (-1: first)
    kmin = [int(np.argmax(np.isfinite(ld))) for ld in logdur]
    for t in range(1, m + 1):
        for s in range(3):
            lo_u, hi_u = max(0, t - len(logdur[s]) + 1), t - kmin[s]
            if hi_u < lo_u:
                if t < m:
                    continue
                lo_u = hi_u = 0                              # block shorter than the minimum
            u = np.arange(lo_u, hi_u + 1)
            G = F[:, u] + trans[:, s][:, None]               # enter s at u from another state
            src = np.argmax(G, axis=0)
            first = u == 0                                   # segment starts at the block edge
            g = np.where(first, 0.0, G[src, np.arange(len(u))])
            w = np.where(first | (t == m), 0.0, logdur[s][t - u])
            val = g + w + cum[s, t] - cum[s, u]
            j = int(np.argmax(val))
            F[s, t], arg[s, t] = val[j], u[j]
            prev[s, t] = -1 if first[j] else src[j]
    segs, s, t = [], int(np.argmax(F[:, m])), m
    while t > 0:
        u = int(arg[s, t])
        segs.append((s, u, t))
        s, t = int(prev[s, t]), u
    return segs[::-1]


def _refine(e, cut, a, b, lo, hi):
    """Frame of the a->b boundary in [lo, hi] that maximises emissions (a before, b after)."""
    cand = np.arange(lo, hi + 1)
    gain = np.concatenate(([0.0], np.cumsum(e[lo:hi, a] - e[lo:hi, b])))
    return int(cand[np.argmax(gain)]) if len(cand) else cut


def decode_block(e, fps):
    """H/A/D letters for one contiguous block of emissions e (n, 3)."""
    n = len(e)
    m = -(-n // CELL)
    pad = np.zeros((m * CELL, 3))
    pad[:n] = e
    cells = pad.reshape(m, CELL, 3).sum(axis=1)
    cum = np.concatenate([np.zeros((3, 1)), np.cumsum(cells.T, axis=1)], axis=1)
    cs = CELL / float(fps)
    live = _log_dur(LIVE_MEDIAN_S, LIVE_SIGMA, LIVE_MIN_S, LIVE_MAX_S, cs)
    dead = _log_dur(DEAD_MEDIAN_S, DEAD_SIGMA, DEAD_MIN_S, DEAD_MAX_S, cs)
    # Diagonal = chaining two segments of the same state. The output merges them into ONE run, so
    # it only caps the duration-prior penalty of a very long possession / stoppage at one
    # transition price instead of forcing a fake H<->A switch (or failing beyond the max length).
    trans = np.array([[-SWITCH_COST, -SWITCH_COST, -DEAD_COST],
                      [-SWITCH_COST, -SWITCH_COST, -DEAD_COST],
                      [-DEAD_COST, -DEAD_COST, -DEAD_COST]])
    segs = _segment_dp(cum, [live, live, dead], trans)
    out = np.empty(n, dtype=np.int8)
    starts = [0]
    for (sa, _, cut), (sb, _, _) in zip(segs, segs[1:]):
        c = cut * CELL                    # boundary moves at most half a cell either way
        starts.append(_refine(e, c, sa, sb, max(starts[-1] + 1, c - CELL // 2), min(n - 1, c + CELL // 2)))
    starts.append(n)
    for (s, _, _), lo, hi in zip(segs, starts, starts[1:]):
        out[lo:hi] = s
    return LETTERS[out]


def decode(trk, e):
    """Full-length labels: semi-Markov path inside each block, X outside."""
    labels = np.full(len(e), "X", dtype="<U1")
    for _, lo, hi in trk["blocks"]:
        labels[lo:hi + 1] = decode_block(e[lo:hi + 1], trk["fps"])
    return labels
