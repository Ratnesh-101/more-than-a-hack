"""Stack 1 (judge): candidate 02 touch-event emissions decoded by candidate 12's semi-Markov decoder.

Composition rule (a): 02's emissions unchanged, 12's segment DP unchanged (same duration priors,
CELL grid and boundary refinement, reused from cand_12), with the transition costs of the emission
model it decodes: 02's switch_cost 30 / dead_cost 20. Those costs were set a priori by advocate 02
(dead_cost > switch_cost / 2 so a hand-over is never routed through a fake stoppage), which also
closes the cheap H->D->A route that advocate 12 disclosed with 2 x 12 < 30. Nothing re-tuned.

Double-counting risk: 02's switch_cost already demands ~1.5 s of touching to flip; 12 adds a hard
2-s live minimum and a duration prior. Both suppress short possessions, so the stack can
under-segment (02 already predicts 55 possessions against 66 true on dev seeds).
"""

import numpy as np

from mtah.experiments import cand_02_touch_events as c02
from mtah.experiments import cand_12_semi_markov as c12

NAME = "stack_1_touch+semimarkov"
PARAMS = dict(c02.PARAMS)
SWITCH_COST = float(c02.PARAMS["switch_cost"])
DEAD_COST = float(c02.PARAMS["dead_cost"])


def emissions(trk):
    """02's emissions, unchanged."""
    return c02.emissions(trk)


def _decode_block(e, fps):
    """cand_12.decode_block with the transition costs taken from 02 (everything else 12's)."""
    n = len(e)
    cell = c12.CELL
    m = -(-n // cell)
    pad = np.zeros((m * cell, 3))
    pad[:n] = e
    cells = pad.reshape(m, cell, 3).sum(axis=1)
    cum = np.concatenate([np.zeros((3, 1)), np.cumsum(cells.T, axis=1)], axis=1)
    cs = cell / float(fps)
    live = c12._log_dur(c12.LIVE_MEDIAN_S, c12.LIVE_SIGMA, c12.LIVE_MIN_S, c12.LIVE_MAX_S, cs)
    dead = c12._log_dur(c12.DEAD_MEDIAN_S, c12.DEAD_SIGMA, c12.DEAD_MIN_S, c12.DEAD_MAX_S, cs)
    trans = np.array([[-SWITCH_COST, -SWITCH_COST, -DEAD_COST],
                      [-SWITCH_COST, -SWITCH_COST, -DEAD_COST],
                      [-DEAD_COST, -DEAD_COST, -DEAD_COST]])
    segs = c12._segment_dp(cum, [live, live, dead], trans)
    out = np.empty(n, dtype=np.int8)
    starts = [0]
    for (sa, _, cut), (sb, _, _) in zip(segs, segs[1:]):
        c = cut * cell
        starts.append(c12._refine(e, c, sa, sb, max(starts[-1] + 1, c - cell // 2),
                                  min(n - 1, c + cell // 2)))
    starts.append(n)
    for (s, _, _), lo, hi in zip(segs, starts, starts[1:]):
        out[lo:hi] = s
    return c12.LETTERS[out]


def decode(trk, e):
    labels = np.full(len(e), "X", dtype="<U1")
    for _, lo, hi in trk["blocks"]:
        labels[lo:hi + 1] = _decode_block(e[lo:hi + 1], trk["fps"])
    return labels
