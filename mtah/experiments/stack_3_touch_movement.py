"""Stack 3 (judge): candidate 02 plus candidate 10's movement evidence delta.

Composition rule (b): delta = cand_10 emissions - incumbent emissions, kept on the H/A columns only
and reduced to its antisymmetric (H-vs-A) part so no live/dead mass leaks in; for 10 the delta is
exactly +-llr/2 already. Added to 02's emissions with weight 1; 02's costs (30 / 20) and the
default Viterbi. Nothing re-tuned.

Double-counting risk: low for proximity (02 does not use the incumbent's nearest-player softmax,
and 10's delta excludes it), but both terms carry possession through ball gaps (02 holds the last
toucher; 10 reads closing speed on an interpolated ball). 10's strength (NATS_PER_S = 10) was picked
on dev seeds against the incumbent's per-frame scale, not 02's (02 gives up to 3 nats per frame,
so 10's <= 0.4 nats per frame is relatively weaker here).
"""

from mtah.experiments import cand_02_touch_events as c02
from mtah.experiments import cand_10_movement as c10
from mtah.experiments.court import incumbent

NAME = "stack_3_touch+movement"
PARAMS = dict(c02.PARAMS)


def team_delta(e_cand, e_inc):
    """Antisymmetric H-vs-A part of (candidate - incumbent) on the team columns."""
    d = e_cand - e_inc
    return 0.5 * (d[:, 0] - d[:, 1])


def emissions(trk):
    e = c02.emissions(trk).copy()
    t = team_delta(c10.emissions(trk), incumbent.emissions(trk))
    e[:, 0] += t
    e[:, 1] -= t
    return e
