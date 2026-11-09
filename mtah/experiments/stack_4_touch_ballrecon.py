"""Stack 4 (judge): candidate 02 plus candidate 05's ball-reconstruction evidence delta.

Composition rule (b): delta = cand_05 emissions - incumbent emissions on the H/A columns,
antisymmetric (H-vs-A) part only, so 05's clean-sample out cue (a live/dead change) does not leak
in. On ball-missing frames that delta is the bridged-ball proximity evidence (the incumbent has
none there); on seen frames it is the small blur correction. Added to 02 with weight 1; 02's costs
and default Viterbi. Nothing re-tuned.

Double-counting risk: 02's carry rule is itself close-range proximity; 05's delta adds proximity
only where 02 has no observation (gaps), but both mechanisms try to place a hand-over hidden in a
gap (02 by midpoint backdating, 05 by the bridged ball), so the two can push the boundary twice.
"""

from mtah.experiments import cand_02_touch_events as c02
from mtah.experiments import cand_05_ball_reconstruction as c05
from mtah.experiments.court import incumbent

NAME = "stack_4_touch+ballrecon"
PARAMS = dict(c02.PARAMS)


def emissions(trk):
    e = c02.emissions(trk).copy()
    d = c05.emissions(trk) - incumbent.emissions(trk)
    t = 0.5 * (d[:, 0] - d[:, 1])
    e[:, 0] += t
    e[:, 1] -= t
    return e
