"""Stack 2 (judge): candidate 02's team evidence with candidate 08's dead-ball model and decoder.

Composition rule (c): 02's emissions are log P(team | live) + log P(live); P(live) is recovered
exactly as logaddexp(H, A) and removed, leaving 02's conditional team log-probabilities. The live /
dead split is replaced by 08's dead_logit (robust walking share, high-speed runners, persistent
ball cues, kickoff geometry), and the path is 08's minimum-stoppage (3 s) Viterbi with v1 costs
30 / 12. Nothing re-tuned.

Double-counting risk: none on the dead column (one dead model replaces the other), but the stack
LOSES 02's own still-ball / fresh-kick dead cues, which advocate 02 credits with about half of its
gain; this stack is the direct test of whether 08's dead model is the better of the two.
"""

import numpy as np

from mtah.experiments import cand_02_touch_events as c02
from mtah.experiments import cand_08_dead_ball as c08

NAME = "stack_2_touch+deadball08"
PARAMS = {}          # 08's decoder uses v1 costs; nothing overridden
EPS = c08.EPS


def emissions(trk):
    e02 = c02.emissions(trk)
    log_live02 = np.logaddexp(e02[:, 0], e02[:, 1])
    team_h, team_a = e02[:, 0] - log_live02, e02[:, 1] - log_live02
    p_dead = np.clip(1.0 / (1.0 + np.exp(-c08.dead_logit(trk))), EPS, 1 - EPS)
    return np.stack([team_h + np.log1p(-p_dead), team_a + np.log1p(-p_dead), np.log(p_dead)], axis=1)


def decode(trk, e):
    return c08.decode(trk, e)
