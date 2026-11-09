"""Local numerical superiority and pressure around the ball.

The nearest-player cue (the incumbent's softmax log-odds) is weighted by how ISOLATED that
player is: opponents of the nearest player's team are counted in rings around the ball, closer
ones weighted more, and the cue is divided by (1 + weighted pressure). A lone player on the ball
keeps the full cue; a nearest player wrapped in opponents is a contested ball (duel, scramble,
tackle in progress), so his cue fades toward a floor and the HMM carries the previous team, as
CHALLENGE.md asks ("contested or loose -> stays with the last team").

Dead-ball evidence is the incumbent's, unchanged; default decoder and costs.
Measured and rejected (see the brief): a signed own-minus-opponent density difference as extra
evidence. Near the ball the DEFENDING team outnumbers (it presses), so raw local superiority
points the wrong way and cost 0.01-0.03 mean spellF1 over dev seeds 1-5.
"""

import numpy as np

from mtah.decode import PARAMS as V1
from mtah.experiments.court import incumbent

NAME = "07_local_superiority"
PARAMS = {}

# (inner m, outer m, weight) rings of opponents around the ball that count as pressure
PRESS_RINGS = ((0.0, 1.5, 1.0),   # tackle reach: one stride plus a leg, the challenge is on
               (1.5, 3.0, 0.5))   # closing down: can arrive in ~0.5 s at sprint speed
ISO_FLOOR = 0.3   # even heavily pressed, the man with the ball at his feet usually keeps it
CHUNK = 20000     # frames per vectorised slab: bounds memory to a few (CHUNK, 46) arrays


def _team_llr(px, py, bx, by, team):
    """Log-odds H vs A per frame of one slab (0 where the ball or both teams are unseen)."""
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    dh = np.where(team[None, :] == 0, d, np.inf)
    da = np.where(team[None, :] == 1, d, np.inf)
    h_min, a_min = dh.min(axis=1), da.min(axis=1)
    seen = np.isfinite(h_min) | np.isfinite(a_min)
    h_min, a_min = np.minimum(h_min, 99.0), np.minimum(a_min, 99.0)

    near = (a_min - h_min) / V1["team_scale_m"]
    # pressure on the nearest player = weighted count of OPPONENTS of his team near the ball
    opp = np.where((h_min <= a_min)[:, None], da, dh)
    press = np.zeros(len(bx))
    for lo, hi, w in PRESS_RINGS:
        press += w * ((opp >= lo) & (opp < hi)).sum(axis=1)
    iso = ISO_FLOOR + (1.0 - ISO_FLOOR) / (1.0 + press)
    conf = np.exp(-np.minimum(h_min, a_min) / V1["control_radius_m"]) * iso
    return np.where(seen & ~np.isnan(bx), conf * near, 0.0)


def emissions(trk):
    """(n, 3) float64 log-likelihoods for H, A, D."""
    n = len(trk["bx"])
    llr = np.empty(n)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        llr[a:b] = _team_llr(trk["px"][a:b], trk["py"][a:b], trk["bx"][a:b], trk["by"][a:b],
                             trk["team"])
    e_dead = incumbent.emissions(trk)[:, 2]
    live = np.log1p(-np.exp(e_dead))
    e = np.empty((n, 3), dtype=np.float64)
    e[:, 0] = live - np.logaddexp(0.0, -llr)   # log sigmoid(llr)
    e[:, 1] = live - np.logaddexp(0.0, llr)
    e[:, 2] = e_dead
    return e
