"""Pass-chain control: possession flips only when the opponent CONFIRMS control.

1. Touch frames: ball seen, a team's nearest player within TOUCH_M of it and clearly nearer than
   the other team's (MARGIN_M). Runs shorter than MIN_TOUCH_S are dropped (fly-bys).
2. Touch events: consecutive same-team touch runs are one event unless the ball moved at least
   PASS_MIN_M between them; then the later run is a new reception (a completed pass).
3. Control machine: the current team keeps possession through loose balls, ball-missing gaps and
   isolated opponent touches; any own-team touch cancels a pending take-over. An opponent takes
   over after CHAIN receptions in a row (a pass between two of its players) or one event held
   HOLD_S. Decoding is offline, so the take-over is back-dated to the opponent's first reception
   in that chain; if the ball was hidden just before it, to the middle of that hidden gap (the
   hand-over happened somewhere inside it; the midpoint is the neutral guess).
4. Emissions: the machine's team keeps the live log-probability, the other team gets -TEAM_LL per
   frame; D is the incumbent's dead-ball model (ball out + everyone slow). Default Viterbi and
   default costs (switch_cost 30 also drops machine runs shorter than ~1 s).
"""

import numpy as np

from mtah.experiments.court import incumbent

NAME = "03_pass_chain"
PARAMS = {}

TOUCH_M = 1.5        # ball at a player's feet: ~0.7 m stride offset + ~0.5 m tracking noise
MARGIN_M = 0.5       # nearest team must be clearly nearer, else it is a duel, not control
MIN_TOUCH_S = 0.12   # 3 frames: shorter proximity is a ball flying past, not a contact
PASS_MIN_M = 5.0     # shortest deliberate pass; ball moving less is the same player dribbling
CHAIN = 2            # two same-team receptions in a row = a completed pass = control
HOLD_S = 0.5         # one player keeping the ball 0.5 s (first touch + settle) is control
TEAM_LL = 1.0        # per-frame log-odds for the machine's team; with switch_cost 30 a flip
                     # needs ~30 frames (1.2 s) of machine evidence, the shortest real spell


def _runs(x):
    """(value, start, stop) for maximal runs of equal values in a 1-d int array."""
    cut = np.flatnonzero(np.diff(x)) + 1
    return x[np.r_[0, cut]], np.r_[0, cut], np.r_[cut, len(x)]


def _touch_team(trk):
    """Per frame: 0 / 1 = that team is touching the ball, -1 = nobody clearly is."""
    vis = ~np.isnan(trk["bx"])
    d = np.hypot(trk["px"] - trk["bx"][:, None], trk["py"] - trk["by"][:, None])
    d[np.isnan(d)] = np.inf
    with np.errstate(invalid="ignore"):
        d_h = np.where(vis, np.where(trk["team"] == 0, d, np.inf).min(axis=1), np.inf)
        d_a = np.where(vis, np.where(trk["team"] == 1, d, np.inf).min(axis=1), np.inf)
        home = (d_h < TOUCH_M) & (d_a - d_h > MARGIN_M)
        away = (d_a < TOUCH_M) & (d_h - d_a > MARGIN_M)
    return np.where(home, 0, np.where(away, 1, -1)).astype(np.int8), vis


def _events(tt, bx, by, fps):
    """Touch events [team, start, stop] in one block (stop exclusive)."""
    ev = []
    for team, a, b in zip(*_runs(tt)):
        if team < 0 or b - a < MIN_TOUCH_S * fps:
            continue
        last = ev[-1][2] - 1 if ev else 0
        if ev and ev[-1][0] == team and np.hypot(bx[a] - bx[last], by[a] - by[last]) < PASS_MIN_M:
            ev[-1][2] = int(b)  # same player still on the ball
        else:
            ev.append([int(team), int(a), int(b)])
    return ev


def _backdate(a, vis, last_own):
    """Take-over frame: the reception, or the middle of a hidden-ball gap just before it."""
    j = a
    while j > last_own and not vis[j - 1]:
        j -= 1
    return (j + a) // 2


def _machine(ev, vis, fps):
    """Per-frame controlling team (-1 before the first touch) for one block."""
    state = np.full(len(vis), -1, np.int8)
    cur, last_own, pend = -1, 0, None
    changes = []
    for team, a, b in ev:
        if cur < 0 or team == cur:
            if cur < 0:
                changes.append((a, team))
            cur, last_own, pend = team, b, None
            continue
        pend = [a, 0] if pend is None else pend
        pend[1] += 1
        if pend[1] >= CHAIN or b - a >= HOLD_S * fps:
            changes.append((_backdate(pend[0], vis, last_own), team))
            cur, last_own, pend = team, b, None
    for k, (a, team) in enumerate(changes):
        state[a:changes[k + 1][0] if k + 1 < len(changes) else len(vis)] = team
    return state


def emissions(trk):
    """(n, 3) float64 log-likelihoods for H, A, D."""
    e = incumbent.emissions(trk)
    p_live = np.log1p(-np.exp(e[:, 2]))
    tt, vis = _touch_team(trk)
    state = np.full(len(tt), -1, np.int8)
    for _, lo, hi in trk["blocks"]:
        s = slice(lo, hi + 1)
        ev = _events(tt[s], trk["bx"][s], trk["by"][s], trk["fps"])
        state[s] = _machine(ev, vis[s], trk["fps"])
    out = np.empty_like(e)
    out[:, 0] = p_live - TEAM_LL * (state == 1)
    out[:, 1] = p_live - TEAM_LL * (state == 0)
    out[:, 2] = e[:, 2]
    return out
