"""Collective movement: the team without the ball converges on it; the team with it does not.

Evidence per frame = incumbent nearest-player softmax (where the ball is seen) + a slow movement
log-odds: the difference between the two teams' mean CLOSING SPEED to the ball (the rate at which
the player-ball distance shrinks, all players of a team averaged). Defenders press, cover and shift
toward the ball; attackers stay level with it or run away from it to offer passing lanes. The read
is smoothed over a couple of seconds and read slightly ahead (teams react after the hand-over), so
it is a soft confirmation that carries possession through ball gaps and resists a brief proximity
flip; it is bounded (tanh) so only ~3 s of saturated team-wide press can switch possession on its
own, which is what a turnover looks like while the ball is hidden. Dead-ball evidence: incumbent's.
"""

import numpy as np

from mtah.decode import PARAMS as V1
from mtah.experiments.court import incumbent

NAME = "10_movement"
PARAMS = {}

FPS = 25             # frame rate fixed by the Trk contract (trk['fps'])
HALF_DIFF = 5        # central difference over +-0.2 s: long enough to beat 0.15 m position jitter
MAX_SPEED = 12.0     # m/s; faster than any sprinter -> a slot swap or ball glitch, not movement
UNIT = 1.0           # m/s; a team-wide closing gap of a walking pace = one unit of (tanh) evidence
LAG_S = 0.5          # s; ~0.3 s reaction + ~0.2 s to accelerate; the window then spans t-0.5..t+1.5 s
WIN_S = 2.0          # s; a press / recovery run lasts a couple of seconds -> average over it
NATS_PER_S = 10.0    # max evidence per s: ~3 s of a saturated team-wide press pays one switch_cost


def _interp_ball(trk):
    """Ball (float32) linearly interpolated across every gap inside each block, NaN elsewhere.

    No gap limit: the line between the last and next sighting is still the best estimate of where
    play is, and a team converging on the real ball converges on that line's end point."""
    bx, by = np.full(len(trk["bx"]), np.nan, np.float32), np.full(len(trk["by"]), np.nan, np.float32)
    for _, lo, hi in trk["blocks"]:
        x, y = trk["bx"][lo:hi + 1], trk["by"][lo:hi + 1]
        seen = np.flatnonzero(~np.isnan(x))
        if len(seen):
            t = np.arange(seen[0], seen[-1] + 1)
            bx[lo + t], by[lo + t] = np.interp(t, seen, x[seen]), np.interp(t, seen, y[seen])
    return bx, by


def _diff(a):
    """Central time derivative per second over +-HALF_DIFF frames; NaN at the edges / gaps."""
    k = HALF_DIFF
    out = np.full(a.shape, np.nan, np.float32)
    out[k:-k] = (a[2 * k:] - a[:-2 * k]) * (FPS / (2.0 * k))
    out[np.abs(out) > MAX_SPEED] = np.nan
    return out


def _team_mean(val, mask):
    """Per-frame mean of `val` over the finite entries of the slots in `mask`; NaN if none."""
    v = val[:, mask]
    ok = np.isfinite(v)
    cnt = ok.sum(axis=1)
    tot = np.where(ok, v, 0.0).sum(axis=1, dtype=np.float64)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _smooth(x, n):
    """Centred moving average of length n that ignores NaN (NaN only where the window is empty)."""
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    h = n // 2
    lo = np.clip(np.arange(len(x)) - h, 0, len(x))
    hi = np.clip(np.arange(len(x)) + h + 1, 0, len(x))
    cnt = k[hi] - k[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def movement_score(trk):
    """Per-frame closing-speed difference (m/s): > 0 = away converges on the ball = home has it."""
    bx, by = _interp_ball(trk)
    closing = -_diff(np.hypot(trk["px"] - bx[:, None], trk["py"] - by[:, None]))
    home, away = trk["team"] == 0, trk["team"] == 1
    return _team_mean(closing, away) - _team_mean(closing, home)


def emissions(trk):
    """(n, 3) float64: incumbent evidence + lagged, smoothed, bounded movement log-odds."""
    e = incumbent.emissions(trk, V1)
    s = movement_score(trk)
    lag = int(LAG_S * FPS)
    ahead = np.concatenate((s[lag:], np.full(lag, np.nan)))
    llr = np.nan_to_num(np.tanh(_smooth(ahead, int(WIN_S * FPS)) / UNIT)) * NATS_PER_S / FPS
    e[:, 0] += llr / 2.0
    e[:, 1] -= llr / 2.0
    return e
