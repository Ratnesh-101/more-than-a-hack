"""Multi-cue dead-ball detection (cand_08): a logit that sums football cues of a stoppage.

  - players walking: ROBUST statistics over known-team slots (fraction below walking pace, count
    of high-speed runners), smoothed over 1 s, so one ghost slot or one sprinter cannot flip it;
  - ball stationary, ball beyond the lines, still ball on a restart spot (corner arc, touchline,
    goal area, centre spot) -- each must PERSIST for most of 2 s of ball sightings, because a
    lofted ball projects beyond the lines and a shielded ball pauses briefly in live play;
  - kickoff geometry: both teams entirely in their own halves (usable with the ball missing);
  - live counter-evidence: ball travelling at pass speed inside the pitch.
"""

import warnings

import numpy as np

OUT_MARGIN_M = 0.5    # ball-tracking error near a line; same margin as v1
WALK_MPS = 2.0        # walking pace tops out near 2 m/s (7 km/h); above it a player is jogging
HSR_MPS = 5.5         # high-speed-running threshold of football physical reports (19.8 km/h)
SLOW_FRAC = 0.9       # stoppage = at most ~2 of 20 outfielders above walking pace (restarter, a runner)
SLOW_WIDTH = 0.03     # about one player of softness around that count
SMOOTH_S = 1.0        # player cues averaged over 1 s: a stoppage is a state, not a frame
SPEED_LAG = 2         # frames each side for a position-derived speed where the tracker gives none
BALL_WIN_S = 1.0      # ball displacement measured over 1 s, long enough to ignore jitter
BALL_STILL_MPS = 1.5  # slower than walking: ball held, placed or carried to the spot
BALL_PASS_MPS = 8.0   # slowest typical ground pass; nobody kicks the ball this hard in a stoppage
PERSIST_S = 2.0       # longer than a lofted ball's flight beyond the line or a shielding pause
PERSIST_FRAC = 0.75   # "most of" the sightings in that window; tolerates a glitch or two
SPOT_R_M = 2.0        # corner arc (1 m) / centre spot plus ball-tracking error
LINE_M = 1.0          # a throw-in is taken on or behind the touchline
GOAL_AREA = (6.0, 10.0)  # goal area is 5.5 m deep and 18.32 m wide, plus error
OWN_HALF_FRAC = 0.95  # kickoff law: every player in his own half (allow one mis-tracked slot)

# Log-odds contributions to "dead"; DEAD_BIAS is the prior of an ordinary frame.
DEAD_BIAS = -2.5      # v1's prior log-odds of dead ball on a normal frame, kept (not re-tuned)
W_SLOW = 3.0          # nearly everyone walking (v1's dead_slow_bonus, now on a robust statistic)
W_HSR = -2.0          # per high-speed runner (capped at 2): sprinting means live play
W_STILL = 1.5         # ball motionless for 2 s: rare but possible live (keeper holding it)
W_OUT = 5.0           # ball beyond the lines (v1's dead_out_bonus)
W_SPOT = 2.0          # still ball on a restart spot: the set piece is being prepared
W_KICKOFF = 3.0       # teams split exactly by the halfway line happens only at a kickoff
W_PASS = -2.0         # ball moving at pass speed inside the pitch: play is live


def _smooth(x, valid, k):
    """Centred moving average over k frames using only valid frames; NaN where none is valid."""
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
    m = np.concatenate([[0.0], np.cumsum(valid.astype(float))])
    lo = np.clip(np.arange(n) - k // 2, 0, n)
    hi = np.clip(np.arange(n) + k // 2 + 1, 0, n)
    cnt = m[hi] - m[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _persist(cue, vis, k):
    """True where the cue holds on >= PERSIST_FRAC of the ball sightings in a k-frame window."""
    return np.nan_to_num(_smooth(cue.astype(float), vis, k), nan=0.0) >= PERSIST_FRAC


def _speeds(trk):
    """Player speeds: the tracker's own, else a central difference of positions."""
    px, py, ps, fps = trk["px"], trk["py"], trk["ps"], trk["fps"]
    k = SPEED_LAG
    der = np.full(ps.shape, np.nan, dtype=np.float32)
    if len(ps) > 2 * k:
        der[k:-k] = np.hypot(px[2 * k:] - px[:-2 * k], py[2 * k:] - py[:-2 * k]) * (fps / (2.0 * k))
    return np.where(np.isnan(ps), der, ps)


def dead_logit(trk):
    """Per-frame log-odds that the ball is dead, from players, ball and restart geometry."""
    team, fps = trk["team"], trk["fps"]
    bx, by = trk["bx"].astype(float), trk["by"].astype(float)
    length, width = trk["pitch"]
    n = len(bx)
    ps = _speeds(trk)
    real = ~np.isnan(ps) & (team >= 0)[None, :]           # ghost / unknown-team slots excluded
    n_real = real.sum(axis=1)
    has = n_real > 0
    frac_walk = (real & (ps < WALK_MPS)).sum(axis=1) / np.maximum(n_real, 1)
    n_hsr = np.minimum((real & (ps > HSR_MPS)).sum(axis=1), 2).astype(float)
    k = int(round(SMOOTH_S * fps))
    frac_walk = np.nan_to_num(_smooth(frac_walk, has, k), nan=0.0)
    n_hsr = np.nan_to_num(_smooth(n_hsr, has, k), nan=0.0)
    slow = 1.0 / (1.0 + np.exp(-(frac_walk - SLOW_FRAC) / SLOW_WIDTH))

    vis = ~np.isnan(bx)
    h = int(round(BALL_WIN_S * fps / 2))
    sx = np.full(n, np.nan)
    if n > 2 * h:
        sx[h:n - h] = np.hypot(bx[2 * h:] - bx[:n - 2 * h], by[2 * h:] - by[:n - 2 * h]) / BALL_WIN_S
    with np.errstate(invalid="ignore"):
        still = vis & (sx < BALL_STILL_MPS)
        moving = sx > BALL_PASS_MPS
    out = vis & ((bx < -OUT_MARGIN_M) | (bx > length + OUT_MARGIN_M)
                 | (by < -OUT_MARGIN_M) | (by > width + OUT_MARGIN_M))
    passing = vis & ~out & moving
    ex, ey = np.minimum(bx, length - bx), np.minimum(by, width - by)
    centre = np.hypot(bx - length / 2, by - width / 2) < SPOT_R_M
    spot = still & ((np.hypot(ex, ey) < SPOT_R_M) | (ey < LINE_M) | centre
                    | ((ex < GOAL_AREA[0]) & (np.abs(by - width / 2) < GOAL_AREA[1])))
    kp = int(round(PERSIST_S * fps))
    still, out, spot = (_persist(c, vis, kp) for c in (still, out, spot))

    # kickoff: the half each team defends from its mean x, then the share of players inside it
    px = np.where(real, trk["px"], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # frames with no player of a team
        mx = [np.nanmean(np.where(team[None, :] == s, px, np.nan), axis=1) for s in (0, 1)]
    home_left = np.nan_to_num(mx[0] - mx[1], nan=0.0) < 0
    left = px < length / 2
    own = np.where(team[None, :] == 0, left == home_left[:, None], left != home_left[:, None])
    frac_own = (real & own).sum(axis=1) / np.maximum(n_real, 1)
    kickoff = has & (frac_own >= OWN_HALF_FRAC) & (~vis | centre)

    return (DEAD_BIAS + W_SLOW * slow + W_HSR * n_hsr + W_STILL * still + W_OUT * out
            + W_SPOT * spot + W_KICKOFF * kickoff + W_PASS * passing)
