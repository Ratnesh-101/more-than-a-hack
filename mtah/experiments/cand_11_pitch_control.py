"""Pitch control at the ball: team evidence from who can GET to the ball first, not who is nearest.

For every tracked player we estimate the time to reach the ball (Spearman 2017 / Fernandez &
Bornn 2018 style): during a reaction time the player keeps his current velocity, then he
accelerates straight at the ball (bounded acceleration, bounded top speed). Each team's control
time is its earliest arrival. The home-control probability is a logistic in the arrival-time
difference (Spearman's control-time uncertainty). The evidence fades towards 50/50 when even the
first arrival is late (loose ball: possession stays with the last team, as CHALLENGE.md says) and
is zero when the ball is not tracked. Dead-ball evidence is the v1 model, unchanged, so this
candidate isolates one question: is time-to-intercept a better possession cue than distance?
"""

import numpy as np

from mtah.decode import PARAMS as V1

NAME = "pitch_control"
PARAMS = {}

VEL_HALF_WINDOW = 5      # central difference over +/-5 frames (0.4 s): a direction, not tracker jitter
REACTION_S = 0.25        # human visual reaction to a ball event ~0.2-0.3 s (sport-science prior)
V_MAX = 7.5              # pro sprint top speed ~7-8 m/s (match tracking studies)
A_MAX = 3.5              # typical maximal acceleration of footballers ~3-4 m/s^2
REACH_M = 1.0            # a player plays a ball within about one stride / leg reach without running
MAX_BALL_SPEED = 35.0    # hardest shots ~35 m/s: a faster finite difference is a tracking error
MAX_PLAYER_SPEED = 12.0  # nobody runs faster (Bolt ~12.4 m/s): faster finite differences are id swaps
CONTROL_SIGMA_S = 0.45   # Spearman 2017 control-time uncertainty; logistic scale = sigma*sqrt(3)/pi
OUT_MARGIN_M = 0.5       # same tolerance as v1 for the ball being over a line
CHUNK = 20000            # frames per block of work: keeps a 199k x 46 match well under 1 GB
EPS = 1e-6               # numerical floor for divisions and probabilities, no modelling role
NO_ARRIVAL_S = 99.0      # "this team has nobody tracked": arrives effectively never


def _velocity(p, fps):
    """(n, S) float32 central-difference velocity in m/s; 0 where either end is missing."""
    k, n = VEL_HALF_WINDOW, len(p)
    v = np.zeros_like(p)
    if n > 2 * k:
        v[k:n - k] = (p[2 * k:] - p[:n - 2 * k]) * (fps / (2.0 * k))
    return np.where(np.isfinite(v), v, 0.0).astype(np.float32)


def _arrival(px, py, vx, vy, bx, by, bvx, bvy):
    """(m, S) seconds for each player to reach the ball, worked in the ball's moving frame (the
    ball keeps its velocity): both drift for REACTION_S, then the player closes the remaining gap
    beyond his reach from closing speed u, accelerating at A_MAX up to V_MAX. A ball travelling
    towards him faster than V_MAX simply arrives at the closing speed. A player who already has
    the ball within reach is not reacting to anything: his time is just the step to it."""
    now = np.hypot(bx[:, None] - px, by[:, None] - py)
    wx, wy = vx - bvx[:, None], vy - bvy[:, None]   # velocity relative to the ball
    dx, dy = bx[:, None] - (px + wx * REACTION_S), by[:, None] - (py + wy * REACTION_S)
    dist = np.hypot(dx, dy)
    d = np.maximum(dist - REACH_M, 0.0)
    u = (wx * dx + wy * dy) / np.maximum(dist, EPS)  # closing speed along the line to the ball
    t_top = np.maximum(V_MAX - u, 0.0) / A_MAX       # time to reach top speed along the line
    d_top = u * t_top + 0.5 * A_MAX * t_top ** 2     # distance covered while accelerating
    t_acc = (np.sqrt(np.maximum(u * u + 2 * A_MAX * d, 0.0)) - u) / A_MAX
    t_run = np.where(d <= d_top, t_acc, t_top + (d - d_top) / np.maximum(V_MAX, u))
    return np.where(now <= REACH_M, now / V_MAX, REACTION_S + t_run)


def _team_times(trk):
    """(n,) earliest arrival time (s) of home and away; NO_ARRIVAL_S where unknown."""
    px, py, team = trk["px"], trk["py"], trk["team"]
    bx, by = trk["bx"], trk["by"]
    vx, vy = _velocity(px, trk["fps"]), _velocity(py, trk["fps"])
    fast = np.hypot(vx, vy) > MAX_PLAYER_SPEED
    vx, vy = np.where(fast, 0.0, vx), np.where(fast, 0.0, vy)
    bvx, bvy = (_velocity(b[:, None], trk["fps"])[:, 0] for b in (bx, by))
    wild = np.hypot(bvx, bvy) > MAX_BALL_SPEED
    bvx, bvy = np.where(wild, 0.0, bvx), np.where(wild, 0.0, bvy)
    n = len(bx)
    t_h, t_a = np.full(n, NO_ARRIVAL_S), np.full(n, NO_ARRIVAL_S)
    for lo in range(0, n, CHUNK):
        sl = slice(lo, min(n, lo + CHUNK))
        t = _arrival(px[sl], py[sl], vx[sl], vy[sl], bx[sl], by[sl], bvx[sl], bvy[sl])
        t = np.where(np.isfinite(t), t, np.inf)
        for side, out in ((0, t_h), (1, t_a)):
            m = t[:, team == side].min(axis=1, initial=np.inf)
            out[sl] = np.where(np.isfinite(m), m, NO_ARRIVAL_S)
    return t_h, t_a


def _dead_prob(trk, p=V1):
    """v1 dead-ball probability (ball out of the lines, everybody slow), rebuilt from trk."""
    bx, by = trk["bx"], trk["by"]
    length, width = trk["pitch"]
    present = ~np.isnan(trk["ps"])
    mean_speed = np.where(present.any(axis=1),
                          np.where(present, trk["ps"], 0.0).sum(axis=1) / np.maximum(present.sum(axis=1), 1),
                          p["slow_speed"])
    slow = 1.0 / (1.0 + np.exp(-(p["slow_speed"] - mean_speed) / p["slow_width"]))
    with np.errstate(invalid="ignore"):
        out = (bx < -OUT_MARGIN_M) | (bx > length + OUT_MARGIN_M) | (by < -OUT_MARGIN_M) | (by > width + OUT_MARGIN_M)
    z = p["dead_bias"] + p["dead_out_bonus"] * out + p["dead_slow_bonus"] * slow
    return np.clip(1.0 / (1.0 + np.exp(-z)), EPS, 1 - EPS)


def emissions(trk):
    """(n, 3) float64 log-likelihoods for H, A, D."""
    vis = ~np.isnan(trk["bx"])
    t_h, t_a = _team_times(trk)
    scale = CONTROL_SIGMA_S * np.sqrt(3.0) / np.pi
    z = (t_a - t_h) / scale                         # > 0: home gets there first
    log_ph = -np.logaddexp(0.0, -z)
    log_pa = -np.logaddexp(0.0, z)
    first = np.minimum(t_h, t_a) - REACTION_S
    # a ball nobody can play within about one more reaction time is loose: evidence fades to flat
    conf = np.where(vis, np.exp(-np.maximum(first, 0.0) / REACTION_S), 0.0)
    p_dead = _dead_prob(trk)
    e = np.empty((len(vis), 3), dtype=np.float64)
    e[:, 0] = conf * log_ph + (1 - conf) * np.log(0.5) + np.log(1 - p_dead)
    e[:, 1] = conf * log_pa + (1 - conf) * np.log(0.5) + np.log(1 - p_dead)
    e[:, 2] = np.log(p_dead)
    return e
