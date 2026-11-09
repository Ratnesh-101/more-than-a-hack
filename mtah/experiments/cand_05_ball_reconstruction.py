"""Ball reconstruction: glitch rejection and gap bridging with an explicit, calibrated position
uncertainty that fades team evidence instead of inventing it.

Pipeline (vectorised; one short Python loop over the decoded blocks only):
  1. Glitch rejection. A visible ball sample that sits further from the median of its +-0.4 s
     neighbourhood than a ball can travel in 0.4 s is a tracker glitch and is dropped.
  2. Gap bridge. In a gap the ball estimate is the straight line between the last and next clean
     samples (one-sided hold at block edges). Its variance follows a random-velocity ball: it grows
     like (v t)^2 for short gaps (the ball keeps its direction) and like 2 v^2 tau t for long ones
     (passes and carries decorrelate the direction), evaluated at the bridge time t0*t1/(t0+t1).
  3. Emissions. Nearest-player softmax as in v1, but each team's distance is the expected distance
     to the uncertain ball, sqrt(d^2 + var), and confidence decays with it, so a ball known only to
     ~10 m gives almost flat H/A evidence (the HMM keeps the last team). Dead-ball part is v1's,
     with the out-of-pitch cue taken from clean observed samples only (a glitch cannot stop play).
  A player-swarm prior (mean shift over nearby players) was tried for long gaps and removed: it
  lowered position error but made the nearest-team answer worse (see the brief).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mtah.decode import PARAMS as V1

NAME = "ball_reconstruction"
PARAMS = {}  # v1 switch / dead costs: this candidate changes the evidence, not the decoder

BALL_VMAX_MPS = 35.0     # hardest struck balls in elite football are ~35 m/s; faster is not a ball
GLITCH_HALF_S = 0.4      # neighbourhood half-width: >= a few frames of glitch stay a minority in it
SIGMA_OBS_M = 0.5        # position noise of a tracked ball (detector + homography), metres
BALL_VRMS_MPS = 7.0      # rms open-play ball speed: carries at 2-5 m/s mixed with 12-20 m/s passes
BALL_TAU_S = 1.0         # direction memory of the ball: a touch or pass roughly every 1-2 s
OUT_MARGIN_M = 0.5       # same line margin as v1: ball this far beyond a line is out of play
EPS = 1e-6               # keeps log(p_dead) finite


def _clean_ball(bx, by, fps):
    """Ball with glitches set to NaN: drop samples further from their local median than
    BALL_VMAX_MPS * GLITCH_HALF_S (a real ball cannot get there and back)."""
    h = int(round(GLITCH_HALF_S * fps))
    pad = np.full(h, np.nan)
    med = []
    for a in (bx, by):
        win = np.sort(sliding_window_view(np.concatenate([pad, a, pad]), 2 * h + 1), axis=1)
        k = np.isfinite(win).sum(axis=1)  # NaN sorts last, so the k valid values come first
        lo, hi = np.maximum((k - 1) // 2, 0), np.maximum(k // 2, 0)
        rows = np.arange(len(a))
        med.append(0.5 * (win[rows, lo] + win[rows, hi]))
    dev = np.hypot(bx - med[0], by - med[1])
    bad = np.isfinite(bx) & (dev > BALL_VMAX_MPS * GLITCH_HALF_S)
    return np.where(bad, np.nan, bx), np.where(bad, np.nan, by)


def _wander(t):
    """Mean squared displacement (m^2) of a ball with rms speed v whose direction decorrelates
    over tau: 2 v^2 tau (t - tau (1 - exp(-t / tau)))  (~ v^2 t^2 short, 2 v^2 tau t long)."""
    tau = BALL_TAU_S
    return 2.0 * BALL_VRMS_MPS ** 2 * tau * (t - tau * (1.0 - np.exp(-t / tau)))


def _bridge(bx, by, fps):
    """Straight-line bridge between the clean samples around each frame of one block and its
    variance (m^2): SIGMA_OBS^2 + _wander(t0*t1/(t0+t1)), one-sided t at the edges, inf with no ball."""
    m = len(bx)
    i = np.arange(m)
    ok = np.isfinite(bx)
    prev = np.maximum.accumulate(np.where(ok, i, -1))
    nxt = np.minimum.accumulate(np.where(ok, i, m)[::-1])[::-1]
    has_p, has_n = prev >= 0, nxt < m
    pc, nc = np.clip(prev, 0, m - 1), np.clip(nxt, 0, m - 1)
    t0, t1 = (i - prev) / fps, (nxt - i) / fps
    both = has_p & has_n
    w = np.where(both & (nxt > prev), (i - prev) / np.maximum(nxt - prev, 1), 0.0)
    w = np.where(has_p, w, 1.0)  # only a next sample: take it
    mx = (1 - w) * bx[pc] + w * bx[nc]
    my = (1 - w) * by[pc] + w * by[nc]
    span = np.where(both, t0 * t1 / np.maximum(t0 + t1, 1e-9), np.where(has_p, t0, t1))
    var = np.where(has_p | has_n, SIGMA_OBS_M ** 2 + _wander(span), np.inf)
    var = np.where(ok, SIGMA_OBS_M ** 2, var)
    return np.where(has_p | has_n, mx, np.nan), np.where(has_p | has_n, my, np.nan), var


def reconstruct(trk):
    """(mx, my, var, clean_x, clean_y): bridged ball estimate and its variance (m^2) per frame;
    outside the blocks only clean observed samples are kept (var inf elsewhere)."""
    fps = float(trk["fps"])
    cx, cy = _clean_ball(trk["bx"].astype(np.float64), trk["by"].astype(np.float64), fps)
    mx, my = cx.copy(), cy.copy()
    var = np.where(np.isfinite(cx), SIGMA_OBS_M ** 2, np.inf)
    for _, lo, hi in trk["blocks"]:
        s = slice(lo, hi + 1)
        mx[s], my[s], var[s] = _bridge(cx[s], cy[s], fps)
    return mx, my, var, cx, cy


def _nearest(trk, side, mx, my):
    """Distance (m) from the ball estimate to the nearest player of `side`; inf where unknown."""
    d = np.hypot(trk["px"] - mx[:, None], trk["py"] - my[:, None])
    d = np.where((trk["team"][None, :] == side) & np.isfinite(d), d, np.inf)
    return d.min(axis=1)


def emissions(trk, p=V1):
    """(n, 3) float64 log-likelihoods for H, A, D."""
    mx, my, var, cx, cy = reconstruct(trk)
    known = np.isfinite(mx) & np.isfinite(var)
    blur = np.where(known, var, 0.0)
    d_h = np.sqrt(_nearest(trk, 0, mx, my) ** 2 + blur)
    d_a = np.sqrt(_nearest(trk, 1, mx, my) ** 2 + blur)
    d_h, d_a = (np.where(known & np.isfinite(d), d, 99.0) for d in (d_h, d_a))

    z_h, z_a = -d_h / p["team_scale_m"], -d_a / p["team_scale_m"]
    m = np.maximum(z_h, z_a)
    log_norm = m + np.log(np.exp(z_h - m) + np.exp(z_a - m))
    conf = np.where(known, np.exp(-np.minimum(d_h, d_a) / p["control_radius_m"]), 0.0)
    lh = conf * (z_h - log_norm) + (1 - conf) * np.log(0.5)
    la = conf * (z_a - log_norm) + (1 - conf) * np.log(0.5)

    present = ~np.isnan(trk["ps"])
    mean_speed = np.where(present.any(axis=1),
                          np.where(present, trk["ps"], 0.0).sum(axis=1) / np.maximum(present.sum(axis=1), 1),
                          p["slow_speed"])
    slow = 1.0 / (1.0 + np.exp(-(p["slow_speed"] - mean_speed) / p["slow_width"]))
    length, width = trk["pitch"]
    with np.errstate(invalid="ignore"):
        out = np.isfinite(cx) & ((cx < -OUT_MARGIN_M) | (cx > length + OUT_MARGIN_M)
                                 | (cy < -OUT_MARGIN_M) | (cy > width + OUT_MARGIN_M))
    p_dead = 1.0 / (1.0 + np.exp(-(p["dead_bias"] + p["dead_out_bonus"] * out
                                   + p["dead_slow_bonus"] * slow)))
    p_dead = np.clip(p_dead, EPS, 1 - EPS)

    e = np.empty((len(mx), 3), dtype=np.float64)
    e[:, 0] = lh + np.log(1 - p_dead)
    e[:, 1] = la + np.log(1 - p_dead)
    e[:, 2] = np.log(p_dead)
    return e
