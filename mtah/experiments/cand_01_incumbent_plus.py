"""Incumbent+ : the v1 evidence model (nearest-player softmax + slowness + ball_out, 3-state
Viterbi with the 25-frame dead chain), tightened where v1 mis-reads the tracking.

Fixes, each a mechanism rather than a fitted number:
  1. Ball glitches: a ball sample > GLITCH_M from the running median of its neighbours is a
     detector jump (another object), not the ball. Treated as missing.
  2. Short ball gaps (<= INTERP_GAP frames, ~0.5 s) are linearly bridged when the implied speed is
     physical (< MAX_BALL_STEP_M per frame) -- same rule as mtah.features.fill_ball_gaps.
  3. Unknown speed (no player has a speed) gives NO slowness evidence, instead of v1's 50 % slow.
  4. Out of play: margin widened to cover tracking error; a ball last seen out that then vanishes
     stays out for HOLD_OUT_S (it does not come back into play unseen at once).
  5. Ball in flight (> FLIGHT_MPS, faster than any player runs): the nearest player is not
     controlling it, so team evidence fades to flat instead of crediting whoever it flies past.
  6. Slowness only means a stoppage if the ball is not visibly being played: a ball covering
     > MOVE_MPS over 1 s cancels the slow-players bonus (slow build-up play is still live).
  7. Decoding: X until the first informative frame of a half (the scorer back-fills it), so a
     half never starts in H merely because H is state 0; a hand-over decided at the end of a
     ball-missing gap is moved to the gap's middle.
"""

import numpy as np

from mtah.decode import PARAMS as V1, viterbi

NAME = "incumbent_plus"
PARAMS = {}                 # v1 switch_cost 30 / dead_cost 12 untouched: no decoder cost is picked
                            # from synthetic scores (the 25-frame dead chain already makes a
                            # hand-over via a fake stoppage pay ~25 frames of dead emission)

TEAM_SCALE_M = V1["team_scale_m"]          # 2 m: a stride; the controller is within ~1-2 m
CONTROL_RADIUS_M = V1["control_radius_m"]  # 3 m: beyond this nobody is playing the ball
SLOW_SPEED, SLOW_WIDTH = V1["slow_speed"], V1["slow_width"]   # walking pace = stoppage
DEAD_BIAS, OUT_BONUS, SLOW_BONUS = V1["dead_bias"], V1["dead_out_bonus"], V1["dead_slow_bonus"]
OUT_MARGIN_M = 1.0          # ball radius 0.11 m + ~0.5 m tracking error, x2 for safety: a ball
                            # at 0.5 m past the line in tracking is often still on the line
HOLD_OUT_S = 5.0            # even a quick throw-in takes ~5 s from the ball leaving to the restart;
                            # past that an unseen ball may be back in play, so the hold expires
MAX_BALL_STEP_M = 1.2      # m/frame = 30 m/s at 25 fps, about the fastest realistic shot
INTERP_GAP = 12             # ~0.5 s: shorter than a pass, so a bridge cannot skip a hand-over
GLITCH_WIN = 7              # median over +-7 seen samples (15) > the 2-6 frame jumps of a glitch
GLITCH_M = 8.0              # 15 samples of a 20 m/s pass move ~12 m but the median tracks a
                            # straight flight exactly; 8 m off the median is not ball motion
SPEED_HALF = 2              # +-2 frames: 0.16 s baseline; 0.2 m position noise -> ~2 m/s speed noise
FLIGHT_MPS = 9.0            # top sprint speed ~9-10 m/s: a ball moving faster is not at anyone's feet
FLIGHT_WIDTH = 1.0          # ~ the speed noise of the +-2 frame difference
MOVE_HALF = 12              # +-12 frames ~ 1 s baseline: speed noise ~0.3 m/s
MOVE_MPS = 3.0              # jogging pace: a ball at rest or walked to the spot never covers 3 m/s
MOVE_WIDTH = 0.5            # ~ the walking/jogging overlap, well above the 0.3 m/s noise
EPS = 1e-6


def _gate(v, centre, width):
    """Smooth 0..1 step of v around centre; 0 where v is unknown (no evidence either way)."""
    z = np.clip((v - centre) / width, -50, 50)
    return np.where(np.isfinite(v), 1.0 / (1.0 + np.exp(-z)), 0.0)


def _deglitch(bx, by):
    """Copies of bx, by with glitch samples set to NaN."""
    bx, by = bx.astype(np.float64), by.astype(np.float64)
    idx = np.flatnonzero(np.isfinite(bx) & np.isfinite(by))
    if len(idx) <= 2 * GLITCH_WIN:
        return bx, by
    w = 2 * GLITCH_WIN + 1
    pad = lambda v: np.pad(v, GLITCH_WIN, mode="edge")
    mx = np.median(np.lib.stride_tricks.sliding_window_view(pad(bx[idx]), w), axis=1)
    my = np.median(np.lib.stride_tricks.sliding_window_view(pad(by[idx]), w), axis=1)
    bad = idx[np.hypot(bx[idx] - mx, by[idx] - my) > GLITCH_M]
    bx[bad], by[bad] = np.nan, np.nan
    return bx, by


def _runs(missing):
    """Maximal runs of True in a bool array: (starts, ends) inclusive."""
    edge = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    return np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1


def _bridge(bx, by):
    """Linearly fill short, physically possible ball gaps. Returns new arrays."""
    n = len(bx)
    s, e = _runs(~np.isfinite(bx))
    keep = (s > 0) & (e < n - 1) & (e - s + 1 <= INTERP_GAP)
    s, e = s[keep], e[keep]
    a, b = s - 1, e + 1
    ok = np.hypot(bx[b] - bx[a], by[b] - by[a]) / (b - a) < MAX_BALL_STEP_M
    s, e, a, b = s[ok], e[ok], a[ok], b[ok]
    bx, by = bx.copy(), by.copy()
    if len(s):
        g = e - s + 1
        idx = np.concatenate([np.arange(i, j + 1) for i, j in zip(s, e)])
        a_r, b_r = np.repeat(a, g), np.repeat(b, g)
        frac = (idx - a_r) / (b_r - a_r)
        bx[idx] = bx[a_r] + frac * (bx[b_r] - bx[a_r])
        by[idx] = by[a_r] + frac * (by[b_r] - by[a_r])
    return bx, by


def _ball(trk):
    return _bridge(*_deglitch(trk["bx"], trk["by"]))


def _nearest(trk, bx, by, side):
    """Distance (m) from the ball to the nearest player of `side`; 99 where unknown."""
    d = np.hypot(trk["px"] - bx[:, None], trk["py"] - by[:, None])
    d = np.where((trk["team"][None, :] == side) & np.isfinite(d), d, np.inf).min(axis=1)
    return np.where(np.isfinite(d), d, 99.0)


def _out_held(bx, by, pitch, fps):
    """1 where the ball is seen beyond the lines, or vanished there less than HOLD_OUT_S ago."""
    length, width = pitch
    vis = np.isfinite(bx)
    out = vis & ((bx < -OUT_MARGIN_M) | (bx > length + OUT_MARGIN_M)
                 | (by < -OUT_MARGIN_M) | (by > width + OUT_MARGIN_M))
    t = np.arange(len(bx))
    last = np.maximum.accumulate(np.where(vis, t, -1))
    held = (last >= 0) & out[np.maximum(last, 0)] & (t - last <= HOLD_OUT_S * fps)
    return np.where(vis, out, held)


def _ball_speed(bx, by, fps, k):
    """Ball speed (m/s) from a centred +-k frame difference; NaN where unknown."""
    v = np.full(len(bx), np.nan)
    v[k:-k] = np.hypot(bx[2 * k:] - bx[:-2 * k], by[2 * k:] - by[:-2 * k]) * fps / (2 * k)
    return v


def emissions(trk):
    """(n, 3) float64 log-likelihoods for H, A, D."""
    bx, by = _ball(trk)
    vis = np.isfinite(bx)
    d_h, d_a = _nearest(trk, bx, by, 0), _nearest(trk, bx, by, 1)
    # a ball faster than any player can run is in flight: whoever is nearest is not controlling it
    flight = _gate(_ball_speed(bx, by, trk["fps"], SPEED_HALF), FLIGHT_MPS, FLIGHT_WIDTH)
    # a ball travelling faster than a jog over a whole second is being played, not walked to a spot
    moving = _gate(_ball_speed(bx, by, trk["fps"], MOVE_HALF), MOVE_MPS, MOVE_WIDTH)

    z_h, z_a = -d_h / TEAM_SCALE_M, -d_a / TEAM_SCALE_M
    m = np.maximum(z_h, z_a)
    log_norm = m + np.log(np.exp(z_h - m) + np.exp(z_a - m))
    conf = np.where(vis, np.exp(-np.minimum(d_h, d_a) / CONTROL_RADIUS_M), 0.0) * (1 - flight)
    lh = conf * (z_h - log_norm) + (1 - conf) * np.log(0.5)
    la = conf * (z_a - log_norm) + (1 - conf) * np.log(0.5)

    ps = trk["ps"]
    known = np.isfinite(ps)
    cnt = known.sum(axis=1)
    speed = np.where(known, ps, 0.0).sum(axis=1) / np.maximum(cnt, 1)
    slow = np.where(cnt > 0, 1.0 / (1.0 + np.exp(-(SLOW_SPEED - speed) / SLOW_WIDTH)), 0.0)
    slow = slow * (1 - moving)
    out = _out_held(bx, by, trk["pitch"], trk["fps"])
    p_dead = 1.0 / (1.0 + np.exp(-(DEAD_BIAS + OUT_BONUS * out + SLOW_BONUS * slow)))
    p_dead = np.clip(p_dead, EPS, 1 - EPS)

    e = np.empty((len(bx), 3), dtype=np.float64)
    e[:, 0] = lh + np.log(1 - p_dead)
    e[:, 1] = la + np.log(1 - p_dead)
    e[:, 2] = np.log(p_dead)
    return e


def _centre_switches(path, info):
    """Move each direct H<->A switch that follows a stretch with no ball to that stretch's middle.

    Team evidence is flat while the ball is unseen, so Viterbi is indifferent to where inside the
    gap the hand-over goes, and its stay-preferring tie rule puts it at the very end. The hand-over
    happened somewhere in the gap; its middle minimises the worst-case boundary error.
    """
    path = path.copy()
    t_sw = np.flatnonzero((path[1:] != path[:-1]) & (path[1:] < 2) & (path[:-1] < 2)) + 1
    idx = np.arange(len(path))
    last_info = np.maximum.accumulate(np.where(info, idx, -1))
    run_start = np.maximum.accumulate(np.where(np.r_[True, path[1:] != path[:-1]], idx, 0))
    for t in t_sw:
        # unseen frames immediately before t, never reaching back past the old team's own run
        g = min(t - 1 - last_info[t - 1], t - run_start[t - 1])
        if g > 0:
            path[t - g + g // 2:t] = path[t]
    return path


def decode(trk, e):
    """v1 Viterbi per half, X before the first frame with a ball and a player (no evidence, no
    guess; the scorer back-fills it from the first real label), hand-overs centred in ball gaps."""
    p = dict(V1, **PARAMS)
    bx, _ = _ball(trk)
    info = np.isfinite(bx) & np.isfinite(trk["px"]).any(axis=1)
    labels = np.full(len(bx), "X", dtype="<U1")
    letters = np.array(["H", "A", "D"])
    for _, lo, hi in trk["blocks"]:
        idx = np.flatnonzero(info[lo:hi + 1])
        if len(idx):
            a = lo + idx[0]
            labels[a:hi + 1] = letters[_centre_switches(viterbi(e[a:hi + 1], p), info[a:hi + 1])]
    return labels
