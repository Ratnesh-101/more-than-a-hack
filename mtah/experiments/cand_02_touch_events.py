"""Touch events: possession = team of the last credible touch, not of the nearest player.

A touch is a KICK (the ball's velocity vector changes sharply while a player is within reach) or a
CARRY (the ball stays within close-control distance of one team's player, and clearly nearer that
team, for a full second). A kick is credited to every team with a player in reach (a duel credits
both). The per-frame H-vs-A log-odds is the clipped difference "time since the last A touch" minus
"time since the last H touch": constant through ball-missing gaps and loose-ball flights (the last
toucher keeps the ball), near zero in scrambles where both teams touch in turn (flat evidence, the
decoder keeps the previous team), and growing once one team keeps touching (control). A ball that
reappears after a gap at the other team's feet is given an unseen hand-over at the gap midpoint.
Dead ball: the incumbent's evidence (ball out, everybody slow) plus two ball-track cues: a ball
seen motionless for a second sits on the spot (dead); a kick in the last two seconds means play.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mtah.experiments.court import incumbent

NAME = "02_touch_events"
# switch_cost 30: after a contested touch the new team needs ~1.5 s of uninterrupted touching
# (lead grows 1 nat/s) before possession flips; a deflection (flip + flip back) costs 60.
# dead_cost 20 > switch_cost / 2 so an H->A hand-over is never routed through a fake stoppage.
PARAMS = {"switch_cost": 30.0, "dead_cost": 20.0}

K = 6            # frames (0.24 s) per side of the velocity line-fit: averages tracker jitter
V_MAX = 40.0     # m/s: faster than the hardest shot (~35 m/s); a step beyond it is a tracking glitch
DV_KICK = 5.0    # m/s (18 km/h) velocity change: a deliberate pass/trap/shot, not jitter or a brush
R_KICK = 2.0     # m: leg reach (~1 m) from a body-centre track plus ~1 m position error
R_CTRL = 1.5     # m: close control while dribbling; the ball is rarely > 1.5 m from the carrier
MARGIN = 0.5     # m: two teams this close to equally near = a contested touch, not control
HOLD = 12        # frames each side (1 s total): "control" = keeping the ball, not passing by it
TAU = 1.0        # s: one second of touch-time lead = 1 nat of evidence for that team
CLIP = 3.0       # nats: cap so one old touch cannot outweigh three seconds of fresh evidence
STILL_R = 1.0    # m: a ball that moves < 1 m in 1 s is resting (on the spot), not in play
STILL_BONUS = 3.0  # log-odds: a resting ball is strong (not certain: a keeper may hold it) dead evidence
KICK_LIVE = 3.0    # log-odds: a fresh kick near a player is strong evidence the ball is in play
LIVE_S = 2.0       # s: a kick keeps play "proven live" this long (one pass flight)
BIG = 10 ** 9    # "never touched" sentinel index


def _team_dist(trk):
    """(d_home, d_away): ball to nearest player of each team (m), inf where unknown."""
    bx, by = trk["bx"], trk["by"]
    out = []
    for side in (0, 1):
        cols = np.flatnonzero(trk["team"] == side)
        d = np.hypot(trk["px"][:, cols] - bx[:, None], trk["py"][:, cols] - by[:, None])
        d = np.where(np.isnan(d), np.inf, d)
        out.append(d.min(axis=1) if len(cols) else np.full(len(bx), np.inf))
    return out


def _slope(z, valid):
    """Least-squares velocity (units/frame) over the K+1 frames ending at t / starting at t."""
    j = np.arange(K + 1, dtype=np.float64)
    w = (j - j.mean()) / ((j - j.mean()) ** 2).sum()
    full = np.convolve(np.where(valid, z, 0.0), w[::-1], mode="valid")  # slope over i..i+K
    back, fwd = np.full(len(z), np.nan), np.full(len(z), np.nan)
    back[K:] = full
    fwd[:len(z) - K] = full
    return back, fwd


def _all_within(ok, k):
    """True at t when every frame in t-k..t+k is ok."""
    pad = np.concatenate([np.zeros(k, bool), ok, np.zeros(k, bool)])
    return sliding_window_view(pad, 2 * k + 1).all(axis=1)


def _spread(z, k):
    """(n, 2k+1) offsets of z over t-k..t+k from z[t] (NaN outside / where missing)."""
    pad = np.concatenate([np.full(k, np.nan), z, np.full(k, np.nan)])
    return sliding_window_view(pad, 2 * k + 1) - z[:, None]


def _backdate(t_h, t_a, vis):
    """A ball that vanishes and reappears at a team's feet changed hands somewhere in the gap; with
    no evidence of when, the gap midpoint minimises the expected boundary error. Credit the first
    toucher after each gap with an unseen touch there (a no-op if it is the same team as before)."""
    t_h, t_a = t_h.copy(), t_a.copy()
    n = len(vis)
    edge = np.diff(np.concatenate([[1], vis.astype(np.int8), [1]]))
    starts, ends = np.flatnonzero(edge == -1), np.flatnonzero(edge == 1)   # gap = [start, end)
    nxt = np.minimum.accumulate(np.where(t_h | t_a, np.arange(n), n)[::-1])[::-1]
    nxt = np.concatenate([nxt, [n]])
    for a, z in zip(starts, ends):
        f = nxt[z]
        if f < n:
            mid = (a + z) // 2
            t_h[mid] |= t_h[f]
            t_a[mid] |= t_a[f]
    return t_h, t_a


def _touches(trk):
    """Frame masks: touch_home, touch_away, kick (any team in reach), ball resting."""
    fps = trk["fps"]
    bx, by = trk["bx"].astype(np.float64), trk["by"].astype(np.float64)
    vis = np.isfinite(bx) & np.isfinite(by)
    with np.errstate(invalid="ignore"):
        jump = np.hypot(np.diff(bx), np.diff(by)) * fps > V_MAX
    good = vis.copy()
    good[1:] &= ~jump
    good[:-1] &= ~jump
    vx_b, vx_f = _slope(bx, good)
    vy_b, vy_f = _slope(by, good)
    dv = np.where(_all_within(good, K), np.hypot(vx_f - vx_b, vy_f - vy_b) * fps, 0.0)
    prev = np.concatenate([[0.0], dv[:-1]])
    nxt = np.concatenate([dv[1:], [0.0]])
    kick = (dv > DV_KICK) & (dv >= prev) & (dv > nxt)   # one event per velocity change

    d_h, d_a = (np.where(vis, d, np.inf) for d in _team_dist(trk))
    near = np.minimum(d_h, d_a)
    reach_h = (d_h < R_KICK) & (d_h <= near + MARGIN)
    reach_a = (d_a < R_KICK) & (d_a <= near + MARGIN)
    held_h = _all_within((d_h < R_CTRL) & (d_h < d_a - MARGIN), HOLD)
    held_a = _all_within((d_a < R_CTRL) & (d_a < d_h - MARGIN), HOLD)
    t_h, t_a = _backdate((kick & reach_h) | held_h, (kick & reach_a) | held_a, vis)

    with np.errstate(invalid="ignore"):
        moved = np.hypot(_spread(bx, HOLD), _spread(by, HOLD)).max(axis=1)  # NaN if any gap
    return t_h, t_a, kick & (reach_h | reach_a), vis & (moved < STILL_R)


def _since(mask, lo, hi):
    """Frames since the last True of mask inside [lo, hi] (huge before the first one)."""
    idx = np.arange(lo, hi + 1)
    return idx - np.maximum.accumulate(np.where(mask[lo:hi + 1], idx, -BIG))


def emissions(trk):
    """(n, 3) float64 log-likelihoods for H, A, D."""
    n, fps = len(trk["bx"]), trk["fps"]
    t_h, t_a, kick, still = _touches(trk)
    r = np.zeros(n)
    recent = np.zeros(n, bool)
    for _, lo, hi in trk["blocks"]:
        lead = (_since(t_a, lo, hi) - _since(t_h, lo, hi)).astype(np.float64)
        r[lo:hi + 1] = np.clip(lead / (TAU * fps), -CLIP, CLIP)
        recent[lo:hi + 1] = _since(kick, lo, hi) < LIVE_S * fps
    log_dead = incumbent.emissions(trk)[:, 2]
    logit = log_dead - np.log1p(-np.exp(log_dead)) + STILL_BONUS * still - KICK_LIVE * recent
    e = np.empty((n, 3), dtype=np.float64)
    e[:, 2] = -np.logaddexp(0.0, -logit)
    log_live = -np.logaddexp(0.0, logit)
    e[:, 0] = -np.logaddexp(0.0, -r) + log_live
    e[:, 1] = -np.logaddexp(0.0, r) + log_live
    return e
