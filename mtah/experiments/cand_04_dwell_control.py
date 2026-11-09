"""Dwell-time control: a team owns the ball when one of its players STAYS with it.

Per frame and per team, a contact score c in [0, 1] is high only when that team's nearest player
is within a control radius of the ball AND moves with it (small ball-player relative velocity).
A pass flying past a player, a deflection or a tracker glitch gives high relative speed and so
little contact; a carry or a received-and-kept ball gives sustained contact.

Contact is integrated over time with exponential decay, separately forward (evidence that has
accumulated up to now) and backward (evidence that is about to accumulate), and the two are
summed: a symmetric two-sided kernel, so the integrated hand-over sits where the contact changes
hands, with no causal lag. While the ball is missing the integrators are FROZEN (they neither
decay nor gain), so a gap neither erases a dwell nor invents one. Emissions are a log-softmax of
the integrated control difference, faded to 50/50 when neither team has dwelt on the ball (loose
ball -> the decoder keeps the last possessor, as CHALLENGE.md asks).

Dead-ball evidence is the incumbent's (slowness + ball out), unchanged, so this candidate differs
from it only in the team evidence. Default Viterbi decoder, switch cost halved (see PARAMS).
"""

import numpy as np

from mtah.experiments.court import incumbent

NAME = "04_dwell_control"
# The emission below is already a temporal integral (each contact counts on ~2*TAU*fps frames), so
# the decoder needs a smaller anti-flicker penalty than the incumbent's per-frame evidence (30).
# Halved to 15 after the dev-seed run showed under-segmentation at 30 (disclosed in the brief).
PARAMS = {"switch_cost": 15.0}

CONTROL_RADIUS_M = 1.5   # playing distance: a player reaches the ball within ~1-2 m of his centre
REL_SPEED_MPS = 5.0      # a hard dribble touch runs ~3-5 m/s ahead of the carrier; passes are >= 8-10 m/s
VEL_HALF_WIN = 4         # +/-4 frames (0.32 s) central difference: averages ~0.2 m position noise
                         # down to ~1 m/s velocity noise while staying shorter than a touch cycle
TAU_S = 0.5              # decay time of control evidence, each side: the two-sided kernel then spans
                         # ~1-2 s, the "keeps the ball for a second or two" of a controlled reception
CTRL_SCALE_S = 0.25      # log-odds per second of control lead: a full exclusive dwell (~2*TAU = 1 s of
                         # integrated contact) gives per-frame odds ~e^4, on par with the incumbent's
                         # strongest nearest-player evidence
CONF_DWELL_S = 0.5       # integrated dwell (s) at which team evidence is trusted 63%; below it the ball
                         # is treated as loose and evidence fades to 50/50 (last possessor kept)
CHUNK = 256              # integrator chunk (frames): keeps a^-CHUNK finite in float64 (numerical only)


def _velocity(x, k):
    """Central difference over +/-k frames in units per frame; NaN where an end is missing."""
    v = np.full(x.shape, np.nan, dtype=np.float32)
    v[k:-k] = (x[2 * k:] - x[:-2 * k]) / (2.0 * k)
    return v


def _leaky(c, a):
    """y[t] = a * y[t-1] + c[t] (y[-1] = 0), vectorised in chunks (no per-frame python loop)."""
    n = len(c)
    m = -(-n // CHUNK)
    buf = np.zeros(m * CHUNK)
    buf[:n] = c
    buf = buf.reshape(m, CHUNK)
    j = np.arange(CHUNK)
    local = np.cumsum(buf * a ** (-j), axis=1) * a ** j      # within-chunk recursion from zero
    ends = local[:, -1]
    carry = np.zeros(m)                                       # y at the end of the previous chunk
    for i in range(1, m):                                     # n / 256 iterations, not n
        carry[i] = carry[i - 1] * a ** CHUNK + ends[i - 1]
    return (local + carry[:, None] * a ** (j + 1)).reshape(-1)[:n]


def _integrate(c, vis, a):
    """Two-sided decayed sum of contact c over visible frames; frozen (carried) across gaps."""
    out = np.zeros(len(c))
    idx = np.flatnonzero(vis)
    if len(idx) == 0:
        return out
    cv = c[idx]
    two_sided = _leaky(cv, a) + _leaky(cv[::-1], a)[::-1] - cv   # the present frame counted once
    pos = np.clip(np.searchsorted(idx, np.arange(len(c)), side="right") - 1, 0, None)
    return two_sided[pos]                                          # gap frames hold the last value


def contact(trk):
    """(n, 2) contact score of home / away: near the ball AND moving with it."""
    fps, k = trk["fps"], VEL_HALF_WIN
    bx, by, px, py = trk["bx"], trk["by"], trk["px"], trk["py"]
    d = np.hypot(px - bx[:, None], py - by[:, None])
    bvx, bvy = _velocity(bx, k), _velocity(by, k)
    pvx, pvy = _velocity(px, k), _velocity(py, k)
    rel = np.hypot(pvx - bvx[:, None], pvy - bvy[:, None]) * fps          # m/s, NaN if unknown
    rows = np.arange(len(bx))
    out = np.zeros((len(bx), 2))
    for side in (0, 1):
        ds = np.where((trk["team"][None, :] == side) & np.isfinite(d), d, np.inf)
        j = ds.argmin(axis=1)
        dn, rn = ds[rows, j], rel[rows, j]
        near = np.exp(-0.5 * (np.minimum(dn, 1e3) / CONTROL_RADIUS_M) ** 2)
        moving_with = np.where(np.isfinite(rn), np.exp(-0.5 * (rn / REL_SPEED_MPS) ** 2), 1.0)
        out[:, side] = np.where(np.isfinite(dn), near * moving_with, 0.0)
    return out


def emissions(trk):
    """(n, 3) float64 log-likelihoods for H, A, D."""
    n, fps = len(trk["bx"]), trk["fps"]
    vis = ~np.isnan(trk["bx"])
    c = contact(trk)
    a = np.exp(-1.0 / (TAU_S * fps))
    s = np.zeros((n, 2))
    for _, lo, hi in trk["blocks"]:                       # integrators restart at every half
        for side in (0, 1):
            s[lo:hi + 1, side] = _integrate(c[lo:hi + 1, side], vis[lo:hi + 1], a) / fps
    z = (s[:, 0] - s[:, 1]) / CTRL_SCALE_S
    conf = 1.0 - np.exp(-s.max(axis=1) / CONF_DWELL_S)
    lh = conf * -np.logaddexp(0.0, -z) + (1 - conf) * np.log(0.5)
    la = conf * -np.logaddexp(0.0, z) + (1 - conf) * np.log(0.5)

    e_dead = incumbent.emissions(trk)[:, 2]               # log p_dead, incumbent's dead model
    live = np.log1p(-np.exp(e_dead))
    return np.stack([lh + live, la + live, e_dead], axis=1)
