"""Territory and attack-direction priors, self-calibrated without labels.

Incumbent nearest-player evidence and dead-ball model, plus a weak H/A prior from two territory
cues that exist even when the ball is lost:
  advance  the whole field's x-velocity, signed by the direction home attacks this half
           (play moves toward the goal of the team that does NOT have the ball);
  zone     how deep the ball sits in home's attacking half (ball if seen, else player centroid).
Attack direction per half comes from the teams' median x separation (each team's block sits in
its own half). The cue weights are NOT set by hand: a logistic fit per half of the cues against
the match's own confident nearest-player frames (pseudo-labels, no human decides any of them).
If a cue carries no information in a match, its fitted weight is ~0 and the prior vanishes.
"""

import numpy as np

from mtah.decode import PARAMS as V1
from mtah.experiments.court import incumbent

NAME = "09_territory"
PARAMS = {}

HALF_WIN_S = 0.5      # +/-0.5 s centred displacement: one stride cycle, longer than tracking jitter
MAX_SPEED_MPS = 10.0  # elite sprint ceiling; faster per-slot motion is a tracker identity jump
ADV_SCALE_MPS = 1.5   # field drift of ~1.5 m/s is a clear attacking push (jog ~2-3 m/s)
SURE_DIST_M = 1.5     # pseudo-label only when a player is within playing distance of the ball
SURE_GAP_M = 3.0      # ... and the other team's nearest is >= 3 m further (no duel)
MIN_SURE = 500        # need >= 20 s of confident frames per half to fit two slopes
RIDGE = 1.0           # unit Gaussian prior on slopes keeps a small fit from exploding
NEWTON_ITERS = 12     # Newton converges in < 10 steps on a 3-parameter logistic
CORR_FRAMES = 25.0    # cues vary on a ~1 s scale: one frame's prior = 1/25 of a second's evidence
CAP_LOGIT = 2.0       # a territory cue alone never claims more than ~88/12 per second


def _attack_dir(px, team):
    """+1 if home attacks toward x = L in this half (home block sits at lower x), else -1."""
    mh = _team_mean_x(px, team == 0)
    ma = _team_mean_x(px, team == 1)
    sep = np.nanmedian(ma - mh) if np.isfinite(ma - mh).any() else 0.0
    return 1.0 if sep >= 0 else -1.0


def _team_mean_x(px, mask):
    sub = px[:, mask]
    cnt = np.isfinite(sub).sum(axis=1)
    tot = np.where(np.isfinite(sub), sub, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / cnt, np.nan)


def _advance(px, team, k, fps):
    """Mean per-slot x-velocity (m/s) over +/-k frames; per-slot so tracklet churn cannot jump it,
    each slot clipped to sprint speed so an identity switch inside a slot cannot either."""
    n = len(px)
    v = np.zeros(n)
    if n <= 2 * k:
        return v
    vel = (px[2 * k:, team >= 0] - px[:-2 * k, team >= 0]) * (fps / (2.0 * k))
    ok = np.isfinite(vel)
    cnt = ok.sum(axis=1)
    tot = np.clip(np.where(ok, vel, 0.0), -MAX_SPEED_MPS, MAX_SPEED_MPS).sum(axis=1)
    v[k:n - k] = tot / np.maximum(cnt, 1)
    return v


def _fit(X, y):
    """Ridge logistic regression (intercept unpenalised), Newton steps; returns coefficients."""
    w = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], RIDGE)
    pen[0] = 0.0
    for _ in range(NEWTON_ITERS):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (y - p) - pen * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen + 1e-9)
        w = w + np.linalg.solve(hess, g)
    return w


def _cues(trk, lo, hi):
    """(n_block, 2) territory cues in home's frame: advance in [-1, 1], zone in [-1, 1]."""
    px = trk["px"][lo:hi + 1]
    team = trk["team"]
    length = trk["pitch"][0]
    d = _attack_dir(px, team)
    adv = np.tanh(d * _advance(px, team, int(HALF_WIN_S * trk["fps"]), trk["fps"]) / ADV_SCALE_MPS)
    bx = trk["bx"][lo:hi + 1].astype(np.float64)
    cx = _team_mean_x(px, team >= 0)
    x = np.where(np.isfinite(bx), bx, cx)
    zone = np.clip(np.where(np.isfinite(x), d * (x - length / 2) / (length / 2), 0.0), -1, 1)
    return np.stack([adv, zone], axis=1)


def emissions(trk):
    """(n, 3) log-likelihoods for H, A, D: incumbent + self-calibrated territory prior."""
    e = incumbent.emissions(trk, V1).copy()
    bx, by = trk["bx"], trk["by"]
    vis = ~np.isnan(bx)
    d_h, d_a = incumbent._nearest(trk, 0, vis), incumbent._nearest(trk, 1, vis)
    for _, lo, hi in trk["blocks"]:
        sl = slice(lo, hi + 1)
        c = _cues(trk, lo, hi)
        dh, da = d_h[sl], d_a[sl]
        sure = (np.minimum(dh, da) < SURE_DIST_M) & (np.abs(dh - da) > SURE_GAP_M)
        if sure.sum() < MIN_SURE:
            continue
        X = np.column_stack([np.ones(int(sure.sum())), c[sure]])
        w = _fit(X, (dh[sure] < da[sure]).astype(np.float64))
        z = np.clip(c @ w[1:], -CAP_LOGIT, CAP_LOGIT) / CORR_FRAMES
        e[sl, 0] += np.log(2.0) - np.logaddexp(0.0, -z)   # log(2 sigmoid(z)): 0 when neutral
        e[sl, 1] += np.log(2.0) - np.logaddexp(0.0, z)
    return e
