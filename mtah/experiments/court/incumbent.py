"""Incumbent (v1) as a court candidate: nearest-player softmax + slowness + ball_out.

Same maths as mtah.decode.emissions, rebuilt from the Trk contract (no features.extract), with the
constants imported from mtah.decode.PARAMS. Default decoder, default costs.
"""

import numpy as np

from mtah.decode import PARAMS as V1

NAME = "incumbent"
PARAMS = {}
OUT_MARGIN_M = 0.5
EPS = 1e-6


def _nearest(trk, side, vis):
    """Distance (m) from the ball to the nearest player of `side`; 99 where unknown."""
    d = np.hypot(trk["px"] - trk["bx"][:, None], trk["py"] - trk["by"][:, None])
    d = np.where(trk["team"][None, :] == side, d, np.inf)
    d = np.where(np.isnan(d), np.inf, d).min(axis=1)
    return np.where(vis & np.isfinite(d), d, 99.0)


def emissions(trk, p=V1):
    """(n, 3) float64 log-likelihoods for H, A, D."""
    bx, by = trk["bx"], trk["by"]
    length, width = trk["pitch"]
    vis = ~np.isnan(bx)
    d_h, d_a = _nearest(trk, 0, vis), _nearest(trk, 1, vis)

    z_h, z_a = -d_h / p["team_scale_m"], -d_a / p["team_scale_m"]
    m = np.maximum(z_h, z_a)
    log_norm = m + np.log(np.exp(z_h - m) + np.exp(z_a - m))
    conf = np.where(vis, np.exp(-np.minimum(d_h, d_a) / p["control_radius_m"]), 0.0)
    lh = conf * (z_h - log_norm) + (1 - conf) * np.log(0.5)
    la = conf * (z_a - log_norm) + (1 - conf) * np.log(0.5)

    present = ~np.isnan(trk["ps"])
    mean_speed = np.where(present.any(axis=1),
                          np.where(present, trk["ps"], 0.0).sum(axis=1) / np.maximum(present.sum(axis=1), 1),
                          p["slow_speed"])
    slow = 1.0 / (1.0 + np.exp(-(p["slow_speed"] - mean_speed) / p["slow_width"]))
    out = vis & ((bx < -OUT_MARGIN_M) | (bx > length + OUT_MARGIN_M)
                 | (by < -OUT_MARGIN_M) | (by > width + OUT_MARGIN_M))
    p_dead = 1.0 / (1.0 + np.exp(-(p["dead_bias"] + p["dead_out_bonus"] * out
                                   + p["dead_slow_bonus"] * slow)))
    p_dead = np.clip(p_dead, EPS, 1 - EPS)

    e = np.empty((len(bx), 3), dtype=np.float64)
    e[:, 0] = lh + np.log(1 - p_dead)
    e[:, 1] = la + np.log(1 - p_dead)
    e[:, 2] = np.log(p_dead)
    return e
