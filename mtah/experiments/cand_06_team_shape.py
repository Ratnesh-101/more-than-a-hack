"""Team shape: ball-free possession evidence from how the two teams move and stretch.

Football: the team in possession progresses toward the opponent goal and stretches long; the team
out of possession drops back toward its own goal and compacts. So (a) the joint block of all
outfield players drifts toward the DEFENDING team's goal, and (b) the attacking team is deeper
(larger front-to-back spread) than the defending one. Both are read from players only, over a
centred ~4 s window, and turned into a small per-frame H/A log-odds by a fixed formula (no labels).

The shape term is weighted by (1 - ball confidence), so it speaks mainly while the ball is missing or
loose; where a player clearly has the ball, the incumbent nearest-player evidence rules. Its main
job is not to create possessions but to LOCATE a hand-over that the ball track hides: with flat
evidence in a gap the decoder would switch at the gap's end; a weak, sign-changing shape term
moves the switch to where the block turned round.
"""

import numpy as np

from mtah.decode import PARAMS as V1
from mtah.experiments.court import incumbent

NAME = "team_shape"
PARAMS = {}

WINDOW_S = 4.0       # a team needs ~3-6 s to turn its block round after a turnover (brief's window)
V_MAX = 10.0         # m/s: above elite sprint speed, so a larger per-slot step is an ID switch
V_REF = 1.0          # m/s: a block progressing ~10 m per 10 s possession = clear advance
DEPTH_REF = 3.0      # m: possession vs defending team length gap (~30-35 m vs ~25-30 m stretch)
W_ADV, W_DEPTH = 2.0, 1.0   # advance is the primary shape cue, depth-stretch a secondary one
# Per-window evidence: a confident shape cue is worth ~3 nats (about a 95/5 call) per
# independent window. Spreading it over the window's frames avoids counting correlated frames
# 100 times; that is what keeps it from overruling switch_cost (30) and inventing possessions.
NATS_PER_WINDOW = 3.0


def _box(x, w):
    """Centred moving mean of x over w frames, ignoring NaN; NaN where the window is empty."""
    ok = np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    k = np.concatenate([[0], np.cumsum(ok)])
    i = np.arange(len(x))
    lo, hi = np.clip(i - w // 2, 0, len(x)), np.clip(i + w // 2 + 1, 0, len(x))
    cnt = k[hi] - k[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _team_mean(a, mask):
    """Row mean of a over the slots in mask, ignoring NaN; NaN where no slot is valid."""
    a = np.where(mask[None, :], a, np.nan)
    ok = np.isfinite(a)
    cnt = ok.sum(axis=1)
    return np.where(cnt > 0, np.where(ok, a, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)


def _depth(px, mask):
    """Front-to-back standard deviation (m) of one team's tracked players per frame."""
    m = _team_mean(px, mask)
    sq = _team_mean((px - m[:, None]) ** 2, mask)
    return np.sqrt(sq)


def shape_logodds(trk, lo, hi):
    """Per-frame log-odds H vs A from team shape on one block [lo, hi] (0 where unknown)."""
    fps = trk["fps"]
    px = trk["px"][lo:hi + 1].astype(np.float64)
    team = trk["team"]
    home, away = team == 0, team == 1
    w = int(WINDOW_S * fps)

    # attack direction of home in this half: home's block sits on the goal it defends
    gap = np.nanmedian(_team_mean(px, away) - _team_mean(px, home))
    d_home = 1.0 if not np.isfinite(gap) or gap >= 0 else -1.0

    # (a) block advance: mean per-slot x velocity of all team-known players (churn-proof: a slot
    # only contributes where it is tracked on both frames), positive = toward away's goal
    vx = np.diff(px, axis=0, prepend=np.nan) * fps
    vx[np.abs(vx) > V_MAX] = np.nan
    adv = _box(_team_mean(vx, home | away), w) * d_home

    # (b) depth stretch: home deeper than away -> home looks like the team in possession
    stretch = _box(_depth(px, home) - _depth(px, away), w)

    score = (W_ADV * np.tanh(np.nan_to_num(adv) / V_REF)
             + W_DEPTH * np.tanh(np.nan_to_num(stretch) / DEPTH_REF)) / (W_ADV + W_DEPTH)
    return score * NATS_PER_WINDOW / w


def emissions(trk):
    """(n, 3) float64 log-likelihoods for H, A, D: incumbent + ball-gated team-shape term."""
    e = incumbent.emissions(trk).copy()
    bx, by = trk["bx"], trk["by"]
    vis = ~np.isnan(bx)
    d = np.hypot(trk["px"] - bx[:, None], trk["py"] - by[:, None])
    d = np.where(np.isnan(d) | (trk["team"][None, :] < 0), np.inf, d).min(axis=1)
    conf = np.where(vis & np.isfinite(d), np.exp(-d / V1["control_radius_m"]), 0.0)
    for _, lo, hi in trk["blocks"]:
        s = shape_logodds(trk, lo, hi) * (1.0 - conf[lo:hi + 1])
        e[lo:hi + 1, 0] += s / 2
        e[lo:hi + 1, 1] -= s / 2
    return e
