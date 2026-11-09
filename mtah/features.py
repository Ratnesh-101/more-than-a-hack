"""Per-frame features from the tactical ATD, cached to data/cache/<game>_features.npz.

One streaming pass over the raw tracking, then everything downstream reads the cache.
All arrays are indexed by VIDEO frame (0..199318) -- the submission index.

Features (NaN where unavailable):
  ball_x, ball_y        ball position in metres (pitch origin at a corner)
  ball_vis              1 if the tracking sees the ball
  d_home, d_away        distance (m) from ball to the nearest home / away player
  n_home, n_away        number of tracked players per team
  mean_speed            mean speed (m/s) of all tracked players -- low during stoppages
  ball_out              1 if the ball is outside the pitch lines (+ small margin)
  pos                   (n, slots, 2) float16 player positions in metres, NaN = slot not tracked
  slot_team             (slots,) int8, 0 = home, 1 = away (official parser)

Derived on demand (not cached, depends on decode.PARAMS):
  ball_interp           1 on frames whose ball position was filled by fill_ball_gaps()
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402  (official parser for slot -> team)

FIELDS = ("ball_x", "ball_y", "ball_vis", "d_home", "d_away",
          "n_home", "n_away", "mean_speed", "ball_out")
CACHED = FIELDS + ("pos", "slot_team")
OUT_MARGIN_M = 0.5
MAX_BALL_STEP_M = 1.2   # m per frame = 30 m/s at 25 fps, about the fastest realistic ball


def _xy_speed(group):
    v = group.split(",")
    s = float(v[2]) if len(v) > 2 and v[2] and v[2][0] != "N" else np.nan
    return float(v[0]), float(v[1]), s


def _ball_out(bx, by, length, width):
    return ((bx < -OUT_MARGIN_M) | (bx > length + OUT_MARGIN_M)
            | (by < -OUT_MARGIN_M) | (by > width + OUT_MARGIN_M)).astype(np.float32)


def extract(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    n_slots = len(team)
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]

    f = {k: np.full(n, np.nan, dtype=np.float32) for k in FIELDS}
    f["ball_vis"][:] = 0
    pos = np.full((n, n_slots, 2), np.nan, dtype=np.float16)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            vf = int(parts[0]) - 1  # ATD frame_id counts from 1
            if not 0 <= vf < n:
                continue

            px, py, ps, pt, sl = [], [], [], [], []
            for slot, g in enumerate(parts[1].split(";")):
                if not g or g[0] == "N":
                    continue
                x, y, s = _xy_speed(g)
                px.append(x * length)
                py.append(y * width)
                ps.append(s)
                pt.append(team[slot] if slot < n_slots else -1)
                sl.append(slot)
            px, py, ps, pt, sl = map(np.asarray, (px, py, ps, pt, sl))
            f["n_home"][vf] = np.sum(pt == 0)
            f["n_away"][vf] = np.sum(pt == 1)
            if len(ps):
                f["mean_speed"][vf] = np.nanmean(ps)
                ok = sl < n_slots
                pos[vf, sl[ok], 0] = px[ok]
                pos[vf, sl[ok], 1] = py[ok]

            ball = parts[2]
            if not ball or ball[0] == "N":
                continue
            bx, by, _ = _xy_speed(ball)
            bx, by = bx * length, by * width
            f["ball_x"][vf], f["ball_y"][vf], f["ball_vis"][vf] = bx, by, 1
            f["ball_out"][vf] = float(bx < -OUT_MARGIN_M or bx > length + OUT_MARGIN_M
                                      or by < -OUT_MARGIN_M or by > width + OUT_MARGIN_M)
            if len(px):
                d = np.hypot(px - bx, py - by)
                if np.any(pt == 0):
                    f["d_home"][vf] = d[pt == 0].min()
                if np.any(pt == 1):
                    f["d_away"][vf] = d[pt == 1].min()
    f["pos"] = pos
    f["slot_team"] = team.astype(np.int8)
    return f


def fill_ball_gaps(f, max_gap, game=GAME, max_step_m=MAX_BALL_STEP_M):
    """Linearly interpolate short ball-missing gaps; recompute d_home/d_away/ball_out there.

    A gap is filled only if it is <= max_gap frames, the ball is seen on both sides, and the
    implied speed (endpoint displacement / (gap + 1) frames) is < max_step_m per frame -- a
    physical limit, not a fitted number. Filled frames get ball_interp = 1; ball_vis stays the
    raw tracking flag.

    max_gap = 0 returns f itself, untouched (byte-identical downstream output). Never mutates f.
    """
    if max_gap <= 0:
        return f
    n = len(f["ball_vis"])
    bx, by = f["ball_x"], f["ball_y"]
    seen = (f["ball_vis"] > 0.5) & np.isfinite(bx) & np.isfinite(by)

    # maximal runs of unseen frames: [s, e] inclusive
    edge = np.diff(np.concatenate(([1], seen.astype(np.int8), [1])))
    starts, ends = np.flatnonzero(edge == -1), np.flatnonzero(edge == 1) - 1
    gap = ends - starts + 1
    keep = (starts > 0) & (ends < n - 1) & (gap <= max_gap)
    starts, ends, gap = starts[keep], ends[keep], gap[keep]
    if len(starts):
        a, b = starts - 1, ends + 1
        step = np.hypot(bx[b] - bx[a], by[b] - by[a]) / (gap + 1)
        ok = step < max_step_m
        starts, ends, gap, a, b = starts[ok], ends[ok], gap[ok], a[ok], b[ok]

    g = dict(f)
    g["ball_interp"] = np.zeros(n, dtype=np.float32)
    if not len(starts):
        return g
    if "pos" not in f or "slot_team" not in f:
        raise ValueError("features lack player positions (old cache?) -- rerun with --refresh")

    # every filled frame, with its gap's endpoints
    idx = np.concatenate([np.arange(s, e + 1) for s, e in zip(starts, ends)])
    a_r, b_r, g_r = np.repeat(a, gap), np.repeat(b, gap), np.repeat(gap, gap)
    frac = ((idx - a_r) / (g_r + 1)).astype(np.float64)
    ix = bx[a_r] + frac * (bx[b_r] - bx[a_r])
    iy = by[a_r] + frac * (by[b_r] - by[a_r])

    for k in ("ball_x", "ball_y", "d_home", "d_away", "ball_out"):
        g[k] = f[k].copy()
    g["ball_x"][idx], g["ball_y"][idx] = ix, iy
    length, width = GAMES[game]["pitch"]
    g["ball_out"][idx] = _ball_out(ix, iy, length, width)

    p = f["pos"][idx].astype(np.float64)                    # (m, slots, 2)
    d = np.hypot(p[..., 0] - ix[:, None], p[..., 1] - iy[:, None])
    team = f["slot_team"]
    for code, key in ((0, "d_home"), (1, "d_away")):
        cols = d[:, team == code]
        g[key][idx] = (np.fmin.reduce(cols, axis=1) if cols.shape[1]
                       else np.full(len(idx), np.nan))       # all-NaN row -> NaN, like extract
    g["ball_interp"][idx] = 1
    return g


def _load_cache(cache, n):
    """The cached features, or None if the file is stale, corrupt or not what we wrote.

    allow_pickle=False: an .npz can embed pickled objects, and unpickling runs code. Our cache
    only ever holds plain numeric arrays, so anything else is refused. Every expected key must be
    present with the expected length and a numeric dtype; unknown keys are rejected too.
    """
    try:
        with np.load(cache, allow_pickle=False) as z:
            if set(z.files) != set(CACHED):
                return None
            f = {k: z[k] for k in CACHED}
    except (OSError, ValueError, EOFError):
        return None
    for k, v in f.items():
        if v.dtype.kind not in "fiu":
            return None
        if k == "slot_team":
            if v.ndim != 1:
                return None
        elif len(v) != n:
            return None
    if f["pos"].ndim != 3 or f["pos"].shape[1:] != (len(f["slot_team"]), 2):
        return None
    return f


def load(game=GAME, data_dir=None, refresh=False):
    """Features dict, from cache if present. Cache lives under data/ so it is never committed.
    A cache from before player positions were stored is rebuilt automatically."""
    root = data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
    cache = os.path.join(root, "cache", "{}_features.npz".format(game))
    if os.path.isfile(cache) and not refresh:
        f = _load_cache(cache, source_frames("tactical", game))
        if f is not None:
            return f
    f = extract(game, data_dir)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache[:-4] + ".partial.npz"           # atomic: never leave a half-written cache
    np.savez_compressed(tmp, **f)
    os.replace(tmp, cache)
    return f
