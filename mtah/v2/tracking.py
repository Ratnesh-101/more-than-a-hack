"""Raw ATD -> Trk, the one input the v2 pipeline reads.

A Trk is a plain dict (arrays read-only):
  px, py, ps  float32 (n, S)  player x / y in metres and speed in m/s, NaN where missing (S >= 46)
  team        int8 (S,)       0 home, 1 away, -1 unknown; fixed per slot (slots are tracklets)
  bx, by      float32 (n,)    ball in metres, NaN where the tracker lost it
  pitch       (L, W)          metres, origin at a corner
  fps         25
  blocks      [(half, lo, hi)] inclusive video-frame ranges that are graded
"""

import hashlib
import inspect
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import games  # noqa: E402
from baseline import team_of_each_track  # noqa: E402  (official parser for slot -> team)

N_SLOTS = 46
FPS = 25
_ARRAYS = ("px", "py", "ps", "team", "bx", "by")
CACHE_VERSION = "v2-trk-1"   # bump by hand on any change the code hash below cannot see


def cache_stamp(pitch, n):
    """Version + hash of everything that shapes the cached arrays: slot count, fps, pitch scaling,
    frame count and the source of the extraction code path (_group, _stream). Any edit to it
    changes the stamp, so a stale cache is recomputed rather than served."""
    code = ""
    for fn in (_group, _stream):
        try:
            code += inspect.getsource(fn)
        except (OSError, TypeError):   # no .py next to a .pyc: fall back to the bytecode
            code += repr(fn.__code__.co_code) + repr(fn.__code__.co_consts)
    blob = "|".join([CACHE_VERSION, str(N_SLOTS), str(FPS), repr(tuple(float(p) for p in pitch)),
                     str(int(n)), code])
    return CACHE_VERSION + ":" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build(px, py, ps, team, bx, by, pitch, blocks):
    """Validate shapes and dtypes, freeze the arrays, return the Trk dict."""
    px, py, ps, bx, by = (np.ascontiguousarray(a, dtype=np.float32) for a in (px, py, ps, bx, by))
    team = np.ascontiguousarray(team, dtype=np.int8)
    n = len(bx)
    if not (px.shape == py.shape == ps.shape == (n, len(team)) and by.shape == (n,)):
        raise ValueError("Trk shapes disagree: px{} ps{} team{} bx{}".format(
            px.shape, ps.shape, team.shape, bx.shape))
    trk = {"px": px, "py": py, "ps": ps, "team": team, "bx": bx, "by": by,
           "pitch": (float(pitch[0]), float(pitch[1])), "fps": FPS,
           "blocks": [(int(h), int(lo), int(hi)) for h, lo, hi in blocks]}
    for k in _ARRAYS:
        trk[k].flags.writeable = False
    return trk


SPEED_MIN_SLOTS = 10         # a frame counts for the unit check when this many slots are tracked
SPEED_MEDIAN_MIN = 0.3       # m/s; below it the field looks pitch-normalised
SPEED_P95_RANGE = (2.0, 15.0)   # m/s; above the top it looks like km/h, below the bottom normalised


def check_speed_unit(trk):
    """Raise ValueError unless the player speed field looks like m/s.

    Over frames with at least SPEED_MIN_SLOTS tracked slots, the median and 95th percentile of the
    non-NaN ps must satisfy median >= 0.3 and 2.0 <= p95 <= 15 (m/s expected: median about 0.5-3,
    p95 about 4-9). All-NaN speeds do not raise (the position-derived fallback handles them); a
    one-line notice is printed instead. Never modifies trk.
    """
    tracked = (~np.isnan(trk["px"])).sum(axis=1) >= SPEED_MIN_SLOTS
    v = trk["ps"][tracked]
    v = v[~np.isnan(v)]
    if not len(v):
        print("speed check: no speed values (all NaN); skipping unit check, "
              "position-derived fallback applies")
        return
    med, p95 = float(np.median(v)), float(np.percentile(v, 95))
    lo, hi = SPEED_P95_RANGE
    if med < SPEED_MEDIAN_MIN or p95 > hi or p95 < lo:
        raise ValueError(
            "player speed field failed the m/s sanity check: over {:,} values on frames with >= {} "
            "tracked slots the median is {:.3g} and the 95th percentile {:.3g}; expected m/s with "
            "median >= {} and p95 between {} and {} (typically median 0.5-3, p95 4-9). The speed "
            "field is not m/s; re-derive from positions or fix the unit; the champion's dead-ball "
            "cue depends on it.".format(len(v), SPEED_MIN_SLOTS, med, p95, SPEED_MEDIAN_MIN, lo, hi))


def _group(text):
    """'x,y[,speed]' -> (x, y, speed); the speed is NaN when absent or 'NaN'."""
    v = text.split(",")
    speed = float(v[2]) if len(v) > 2 and v[2] and v[2][0] != "N" else np.nan
    return float(v[0]), float(v[1]), speed


def _stream(meta_path, raw_path, n, pitch):
    team = team_of_each_track(meta_path)
    slots = max(N_SLOTS, len(team))
    team = np.concatenate([team, np.full(slots - len(team), -1)]).astype(np.int8)
    length, width = pitch
    px, py, ps = (np.full((n, slots), np.nan, dtype=np.float32) for _ in range(3))
    bx, by = (np.full(n, np.nan, dtype=np.float32) for _ in range(2))
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            vf = int(parts[0]) - 1  # ATD frame_id counts from 1
            if not 0 <= vf < n:
                continue
            for slot, g in enumerate(parts[1].split(";")):
                if slot >= slots or not g or g[0] == "N":
                    continue
                x, y, s = _group(g)
                px[vf, slot], py[vf, slot], ps[vf, slot] = x * length, y * width, s
            if parts[2] and parts[2][0] != "N":
                x, y, _ = _group(parts[2])
                bx[vf], by[vf] = x * length, y * width
    return px, py, ps, team, bx, by


def from_atd(data_dir=None, game=games.GAME, refresh=False):
    """Trk streamed from the raw ATD (about 2 min), cached at <data>/cache/.

    data_dir defaults to MTAH_DATA_DIR, else ./data (games.DATA_DIR). Raises FileNotFoundError,
    naming the missing files, when the gated data is absent and there is no cache.
    """
    root = data_dir or games.DATA_DIR
    pitch = games.GAMES[game]["pitch"]
    blocks = games.half_video_blocks(game)
    n = games.source_frames("tactical", game)
    stamp = cache_stamp(pitch, n)
    cache = os.path.join(root, "cache", "{}_v2_trk.npz".format(game))
    if os.path.isfile(cache) and not refresh:
        z = np.load(cache)
        if "stamp" in z.files and str(z["stamp"]) == stamp:
            return build(*(z[k] for k in _ARRAYS), pitch, blocks)
        # missing or different stamp: stale cache, fall through and recompute from the raw ATD
    try:
        meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir)
    except IOError as err:
        raise FileNotFoundError(str(err))
    arrays = _stream(meta_path, raw_path, n, pitch)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, stamp=np.array(stamp), **dict(zip(_ARRAYS, arrays)))
    return build(*arrays, pitch, blocks)
