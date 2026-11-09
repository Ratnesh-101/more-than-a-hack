"""Unlabeled statistics of the tracking data, to justify mtah.decode.PARAMS and features.OUT_MARGIN_M.

    python -m analysis.unlabeled_stats [--out analysis/out]

Uses ONLY the cached features (mtah.features.load) and the graded-half frame ranges
(games.half_video_blocks). It never reads evaluation/gt or any ground truth, public or private, so
nothing printed here is tuned on labels. Statistics cover graded frames only.
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from mtah import features  # noqa: E402

PCTS = (1, 5, 25, 50, 75, 95, 99)


def runs(mask):
    """Lengths of every maximal run of True in a boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return edges[1::2] - edges[::2]


def pct(name, x, unit=""):
    x = x[np.isfinite(x)]
    if not len(x):
        print("  {}: no data".format(name))
        return
    print("  {} (n={}): ".format(name, len(x)) + "  ".join(
        "p{}={:.2f}".format(q, v) for q, v in zip(PCTS, np.percentile(x, PCTS))) + unit)


def fracs(x, cuts, op, label):
    x = x[np.isfinite(x)]
    print("  " + "   ".join("{} {:g}: {:.1f}%".format(label, c, 100.0 * np.mean(op(x, c)))
                            for c in cuts) if len(x) else "  no data")


def hist(x, bins, title, xlabel, path, rng=None):
    x = x[np.isfinite(x)]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(x, bins=bins, range=rng, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("  wrote " + path)


def valley(x, bin_w=0.1, smooth=5):
    """Valley between the two largest modes of a moving-average-smoothed histogram, or None.

    Modes are local maxima of the smoothed counts; the valley is the lowest smoothed bin between
    the two tallest modes. It counts only if it is below 80% of the shorter mode.
    """
    x = x[np.isfinite(x)]
    counts, edges = np.histogram(x, bins=np.arange(0, np.percentile(x, 99.5) + bin_w, bin_w))
    s = np.convolve(counts, np.ones(smooth) / smooth, mode="same")
    peaks = [i for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] >= s[i + 1]]
    if len(peaks) < 2:
        return None
    a, b = sorted(sorted(peaks, key=lambda i: -s[i])[:2])
    v = a + int(np.argmin(s[a:b + 1]))
    if s[v] >= 0.8 * min(s[a], s[b]):
        return None
    return edges[v] + bin_w / 2, edges[a] + bin_w / 2, edges[b] + bin_w / 2


def main():
    ap = argparse.ArgumentParser(description="Unlabeled statistics behind the decoder constants.")
    ap.add_argument("--out", default=os.path.join("analysis", "out"),
                    help="folder for PNG histograms, relative to the repo root (default: analysis/out)")
    args = ap.parse_args()
    try:
        f = features.load()
    except (IOError, OSError):
        sys.exit("unlabeled_stats: match data not found. Unpack the bundle into data/ or set "
                 "MTAH_DATA_DIR (see README.md, 'Reproduce').")
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    blocks = half_video_blocks(GAME)
    fps = GAMES[GAME]["fps"]
    length, width = GAMES[GAME]["pitch"]
    graded = np.zeros(len(f["ball_vis"]), dtype=bool)
    for _, lo, hi in blocks:
        graded[lo:hi + 1] = True
    g = {k: f[k][graded].astype(np.float64) for k in features.FIELDS}   # per-frame fields only
    vis = g["ball_vis"] > 0.5
    print("graded frames: {}   (no ground truth is read)".format(int(graded.sum())))

    print("\na) mean_speed (m/s), graded frames")
    sp = g["mean_speed"]
    pct("mean_speed", sp)
    v = valley(sp)
    if v is None:
        print("  bimodality: no clear valley (method: 0.1 m/s bins, 5-bin moving average, valley "
              "must be <80% of the shorter of the two tallest modes)")
    else:
        print("  bimodality: valley at {:.2f} m/s between modes at {:.2f} and {:.2f} m/s "
              "(0.1 m/s bins, 5-bin moving average, valley <80% of shorter mode)".format(v[0], v[1], v[2]))
    fracs(sp, (0.5, 1.0, 1.5, 2.0, 2.5), np.less, "<")
    hist(sp, 100, "mean_speed, graded frames", "m/s", os.path.join(out, "mean_speed.png"))

    print("\nb) nearest player distance min(d_home, d_away), ball visible")
    dh, da = g["d_home"][vis], g["d_away"][vis]
    near = np.fmin(dh, da)
    pct("nearest (m)", near)
    fracs(near, (1, 2, 3, 5), np.less_equal, "<=")
    hist(near, 100, "nearest player to ball", "m", os.path.join(out, "nearest_player.png"), (0, 20))

    print("\nc) margin d_home - d_away, ball visible")
    margin = dh - da
    pct("margin (m)", margin)
    print("  |margin| < 1 m: {:.1f}% of frames with both distances".format(
        100.0 * np.mean(np.abs(margin[np.isfinite(margin)]) < 1)))
    hist(margin, 100, "d_home - d_away", "m (negative = home closer)",
         os.path.join(out, "margin.png"), (-20, 20))

    print("\nd) ball outside the lines (signed distance, positive = outside), ball visible")
    bx, by = g["ball_x"][vis], g["ball_y"][vis]
    signed = np.maximum.reduce([-bx, bx - length, -by, by - width])
    print("  (features.py stores only the {:g} m flag; recomputed here from ball_x/ball_y on a "
          "{:g} x {:g} m pitch)".format(features.OUT_MARGIN_M, length, width))
    fracs(signed, (0, 0.5, 1, 2), np.greater, "out >")
    near_line = signed[np.abs(signed) <= 5]
    print("  frames within 5 m of a line: {:.1f}% of visible-ball frames".format(
        100.0 * len(near_line) / max(len(signed), 1)))
    hist(near_line, 100, "ball signed distance to nearest touch/goal line", "m (positive = outside)",
         os.path.join(out, "ball_line_distance.png"), (-5, 5))

    print("\ne) ball visibility")
    for half, lo, hi in blocks:
        print("  half {}: {:.1f}% visible".format(half, 100.0 * np.mean(f["ball_vis"][lo:hi + 1] > 0.5)))
    gaps = np.concatenate([runs(f["ball_vis"][lo:hi + 1] < 0.5) for _, lo, hi in blocks]) / fps
    pct("missing run length (s)", gaps)
    if len(gaps):
        for t in (5, 20):
            print("  share of missing frames in runs > {} s: {:.1f}%".format(
                t, 100.0 * gaps[gaps > t].sum() / gaps.sum()))
    hist(gaps, 100, "ball-missing run lengths", "s", os.path.join(out, "missing_runs.png"), (0, 30))

    print("\nf) players tracked per frame")
    pct("n_home", g["n_home"])
    pct("n_away", g["n_away"])

    print("\ng) ball-stationary runs (ball speed < 0.5 m/s over 1 s, consecutive visible positions)")
    stat = []
    for _, lo, hi in blocks:
        x, y, ok = (f[k][lo:hi + 1].astype(np.float64) for k in ("ball_x", "ball_y", "ball_vis"))
        both = (ok[fps:] > 0.5) & (ok[:-fps] > 0.5)
        slow = np.zeros(len(x), dtype=bool)
        slow[fps:] = both & (np.hypot(x[fps:] - x[:-fps], y[fps:] - y[:-fps]) < 0.5)
        stat.append(runs(slow))
    stat = np.concatenate(stat) / fps
    pct("stationary run (s)", stat)
    hist(stat, 100, "ball-stationary run lengths", "s", os.path.join(out, "stationary_runs.png"), (0, 60))


if __name__ == "__main__":
    main()
