"""Diagnostics for the nirakshak possession pipeline: plots and summary statistics.

    python -m analysis.diagnostics                          # first 5 graded minutes of half 1
    python -m analysis.diagnostics --start 20000 --end 25000 --out analysis/out
    python -m analysis.diagnostics --check-alignment        # also compare with the public GT

Features are read only through mtah.features.load() and labels only through mtah.decode.decode(),
so this reports exactly what the submission is built from. Nothing here changes any parameter.
"""

import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from mtah import decode, features  # noqa: E402

GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
CODES = {"H": 0, "A": 1, "D": 2, "X": 3}
COLOURS = ["#1f77b4", "#d55e00", "#7f7f7f", "#e8e8e8"]   # H, A, D, X
WINDOW_S = 300                                          # default plot range: 5 minutes


def runs(mask):
    """(start, length) of every maximal run of True in a boolean array."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2] - edges[::2]))


def strip(ax, labels, start, end, ylabel):
    """Draw a label array as a colour strip over frames start..end (inclusive)."""
    codes = np.array([CODES[c] for c in labels[start:end + 1]])[None, :]
    ax.imshow(codes, aspect="auto", cmap=ListedColormap(COLOURS), vmin=0, vmax=3,
              interpolation="nearest", extent=(start, end + 1, 0, 1))
    ax.set_yticks([])
    ax.set_ylabel(ylabel, rotation=0, ha="right", va="center")


def legend(ax):
    ax.legend(handles=[Patch(color=COLOURS[CODES[k]], label=k) for k in "HADX"],
              ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)


def plot_range(f, labels, start, end, path):
    x = np.arange(start, end + 1)
    sl = slice(start, end + 1)
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [1, 3, 2, 1, 1]})
    axes[0].fill_between(x, f["ball_vis"][sl], step="mid", color="#444444")
    axes[0].set_ylabel("ball_vis")
    axes[1].plot(x, f["d_home"][sl], lw=0.8, color=COLOURS[0], label="d_home")
    axes[1].plot(x, f["d_away"][sl], lw=0.8, color=COLOURS[1], label="d_away")
    axes[1].set_ylabel("nearest player (m)")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    axes[2].plot(x, f["mean_speed"][sl], lw=0.8, color="#444444")
    axes[2].set_ylabel("mean_speed (m/s)")
    axes[3].fill_between(x, np.nan_to_num(f["ball_out"][sl]), step="mid", color="#444444")
    axes[3].set_ylabel("ball_out")
    strip(axes[4], labels, start, end, "predicted")
    legend(axes[4])
    axes[4].set_xlim(start, end + 1)
    axes[4].set_xlabel("video frame (25 fps)")
    fig.suptitle("{}  frames {}-{}".format(GAME, start, end))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def read_gt_letters(n):
    """Public GT letters indexed by video frame (X where the file has no row)."""
    gt = np.full(n, "X", dtype="<U1")
    with open(GT_PUBLIC, newline="") as fh:
        for row in csv.DictReader(fh):
            gt[int(row["frame"])] = row["label"].strip().upper()
    return gt


def check_alignment(labels, out_dir):
    """Compare our labels with the public GT on the first 5 minutes and plot both.

    This exists only to confirm that our frame indexing lines up with the GT (an off-by-one or a
    wrong half offset would show as a systematic shift between the strips). It reports plain
    letter agreement on graded frames, never spellF1 or macroF1, and its result must not be used
    to pick or tune anything (RULES.md rule 2).
    """
    start = half_video_blocks(GAME)[0][1]
    end = start + WINDOW_S * GAMES[GAME]["fps"] - 1
    gt = read_gt_letters(len(labels))
    graded = gt[start:end + 1] != "X"
    agree = labels[start:end + 1][graded] == gt[start:end + 1][graded]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13, 3))
    strip(axes[0], labels, start, end, "predicted")
    strip(axes[1], gt, start, end, "public GT")
    legend(axes[1])
    axes[1].set_xlabel("video frame (25 fps)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alignment_check.png"), dpi=110)
    plt.close(fig)
    print("alignment check: letters agree on {:.1f}% of {} graded frames in {}-{}".format(
        100.0 * agree.mean(), int(graded.sum()), start, end))


def print_stats(f, labels, blocks, fps):
    print("\nball visible per graded half")
    for half, lo, hi in blocks:
        print("  half {}: {:.1f}%".format(half, 100.0 * np.mean(f["ball_vis"][lo:hi + 1] > 0.5)))

    durations, switches = [], {"live<->dead": 0, "H<->A": 0}
    for _, lo, hi in blocks:
        seg = labels[lo:hi + 1]
        for letter in "HA":
            durations += [n / fps for _, n in runs(seg == letter)]
        a, b = seg[:-1], seg[1:]
        changed = a != b
        switches["live<->dead"] += int(np.sum(changed & ((a == "D") | (b == "D"))))
        switches["H<->A"] += int(np.sum(changed & (a != "D") & (b != "D")))
    d = np.array(durations)
    print("\npossessions (maximal H or A runs, per half): {}".format(len(d)))
    if len(d):
        print("  duration s: min {:.1f}  median {:.1f}  mean {:.1f}  max {:.1f}".format(
            d.min(), np.median(d), d.mean(), d.max()))
        print("  shorter than 3 s: {}   shorter than 5 s: {}".format(
            int(np.sum(d < 3)), int(np.sum(d < 5))))

    print("\nlabel counts: {}".format({k: int(np.sum(labels == k)) for k in "HADX"}))
    print("switches between consecutive frames (H->D->A counts as two live<->dead): {}".format(
        switches))

    missing = []
    for _, lo, hi in blocks:
        missing += [(lo + s, n) for s, n in runs(f["ball_vis"][lo:hi + 1] < 0.5)]
    missing.sort(key=lambda r: -r[1])
    print("\n10 longest ball-missing stretches in graded halves")
    for start, n in missing[:10]:
        print("  frame {:>6}  {:6.1f} s".format(start, n / fps))


def main():
    blocks = half_video_blocks(GAME)
    fps = GAMES[GAME]["fps"]
    ap = argparse.ArgumentParser(description="Plots and statistics for the possession pipeline.")
    ap.add_argument("--start", type=int, default=blocks[0][1],
                    help="first video frame to plot (default: start of half 1)")
    ap.add_argument("--end", type=int, default=None,
                    help="last video frame to plot (default: start + 5 minutes)")
    ap.add_argument("--out", default=os.path.join("analysis", "out"),
                    help="output folder, relative to the repo root (default: analysis/out)")
    ap.add_argument("--check-alignment", action="store_true",
                    help="overlay the public GT on the first 5 minutes and print letter agreement")
    args = ap.parse_args()

    try:
        f = features.load()
    except (IOError, OSError):
        sys.exit("diagnostics: match data not found. Unpack the bundle into data/ or set "
                 "MTAH_DATA_DIR (see README.md, 'Reproduce').")
    labels = decode.decode(f)

    end = args.end if args.end is not None else args.start + WINDOW_S * fps - 1
    if not 0 <= args.start < end < len(labels):
        sys.exit("diagnostics: need 0 <= --start < --end < {}".format(len(labels)))
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    png = os.path.join(out_dir, "diagnostics.png")
    plot_range(f, labels, args.start, end, png)
    print("wrote {}".format(png))
    print_stats(f, labels, blocks, fps)
    if args.check_alignment:
        check_alignment(labels, out_dir)


if __name__ == "__main__":
    main()
