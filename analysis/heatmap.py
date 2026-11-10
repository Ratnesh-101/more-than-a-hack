"""Pitch drawing helpers, colour palettes, and possession visualizations."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mtah.v2 import tracking


class Palette:
    """Design palette for publication-quality figures."""
    PITCH_LINE = "#b0b0b0"
    PITCH_LINE_WIDTH = 0.9
    BG_COLOR = "#ffffff"
    TEXT_COLOR = "#222222"

    # Heatmap color ramps: single-hue sequential ramps with matched lightness
    HOME_RAMP = ["#ffffff", "#e0f3f8", "#7bccc4", "#2b8cbe", "#084081"]
    AWAY_RAMP = ["#ffffff", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"]

    CMAP_HOME = mcolors.LinearSegmentedColormap.from_list("home_teal_blue", HOME_RAMP)
    CMAP_AWAY = mcolors.LinearSegmentedColormap.from_list("away_orange_red", AWAY_RAMP)

    # Timeline strip colors: distinguishable from heatmap ramps; muted neutral for dead ball
    STRIP_H = "#2b8cbe"
    STRIP_A = "#e34a33"
    STRIP_D = "#969696"
    STRIP_COLORS = {"H": STRIP_H, "A": STRIP_A, "D": STRIP_D}


def draw_pitch(ax, length=105.3, width=68.0, color=None, lw=None):
    """Draw a football pitch with standard markings on the given axes.

    Markings:
      - touchlines and goal lines (outer boundary)
      - halfway line and centre circle (radius 9.15 m)
      - penalty areas (16.5 x 40.32 m)
      - goal areas (5.5 x 18.32 m)
      - penalty spots and centre spot
      - penalty arcs (radius 9.15 m)
      - corner arcs (radius 1.0 m)

    Drawn with thin neutral-grey lines, no fill, equal aspect ratio, and axes disabled.
    """
    line_color = color if color is not None else Palette.PITCH_LINE
    line_width = lw if lw is not None else Palette.PITCH_LINE_WIDTH

    # Touchlines & goal lines (outer boundary)
    ax.plot([0, length, length, 0, 0], [0, 0, width, width, 0],
            color=line_color, lw=line_width, zorder=2)

    # Halfway line
    ax.plot([length / 2, length / 2], [0, width],
            color=line_color, lw=line_width, zorder=2)

    # Centre circle (9.15 m radius)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(length / 2 + 9.15 * np.cos(theta), width / 2 + 9.15 * np.sin(theta),
            color=line_color, lw=line_width, zorder=2)

    # Centre spot
    ax.plot([length / 2], [width / 2], "o", color=line_color, markersize=3, zorder=2)

    # Penalty areas (16.5 x 40.32 m)
    pen_y0 = (width - 40.32) / 2
    pen_y1 = (width + 40.32) / 2
    ax.plot([0, 16.5, 16.5, 0], [pen_y0, pen_y0, pen_y1, pen_y1],
            color=line_color, lw=line_width, zorder=2)
    ax.plot([length, length - 16.5, length - 16.5, length], [pen_y0, pen_y0, pen_y1, pen_y1],
            color=line_color, lw=line_width, zorder=2)

    # Goal areas (5.5 x 18.32 m)
    goal_y0 = (width - 18.32) / 2
    goal_y1 = (width + 18.32) / 2
    ax.plot([0, 5.5, 5.5, 0], [goal_y0, goal_y0, goal_y1, goal_y1],
            color=line_color, lw=line_width, zorder=2)
    ax.plot([length, length - 5.5, length - 5.5, length], [goal_y0, goal_y0, goal_y1, goal_y1],
            color=line_color, lw=line_width, zorder=2)

    # Penalty spots (11.0 m from goal line)
    ax.plot([11.0], [width / 2], "o", color=line_color, markersize=3, zorder=2)
    ax.plot([length - 11.0], [width / 2], "o", color=line_color, markersize=3, zorder=2)

    # Penalty arcs (radius 9.15 m from penalty spot, outside penalty area: cos(angle) = 5.5 / 9.15)
    arc_angle = np.arccos(5.5 / 9.15)
    arc_t = np.linspace(-arc_angle, arc_angle, 40)
    ax.plot(11.0 + 9.15 * np.cos(arc_t), width / 2 + 9.15 * np.sin(arc_t),
            color=line_color, lw=line_width, zorder=2)
    arc_t_r = np.linspace(np.pi - arc_angle, np.pi + arc_angle, 40)
    ax.plot(length - 11.0 + 9.15 * np.cos(arc_t_r), width / 2 + 9.15 * np.sin(arc_t_r),
            color=line_color, lw=line_width, zorder=2)

    # Corner arcs (radius 1.0 m)
    c_t = np.linspace(0, np.pi / 2, 20)
    ax.plot(1.0 * np.cos(c_t), 1.0 * np.sin(c_t),
            color=line_color, lw=line_width, zorder=2)
    ax.plot(1.0 * np.cos(c_t), width - 1.0 * np.sin(c_t),
            color=line_color, lw=line_width, zorder=2)
    ax.plot(length - 1.0 * np.cos(c_t), 1.0 * np.sin(c_t),
            color=line_color, lw=line_width, zorder=2)
    ax.plot(length - 1.0 * np.cos(c_t), width - 1.0 * np.sin(c_t),
            color=line_color, lw=line_width, zorder=2)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.0, length + 2.0)
    ax.set_ylim(-2.0, width + 2.0)
    return ax


def _gaussian_kernel_1d(sigma=1.0, radius=2):
    """Normalized 1D Gaussian kernel."""
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    return k / k.sum()


def _smooth_2d(grid, sigma=1.0, radius=2):
    """Separable 2D Gaussian smoothing using numpy only with normalized boundary handling."""
    k = _gaussian_kernel_1d(sigma, radius)
    k2d = np.outer(k, k)
    pad_grid = np.pad(grid, radius, mode="constant", constant_values=0.0)
    pad_ones = np.pad(np.ones_like(grid, dtype=float), radius, mode="constant", constant_values=0.0)

    from numpy.lib.stride_tricks import sliding_window_view
    w_grid = sliding_window_view(pad_grid, (len(k), len(k)))
    w_ones = sliding_window_view(pad_ones, (len(k), len(k)))

    conv_grid = (w_grid * k2d).sum(axis=(-2, -1))
    conv_ones = (w_ones * k2d).sum(axis=(-2, -1))
    res = conv_grid / np.maximum(conv_ones, 1e-12)
    s = res.sum()
    if s > 0:
        res = res / s
    return res


def possession_density(trk, labels, team, bins=(21, 14), sigma=1.0):
    """Compute normalized, attacking-direction normalized 2D possession density for a team.

    Parameters:
      trk: Trk dict containing pitch, bx, by, px, team, blocks.
      labels: sequence of frame labels ('H', 'A', 'D', 'X').
      team: 'H' or 'A' (or 0 or 1).
      bins: tuple of (x_bins, y_bins), default (21, 14).
      sigma: Gaussian smoothing radius in cells, default 1.0.

    Returns:
      2D numpy array of shape bins, summing to 1.0 (share of visible possession frames).
    """
    team_label = "H" if team in (0, "H", "h", "home", "Home") else "A"
    length, width = float(trk["pitch"][0]), float(trk["pitch"][1])
    bx = np.asarray(trk["bx"], dtype=float)
    by = np.asarray(trk["by"], dtype=float)
    px = np.asarray(trk.get("px", []))
    trk_teams = np.asarray(trk.get("team", []))
    labels = np.asarray(labels)

    blocks = trk.get("blocks", [(1, 0, len(bx) - 1)])
    x_norm = np.copy(bx)
    in_blocks = np.zeros(len(bx), dtype=bool)

    for half, lo, hi in blocks:
        sl = slice(lo, hi + 1)
        in_blocks[sl] = True
        h_mask = (trk_teams == 0)
        a_mask = (trk_teams == 1)
        if px.ndim == 2 and h_mask.any() and a_mask.any():
            mx_h = np.nanmean(px[sl, :][:, h_mask])
            mx_a = np.nanmean(px[sl, :][:, a_mask])
            home_defends_left = bool(mx_h < mx_a)
        else:
            home_defends_left = (half == 1)

        defends_left = home_defends_left if team_label == "H" else (not home_defends_left)
        if not defends_left:
            x_norm[sl] = length - bx[sl]

    vis_mask = ~np.isnan(bx) & ~np.isnan(by)
    valid = (labels == team_label) & vis_mask & in_blocks
    if not valid.any():
        return np.zeros(bins, dtype=float)

    x_c = np.clip(x_norm[valid], 0.0, length - 1e-6)
    y_c = np.clip(by[valid], 0.0, width - 1e-6)
    hist, _, _ = np.histogram2d(x_c, y_c, bins=bins, range=[[0.0, length], [0.0, width]])
    density = hist / float(valid.sum())
    return _smooth_2d(density, sigma=sigma)


def make_heatmap(csv, out_png, data_dir=None, dpi=200, figsize=(13.0, 6.2)):
    """Generate two-panel possession heatmap figure and save to out_png."""
    if isinstance(csv, str):
        df = pd.read_csv(csv)
        labels = df["label"].values
    elif isinstance(csv, pd.DataFrame):
        labels = csv["label"].values
    else:
        labels = np.asarray(csv)

    trk = tracking.from_atd(data_dir=data_dir)
    length, width = float(trk["pitch"][0]), float(trk["pitch"][1])

    density_h = possession_density(trk, labels, "H")
    density_a = possession_density(trk, labels, "A")

    blocks = trk.get("blocks", [(1, 0, len(trk["bx"]) - 1)])
    in_blocks = np.zeros(len(labels), dtype=bool)
    for _, lo, hi in blocks:
        in_blocks[lo:hi + 1] = True
    vis = ~np.isnan(trk["bx"]) & ~np.isnan(trk["by"])

    h_graded = (labels == "H") & in_blocks
    h_count = int(h_graded.sum())
    h_vis = int((h_graded & vis).sum())
    h_pct = (h_vis / h_count * 100.0) if h_count > 0 else 0.0

    a_graded = (labels == "A") & in_blocks
    a_count = int(a_graded.sum())
    a_vis = int((a_graded & vis).sum())
    a_pct = (a_vis / a_count * 100.0) if a_count > 0 else 0.0

    vmax = max(float(density_h.max()), float(density_a.max()))

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.12], width_ratios=[1.0, 1.0],
                          left=0.04, right=0.96, top=0.88, bottom=0.08, wspace=0.10, hspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    draw_pitch(ax1, length=length, width=width)
    draw_pitch(ax2, length=length, width=width)

    ax1.imshow(density_h.T, origin="lower", extent=[0, length, 0, width],
               cmap=Palette.CMAP_HOME, vmin=0.0, vmax=vmax, zorder=1, aspect="equal")
    ax2.imshow(density_a.T, origin="lower", extent=[0, length, 0, width],
               cmap=Palette.CMAP_AWAY, vmin=0.0, vmax=vmax, zorder=1, aspect="equal")

    ax1.set_title("Barcelona (Home)\n{:,} possession frames ({:.1f}% visible ball)".format(h_count, h_pct),
                  fontsize=11, fontweight="bold", pad=10, color=Palette.TEXT_COLOR)
    ax2.set_title("Athletic Club (Away)\n{:,} possession frames ({:.1f}% visible ball)".format(a_count, a_pct),
                  fontsize=11, fontweight="bold", pad=10, color=Palette.TEXT_COLOR)

    for ax in (ax1, ax2):
        ax.annotate("", xy=(length * 0.72, -5.5), xytext=(length * 0.28, -5.5),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2), zorder=3)
        ax.text(length * 0.5, -4.5, "Attacking direction",
                ha="center", va="bottom", fontsize=8.5, color="#555555", zorder=3)
        ax.set_ylim(-9.0, width + 3.0)

    # Shared colourbar at center bottom
    cax = fig.add_subplot(gs[1, :])
    vals = np.linspace(0, vmax, 256)
    vals_norm = vals / max(vmax, 1e-12)
    c_home = Palette.CMAP_HOME(vals_norm)
    c_away = Palette.CMAP_AWAY(vals_norm)
    img = np.stack([c_away, c_home], axis=0)

    cax.imshow(img, extent=[0, vmax * 100, 0, 2], aspect="auto")
    cax.set_yticks([0.5, 1.5])
    cax.set_yticklabels(["Athletic Club", "Barcelona"], fontsize=8.5, color=Palette.TEXT_COLOR)
    cax.set_xlabel("share of visible-ball possession frames (%)", fontsize=9, color=Palette.TEXT_COLOR)
    cax.tick_params(left=False, labelsize=8.5)
    for spine in cax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.8)
    cax.set_position([0.30, 0.08, 0.40, 0.05])

    target_path = out_png
    if os.path.isdir(target_path) or not target_path.lower().endswith(".png"):
        os.makedirs(target_path, exist_ok=True)
        target_path = os.path.join(target_path, "possession_heatmap.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    fig.savefig(target_path, dpi=dpi, metadata={"Software": None, "Creation Time": None, "Date": None})
    plt.close(fig)
    return target_path


def main():
    parser = argparse.ArgumentParser(description="Generate possession heatmaps.")
    parser.add_argument("--csv", default="submissions/nirakshak.csv", help="Submission CSV path")
    parser.add_argument("--out", default="docs/out/figures", help="Output directory or file path")
    parser.add_argument("--data-dir", default=None, help="Gated match data directory")
    args = parser.parse_args()

    saved_path = make_heatmap(args.csv, args.out, data_dir=args.data_dir)
    print("Saved heatmap to {}".format(saved_path))


if __name__ == "__main__":
    main()
