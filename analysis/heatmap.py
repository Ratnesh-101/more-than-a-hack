"""Pitch drawing helpers, colour palettes, and possession visualizations.

Caption for publication document:
Possession density and match timeline for FC Barcelona vs Athletic Club, derived from
the submission's labels rather than ground truth. The ball is visible on 59.3% of graded frames
(94,214 of 158,857 frames), so densities describe visible-ball frames only; labels are the
submission's, not ground truth. Attacking direction is normalised left to right for each team
across both halves. Frames with the visible ball outside the pitch lines (490 frames: 288 Barcelona,
202 Atletic Club) are excluded from the density. Each team's spatial density is normalised to its
own total visible possession frames on the pitch so panels share an identical colour scale,
smoothed with a normalised Gaussian kernel.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mtah.v2 import tracking

SAVEFIG_METADATA = {"Software": None, "Creation Time": None, "Date": None}

CAPTION = (
    "Possession density and match timeline for FC Barcelona vs Athletic Club, derived from "
    "the submission's labels rather than ground truth. The ball is visible on 59.3% of graded frames "
    "(94,214 of 158,857 frames), so densities describe visible-ball frames only; labels are the "
    "submission's, not ground truth. Attacking direction is normalised left to right for each team "
    "across both halves. Frames with the visible ball outside the pitch lines (490 frames: 288 Barcelona, "
    "202 Atletic Club) are excluded from the density. Each team's spatial density is normalised to its "
    "own total visible possession frames on the pitch so panels share an identical colour scale, "
    "smoothed with a normalised Gaussian kernel."
)


def get_team_names(trk=None, data_dir=None):
    """Retrieve (home, away) team names from trk, match metadata XML, or fallback."""
    if trk is not None and "team_names" in trk:
        names = trk["team_names"]
        if len(names) >= 2:
            return str(names[0]), str(names[1])
    try:
        import games
        import xml.etree.ElementTree as ET
        paths = games.asset_paths(data_dir=data_dir)
        meta_path = paths.get("atd_meta")
        if meta_path and os.path.isfile(meta_path):
            root = ET.parse(meta_path).getroot()
            names = [team.findtext("Name") for team in root.iter("Team") if team.findtext("Name")]
            if len(names) >= 2:
                return names[0], names[1]
    except Exception:
        pass
    return "Barcelona", "Atletic Club"


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


def print_caption():
    """Print a one-paragraph caption for the document.

    The ball is visible on 59.3% of graded frames, so densities describe visible-ball
    frames only; labels are the submission's, not ground truth. Attacking direction is
    normalised left to right for both teams, and densities reflect each team's own share of
    visible possession frames across a 21x14 grid.
    """
    print(CAPTION)


def parse_size(size_str):
    """Parse figure size string into (width, height) float tuple, e.g. '13x6' or '13,6'."""
    if not size_str:
        return None
    cleaned = size_str.strip().lower()
    if "x" in cleaned:
        parts = cleaned.split("x")
    elif "," in cleaned:
        parts = cleaned.split(",")
    else:
        parts = cleaned.split()
    return float(parts[0]), float(parts[1])


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
    # Exclude frames where the visible ball lies outside pitch lines [0, length] x [0, width]
    on_pitch = (bx >= 0.0) & (bx <= length) & (by >= 0.0) & (by <= width)
    valid = (labels == team_label) & vis_mask & in_blocks & on_pitch
    if not valid.any():
        return np.zeros(bins, dtype=float)

    x_c = np.clip(x_norm[valid], 0.0, length - 1e-6)
    y_c = np.clip(by[valid], 0.0, width - 1e-6)
    hist, _, _ = np.histogram2d(x_c, y_c, bins=bins, range=[[0.0, length], [0.0, width]])
    density = hist / float(valid.sum())
    return _smooth_2d(density, sigma=sigma)


def timeline_bins(labels, lo, hi, fps=25):
    """Discretise frame labels in range lo..hi into 1-second bins.

    Returns integer codes:
      0: omitted (X or unlabelled)
      1: Barcelona ('H')
      2: Athletic Club ('A')
      3: Dead ball ('D')
    """
    sub_lbls = labels[lo:hi + 1]
    n_frames = len(sub_lbls)
    n_bins = int(np.ceil(n_frames / fps))
    out = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        chunk = sub_lbls[b * fps:(b + 1) * fps]
        if len(chunk) == 0:
            continue
        counts = {"H": 0, "A": 0, "D": 0}
        for c in chunk:
            if c in counts:
                counts[c] += 1
        best = max(counts, key=counts.get)
        if counts[best] > 0:
            out[b] = 1 if best == "H" else (2 if best == "A" else 3)
    return out


def possession_thirds(density):
    """Calculate defensive, middle, and attacking thirds possession shares from a 21x14 density array.

    Bins 0..6: Defensive third (own third)
    Bins 7..13: Middle third (neutral)
    Bins 14..20: Attacking third (opponent third)
    """
    def_share = float(density[0:7, :].sum())
    mid_share = float(density[7:14, :].sum())
    att_share = float(density[14:21, :].sum())
    return def_share, mid_share, att_share


def make_heatmap(csv, out_png, data_dir=None, dpi=200, figsize=(13.0, 5.5)):
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
    home_name, away_name = get_team_names(trk=trk, data_dir=data_dir)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi, facecolor="white")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.15, wspace=0.10)

    draw_pitch(ax1, length=length, width=width)
    draw_pitch(ax2, length=length, width=width)

    ax1.imshow(density_h.T, origin="lower", extent=[0, length, 0, width],
               cmap=Palette.CMAP_HOME, vmin=0.0, vmax=vmax, zorder=1, aspect="equal")
    ax2.imshow(density_a.T, origin="lower", extent=[0, length, 0, width],
               cmap=Palette.CMAP_AWAY, vmin=0.0, vmax=vmax, zorder=1, aspect="equal")

    ax1.set_title(f"{home_name} (Home)\n{h_count:,} possession frames ({h_pct:.1f}% visible ball)",
                  fontsize=10.5, fontweight="bold", pad=8, color=Palette.TEXT_COLOR)
    ax2.set_title(f"{away_name} (Away)\n{a_count:,} possession frames ({a_pct:.1f}% visible ball)",
                  fontsize=10.5, fontweight="bold", pad=8, color=Palette.TEXT_COLOR)

    for ax in (ax1, ax2):
        ax.annotate("", xy=(length * 0.72, -5.5), xytext=(length * 0.28, -5.5),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2), zorder=3)
        ax.text(length * 0.5, -4.5, "Attacking direction",
                ha="center", va="bottom", fontsize=8.0, color="#555555", zorder=3)
        ax.set_ylim(-9.0, width + 3.0)

    # Position two slim side-by-side colourbars dynamically directly under each pitch
    fig.canvas.draw()
    p1 = ax1.get_position()
    p2 = ax2.get_position()

    cbar_h = 0.03
    cbar_w = p1.width * 0.82
    cbar_x1 = p1.x0 + (p1.width - cbar_w) / 2
    cbar_x2 = p2.x0 + (p2.width - cbar_w) / 2
    cbar_y = max(p1.y0 - 0.08, 0.03)

    cax1 = fig.add_axes([cbar_x1, cbar_y, cbar_w, cbar_h])
    cax2 = fig.add_axes([cbar_x2, cbar_y, cbar_w, cbar_h])

    vals = np.linspace(0, vmax, 256).reshape(1, -1)
    cax1.imshow(vals, cmap=Palette.CMAP_HOME, extent=[0, vmax * 100, 0, 1], aspect="auto", vmin=0, vmax=vmax)
    cax2.imshow(vals, cmap=Palette.CMAP_AWAY, extent=[0, vmax * 100, 0, 1], aspect="auto", vmin=0, vmax=vmax)

    for cax, name in [(cax1, home_name), (cax2, away_name)]:
        cax.set_yticks([])
        cax.set_xlabel(f"{name} — share of visible possession (%)", fontsize=8.0, color=Palette.TEXT_COLOR)
        cax.tick_params(labelsize=7.5)
        for sp in cax.spines.values():
            sp.set_color("#cccccc")
            sp.set_linewidth(0.8)

    target_path = out_png
    if os.path.isdir(target_path) or not target_path.lower().endswith(".png"):
        os.makedirs(target_path, exist_ok=True)
        target_path = os.path.join(target_path, "possession_heatmap.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    fig.savefig(target_path, dpi=dpi, metadata=SAVEFIG_METADATA)
    plt.close(fig)
    return target_path


def make_timeline(csv, out_png, trk=None, data_dir=None, dpi=200, figsize=(12.0, 5.8)):
    """Generate possession timeline strip and pitch-thirds bar chart figure."""
    if isinstance(csv, str):
        df = pd.read_csv(csv)
        labels = df["label"].values
    elif isinstance(csv, pd.DataFrame):
        labels = csv["label"].values
    else:
        labels = np.asarray(csv)

    if trk is None:
        trk = tracking.from_atd(data_dir=data_dir)

    home_name, away_name = get_team_names(trk=trk, data_dir=data_dir)

    fps = trk.get("fps", 25)
    blocks = trk.get("blocks", [(1, 0, len(labels) - 1)])

    b1 = blocks[0]
    b2 = blocks[1] if len(blocks) > 1 else blocks[0]

    bins1 = timeline_bins(labels, b1[1], b1[2], fps=fps)
    bins2 = timeline_bins(labels, b2[1], b2[2], fps=fps)

    dur1_min = len(bins1) / 60.0
    dur2_min = len(bins2) / 60.0

    density_h = possession_density(trk, labels, "H")
    density_a = possession_density(trk, labels, "A")

    h_def, h_mid, h_att = possession_thirds(density_h)
    a_def, a_mid, a_att = possession_thirds(density_a)

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.22, 0.78], width_ratios=[1.0, 1.0],
                          wspace=0.15, hspace=0.42, left=0.08, right=0.94, top=0.88, bottom=0.10)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    cmap = mcolors.ListedColormap(["#ffffff", Palette.STRIP_H, Palette.STRIP_A, Palette.STRIP_D])

    ax1.imshow(bins1[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=3,
               extent=[0, dur1_min, 0, 1], interpolation="nearest")
    ax2.imshow(bins2[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=3,
               extent=[45, 45 + dur2_min, 0, 1], interpolation="nearest")

    for ax in (ax1, ax2):
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax1.set_xticks([0, 15, 30, 45])
    ax1.set_xticklabels(["0'", "15'", "30'", "45'"], fontsize=8.5, color=Palette.TEXT_COLOR)
    ax1.set_title("First Half (1 s bins)", fontsize=9.5, fontweight="bold", loc="left", color=Palette.TEXT_COLOR)

    ax2.set_xticks([45, 60, 75, 90])
    ax2.set_xticklabels(["45'", "60'", "75'", "90'"], fontsize=8.5, color=Palette.TEXT_COLOR)
    ax2.set_title("Second Half (1 s bins)", fontsize=9.5, fontweight="bold", loc="left", color=Palette.TEXT_COLOR)

    # Legend at top
    patches = [
        mpatches.Patch(color=Palette.STRIP_H, label=f"{home_name} (Home)"),
        mpatches.Patch(color=Palette.STRIP_A, label=f"{away_name} (Away)"),
        mpatches.Patch(color=Palette.STRIP_D, label="Dead ball (stoppage)"),
    ]
    fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.51, 0.98),
               ncol=3, frameon=False, fontsize=9)

    # Bar chart
    categories = ["Defensive third\n(own third)", "Middle third\n(neutral)", "Attacking third\n(opponent third)"]
    x = np.arange(len(categories))
    bar_width = 0.30

    h_vals = [h_def * 100.0, h_mid * 100.0, h_att * 100.0]
    a_vals = [a_def * 100.0, a_mid * 100.0, a_att * 100.0]

    rects1 = ax_bar.bar(x - bar_width / 2, h_vals, bar_width, label=home_name, color=Palette.STRIP_H)
    rects2 = ax_bar.bar(x + bar_width / 2, a_vals, bar_width, label=away_name, color=Palette.STRIP_A)

    for rect in rects1:
        h = rect.get_height()
        ax_bar.annotate("{:.1f}%".format(h), xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=Palette.TEXT_COLOR)

    for rect in rects2:
        h = rect.get_height()
        ax_bar.annotate("{:.1f}%".format(h), xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color=Palette.TEXT_COLOR)

    ax_bar.set_ylabel("Share of visible-ball possession (%)", fontsize=9.5, color=Palette.TEXT_COLOR)
    ax_bar.set_title("Possession share by pitch third (normalised attacking direction)",
                     fontsize=10.5, fontweight="bold", pad=12, color=Palette.TEXT_COLOR)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories, fontsize=9.5, color=Palette.TEXT_COLOR)
    ax_bar.set_ylim(0, max(max(h_vals), max(a_vals)) * 1.18)

    # Styling: no chart borders, no gridlines
    for sp in ("top", "right"):
        ax_bar.spines[sp].set_visible(False)
    ax_bar.spines["left"].set_color("#cccccc")
    ax_bar.spines["bottom"].set_color("#cccccc")
    ax_bar.grid(False)
    ax_bar.legend(frameon=False, loc="upper right", fontsize=9)

    target_path = out_png
    if os.path.isdir(target_path) or not target_path.lower().endswith(".png"):
        os.makedirs(target_path, exist_ok=True)
        target_path = os.path.join(target_path, "possession_timeline.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    fig.savefig(target_path, dpi=dpi, metadata=SAVEFIG_METADATA)
    plt.close(fig)
    return target_path


def main():
    parser = argparse.ArgumentParser(description="Generate possession heatmaps and timelines.")
    parser.add_argument("--csv", default="submissions/nirakshak.csv", help="Submission CSV path")
    parser.add_argument("--out", default="docs/out/figures", help="Output directory or file path")
    parser.add_argument("--data-dir", default=None, help="Gated match data directory")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI (default: 200)")
    parser.add_argument("--size", default=None, help="Figure size as WxH or W,H in inches (e.g. 13x6)")
    parser.add_argument("--caption", action="store_true", help="Print the document caption paragraph")
    args = parser.parse_args()

    if args.caption:
        print_caption()
        return

    parsed_size = parse_size(args.size)
    hm_size = parsed_size if parsed_size is not None else (13.0, 5.5)
    tl_size = parsed_size if parsed_size is not None else (12.0, 5.8)

    out_heatmap = args.out
    out_timeline = args.out
    if not os.path.isdir(args.out) and args.out.lower().endswith(".png"):
        base, ext = os.path.splitext(args.out)
        out_timeline = "{}_timeline{}".format(base, ext)

    saved_heatmap = make_heatmap(args.csv, out_heatmap, data_dir=args.data_dir, dpi=args.dpi, figsize=hm_size)
    saved_timeline = make_timeline(args.csv, out_timeline, data_dir=args.data_dir, dpi=args.dpi, figsize=tl_size)
    print("Saved heatmap to {}".format(saved_heatmap))
    print("Saved timeline to {}".format(saved_timeline))
    print("\nCaption:\n{}".format(CAPTION))


if __name__ == "__main__":
    main()
