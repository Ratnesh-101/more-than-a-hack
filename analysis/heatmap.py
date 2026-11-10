"""Pitch drawing helpers, colour palettes, and possession visualizations."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
