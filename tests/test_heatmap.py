"""Tests for analysis.heatmap."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from analysis.heatmap import Palette, draw_pitch


def test_draw_pitch_default():
    fig, ax = plt.subplots()
    draw_pitch(ax)
    assert ax.get_aspect() in ("equal", 1.0)
    assert not ax.axison
    # Outer lines, halfway, circle, spots, boxes, arcs
    assert len(ax.lines) == 16
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim[0] < 0 and xlim[1] > 105.3
    assert ylim[0] < 0 and ylim[1] > 68.0
    plt.close(fig)


def test_draw_pitch_custom_dimensions():
    fig, ax = plt.subplots()
    draw_pitch(ax, length=100.0, width=64.0, color="#888888", lw=1.2)
    assert ax.get_aspect() in ("equal", 1.0)
    assert not ax.axison
    assert len(ax.lines) == 16
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim[0] < 0 and xlim[1] > 100.0
    assert ylim[0] < 0 and ylim[1] > 64.0
    plt.close(fig)


def test_palette_definitions():
    assert len(Palette.HOME_RAMP) >= 2
    assert len(Palette.AWAY_RAMP) >= 2
    assert Palette.HOME_RAMP[0] == "#ffffff"
    assert Palette.AWAY_RAMP[0] == "#ffffff"
    assert "H" in Palette.STRIP_COLORS
    assert "A" in Palette.STRIP_COLORS
    assert "D" in Palette.STRIP_COLORS
