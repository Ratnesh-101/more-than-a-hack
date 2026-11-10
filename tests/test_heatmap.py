"""Tests for analysis.heatmap."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis.heatmap import Palette, draw_pitch, possession_density, make_heatmap


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


def test_possession_density_spot_peak():
    """A team that always holds the ball at one spot produces its peak in that cell."""
    n = 100
    length, width = 105.3, 68.0
    px = np.zeros((n, 2))
    px[:, 0] = 40.0  # home mean x (defends left: mx_h < mx_a)
    px[:, 1] = 65.0  # away mean x
    trk = {
        "pitch": (length, width),
        "bx": np.full(n, 25.0),
        "by": np.full(n, 34.0),
        "px": px,
        "team": np.array([0, 1]),
        "blocks": [(1, 0, n - 1)],
    }
    labels = np.full(n, "H")
    density = possession_density(trk, labels, "H")

    assert density.shape == (21, 14)
    assert np.isclose(density.sum(), 1.0)

    expected_ix = int(25.0 / length * 21)
    expected_iy = int(34.0 / width * 14)
    peak_ix, peak_iy = np.unravel_index(density.argmax(), density.shape)
    assert (peak_ix, peak_iy) == (expected_ix, expected_iy)


def test_possession_density_flipping_halves():
    """Flipping halves flips x as specified when team defends the right goal."""
    n = 100
    length, width = 105.3, 68.0
    px = np.zeros((n, 2))
    # In half 2: home mean x > away mean x => home defends right, attacks left (must flip x)
    px[:, 0] = 65.0
    px[:, 1] = 40.0
    trk = {
        "pitch": (length, width),
        "bx": np.full(n, 25.0),
        "by": np.full(n, 34.0),
        "px": px,
        "team": np.array([0, 1]),
        "blocks": [(2, 0, n - 1)],
    }
    labels = np.full(n, "H")
    density = possession_density(trk, labels, "H")

    assert density.shape == (21, 14)
    assert np.isclose(density.sum(), 1.0)

    # Ball at 25.0 flipped becomes length - 25.0
    expected_flipped_ix = int((length - 25.0) / length * 21)
    expected_iy = int(34.0 / width * 14)
    peak_ix, peak_iy = np.unravel_index(density.argmax(), density.shape)
    assert (peak_ix, peak_iy) == (expected_flipped_ix, expected_iy)


def test_possession_density_away_team_flipping():
    """Away team flips when home defends left and away defends right."""
    n = 100
    length, width = 105.3, 68.0
    px = np.zeros((n, 2))
    px[:, 0] = 40.0  # home defends left
    px[:, 1] = 65.0  # away defends right -> away must flip x
    trk = {
        "pitch": (length, width),
        "bx": np.full(n, 20.0),
        "by": np.full(n, 30.0),
        "px": px,
        "team": np.array([0, 1]),
        "blocks": [(1, 0, n - 1)],
    }
    labels = np.full(n, "A")
    density = possession_density(trk, labels, "A")

    assert density.shape == (21, 14)
    expected_flipped_ix = int((length - 20.0) / length * 21)
    expected_iy = int(30.0 / width * 14)
    peak_ix, peak_iy = np.unravel_index(density.argmax(), density.shape)
    assert (peak_ix, peak_iy) == (expected_flipped_ix, expected_iy)


def test_make_heatmap_synthetic(tmp_path, monkeypatch):
    """End-to-end figure generation with synthetic data."""
    n = 200
    length, width = 105.3, 68.0
    px = np.zeros((n, 2))
    px[:, 0] = 45.0
    px[:, 1] = 60.0
    synthetic_trk = {
        "pitch": (length, width),
        "bx": np.linspace(10, 90, n),
        "by": np.linspace(10, 50, n),
        "px": px,
        "team": np.array([0, 1]),
        "blocks": [(1, 0, n - 1)],
    }
    monkeypatch.setattr("analysis.heatmap.tracking.from_atd", lambda data_dir=None: synthetic_trk)

    csv_path = str(tmp_path / "sub.csv")
    out_png = str(tmp_path / "heatmap.png")
    with open(csv_path, "w") as f:
        f.write("frame,label\n")
        for i in range(n):
            lbl = "H" if i % 2 == 0 else "A"
            f.write(f"{i},{lbl}\n")

    saved = make_heatmap(csv_path, out_png)
    assert os.path.isfile(saved)
    assert os.path.getsize(saved) > 1000
