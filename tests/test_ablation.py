"""Unit tests for analysis.ablation helper logic on synthetic arrays (no match data)."""

import hashlib
import io
import numpy as np
import pytest

from analysis import ablation


def test_compute_sha256_synthetic():
    labels = np.array(["H", "A", "D"])
    expected_buf = io.StringIO()
    expected_buf.write("frame,label\n0,H\n1,A\n2,D\n")
    expected_sha = hashlib.sha256(expected_buf.getvalue().encode("utf-8")).hexdigest()

    computed_sha = ablation.compute_sha256(labels)
    assert computed_sha == expected_sha


def test_compute_metrics_identical():
    # 20 frames, two halves: [0..9] and [10..19], 10 fps
    labels = np.array(["H"] * 5 + ["D"] * 5 + ["A"] * 5 + ["D"] * 5)
    default_labels = labels.copy()
    blocks = [(1, 0, 9), (2, 10, 19)]
    fps = 10

    m = ablation.compute_metrics(labels, default_labels, blocks, fps)
    assert m["diff_count"] == 0
    assert m["diff_pct"] == 0.0
    assert m["h1_n"] == 1
    assert m["h2_n"] == 1
    assert m["total_n"] == 2
    assert m["d_share"] == 0.5


def test_compute_metrics_with_diff_inside_and_outside_blocks():
    # 30 frames: [0..9] ungraded, [10..19] half 1, [20..29] half 2
    default_labels = np.full(30, "D")
    labels = default_labels.copy()

    # Change 3 frames outside graded blocks (0..2)
    labels[0:3] = "H"
    # Change 2 frames in half 1 (10..11)
    labels[10:12] = "H"
    # Change 1 frame in half 2 (20)
    labels[20] = "A"

    blocks = [(1, 10, 19), (2, 20, 29)]
    fps = 10
    n_graded = 20

    m = ablation.compute_metrics(labels, default_labels, blocks, fps)
    # Only 3 differing frames are within graded blocks (10, 11, 20)
    assert m["diff_count"] == 3
    assert m["diff_pct"] == pytest.approx(100.0 * 3 / n_graded)


def test_format_markdown_table():
    sample_rows = [
        (
            "(0) Shipped default",
            {
                "h1_n": 10,
                "h2_n": 12,
                "total_n": 22,
                "mean_s": 15.0,
                "lt3": 0.05,
                "flicker": 1.20,
                "d_share": 0.300,
                "diff_count": 0,
                "diff_pct": 0.0,
            },
        ),
        (
            "(a) Alternative",
            {
                "h1_n": 11,
                "h2_n": 13,
                "total_n": 24,
                "mean_s": 14.2,
                "lt3": 0.08,
                "flicker": 1.35,
                "d_share": 0.285,
                "diff_count": 150,
                "diff_pct": 1.25,
            },
        ),
    ]

    md = ablation.format_markdown_table(sample_rows)
    lines = md.splitlines()
    assert len(lines) == 4  # Header + separator + 2 data rows
    assert "Component / Variant" in lines[0]
    assert "Poss (H1 / H2 / Total)" in lines[0]
    assert "(0) Shipped default" in lines[2]
    assert "10 / 12 / 22" in lines[2]
    assert "0.00%" in lines[2]
    assert "(a) Alternative" in lines[3]
    assert "1.25%" in lines[3]


def test_format_text_table():
    sample_rows = [
        (
            "Test Variant",
            {
                "h1_n": 5,
                "h2_n": 6,
                "total_n": 11,
                "mean_s": 12.5,
                "lt3": 0.100,
                "flicker": 1.00,
                "d_share": 0.250,
                "diff_count": 50,
                "diff_pct": 0.50,
            },
        )
    ]
    txt = ablation.format_text_table(sample_rows)
    assert "Test Variant" in txt
    assert "5 / 6 / 11" in txt
    assert "0.50% (50)" in txt
