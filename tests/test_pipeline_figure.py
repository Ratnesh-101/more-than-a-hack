"""Tests for analysis.pipeline_figure."""

import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from analysis.pipeline_figure import (
    BLOCK_DEFINITIONS,
    check_has_island_pass,
    draw_pipeline,
    get_pipeline_blocks,
    make_pipeline_figure,
)


def test_draw_pipeline_headless():
    """Verify that drawing runs cleanly on a headless figure and returns the expected structure."""
    fig, ax, blocks, arrows = draw_pipeline()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(blocks) in (7, 8)
    assert len(arrows) == len(blocks) - 1
    plt.close(fig)


def test_expected_block_texts_and_word_counts():
    """Verify that every block label has between 6 and 12 plain words, with no code names."""
    code_names = ["semi_markov", "_viterbi_min_dead", "Trk", "bx", "by", "from_atd",
                  "_segment_dp", "decode_half", "Viterbi", "ps", "px", "py"]

    blocks = get_pipeline_blocks(include_islands=True)
    assert len(blocks) == 8

    expected_fragments = [
        "tracking file",
        "tracks",
        "touch evidence",
        "dead-ball evidence",
        "decoding",
        "Smart Tagging",
        "live runs",
        "submission file",
    ]

    for b, frag in zip(blocks, expected_fragments):
        label = b["label"]
        assert frag.lower() in label.lower(), f"Expected fragment {frag!r} in label {label!r}"

        # Word count check: between 6 and 12 plain words
        words = label.split()
        assert 6 <= len(words) <= 12, f"Label {label!r} has {len(words)} words; expected 6 to 12"

        # Check for plain words: no code names
        for code in code_names:
            assert code not in label, f"Code name {code!r} found in block label {label!r}"


def test_figure_text_content():
    """Verify that the rendered figure contains all block labels, markers, and titles."""
    fig, ax, blocks, arrows = draw_pipeline(include_islands=True)

    rendered_texts = [t.get_text() for t in ax.texts]
    full_text_blob = " ".join(rendered_texts)

    # Check input, output, and title markers
    assert "INPUTS" in full_text_blob
    assert "OUTPUT" in full_text_blob
    assert "Possession pipeline (semi-Markov decoding)" in full_text_blob
    assert "Smart Tagging" in full_text_blob

    # Check that block labels are rendered
    for b in blocks:
        for line in b["lines"]:
            assert line in full_text_blob, f"Expected line {line!r} in rendered figure texts"

    plt.close(fig)


def test_arrow_counts_with_and_without_islands():
    """Verify exactly the expected number of arrows: 7 with islands, 6 without."""
    # With islands: 8 blocks -> exactly 7 arrows
    fig1, ax1, blocks1, arrows1 = draw_pipeline(include_islands=True)
    assert len(blocks1) == 8
    assert len(arrows1) == 7
    plt.close(fig1)

    # Without islands: 7 blocks -> exactly 6 arrows
    fig2, ax2, blocks2, arrows2 = draw_pipeline(include_islands=False)
    assert len(blocks2) == 7
    assert len(arrows2) == 6
    plt.close(fig2)


def test_smart_tagging_marked():
    """Verify that the blocks using Smart Tagging are explicitly marked."""
    blocks = get_pipeline_blocks(include_islands=True)
    smart_blocks = [b for b in blocks if b.get("smart")]
    assert len(smart_blocks) == 2
    assert [b["num"] for b in smart_blocks] == ["(1)", "(6)"]
    for b in smart_blocks:
        assert "Smart Tagging" in b["label"]


def test_pipeline_figure_determinism(tmp_path):
    """Verify that two successive runs produce bit-identical PNG files."""
    out1 = str(tmp_path / "run1.png")
    out2 = str(tmp_path / "run2.png")

    path1, hash1 = make_pipeline_figure(out1, include_islands=True)
    path2, hash2 = make_pipeline_figure(out2, include_islands=True)

    assert path1 == out1
    assert path2 == out2
    assert hash1 == hash2

    with open(out1, "rb") as fh1, open(out2, "rb") as fh2:
        assert fh1.read() == fh2.read()


def test_pipeline_figure_dimensions(tmp_path):
    """Verify output image dimensions: 2400 x 520 px at 200 dpi."""
    from PIL import Image

    out_file = str(tmp_path / "dim_test.png")
    make_pipeline_figure(out_file, include_islands=True, dpi=200, figsize=(12.0, 2.6))

    with Image.open(out_file) as img:
        assert img.size == (2400, 520)


def test_check_has_island_pass():
    """Verify that island pass detection correctly responds to code content."""
    assert check_has_island_pass("def _label_evidence_free_islands_dead(labels, bx): pass") is True
    assert check_has_island_pass("def decode_half(): pass") is False
