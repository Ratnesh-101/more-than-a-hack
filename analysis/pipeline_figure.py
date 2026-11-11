"""Pipeline diagram for the technical document.

Draws the default (semi_markov) end-to-end possession pipeline as a left-to-right
block diagram with short labels (plain words, no code names inside the boxes).
Deterministic rendering with fixed fonts, Agg backend, and no timestamps in
image metadata.
"""

import argparse
import hashlib
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

import matplotlib.patches as patches
import matplotlib.pyplot as plt

SAVEFIG_METADATA = {"Software": None, "Creation Time": None, "Date": None}

# Style palette: white background, single muted teal for process blocks,
# muted grey for input/output blocks, and thin dark-grey arrows.
COLOR_PROCESS_BG = "#2b757e"
COLOR_PROCESS_BORDER = "#1a4e54"
COLOR_IO_BG = "#647480"
COLOR_IO_BORDER = "#445058"
COLOR_TEXT = "#ffffff"
COLOR_ARROW = "#3d4850"

# Smart Tagging indicator styling
COLOR_SMART_BG = "#d2eff2"
COLOR_SMART_BORDER = "#1f5d66"
COLOR_SMART_TEXT = "#0e434a"
COLOR_SMART_ACCENT_PROCESS = "#38b2ac"
COLOR_SMART_ACCENT_IO = "#5ebec4"

# Default blocks definition: plain words per block.
BLOCK_DEFINITIONS = [
    {
        "num": "(1)",
        "label": "Match tracking file, match metadata, and Smart Tagging",
        "lines": ["Match tracking file,", "match metadata,", "and Smart Tagging"],
        "kind": "input",
        "tag": "Input",
        "smart": True,
        "is_island": False,
    },
    {
        "num": "(2)",
        "label": "Per-frame tracks, pitch size, team of each track",
        "lines": ["Per-frame tracks,", "pitch size, team", "of each track"],
        "kind": "process",
        "tag": "Process",
        "smart": False,
        "is_island": False,
    },
    {
        "num": "(3)",
        "label": "Touch evidence: carries, kicks, and team log-odds",
        "lines": ["Touch evidence:", "carries, kicks, and", "team log-odds"],
        "kind": "process",
        "tag": "Process",
        "smart": False,
        "is_island": False,
    },
    {
        "num": "(4)",
        "label": "Dead-ball evidence: walking players, still ball, restarts",
        "lines": ["Dead-ball evidence:", "walking players,", "still ball, restarts"],
        "kind": "process",
        "tag": "Process",
        "smart": False,
        "is_island": False,
    },
    {
        "num": "(5)",
        "label": "Three-state possession decoding with minimum segment durations",
        "lines": ["Three-state", "possession decoding", "with minimum", "segment durations"],
        "kind": "process",
        "tag": "Process",
        "smart": False,
        "is_island": False,
    },
    {
        "num": "(6)",
        "label": "Rules from Smart Tagging: pre-kickoff frames are dead",
        "lines": ["Rules from Smart", "Tagging: pre-kickoff", "frames are dead"],
        "kind": "process",
        "tag": "Process",
        "smart": True,
        "is_island": False,
    },
    {
        "num": "(7)",
        "label": "Live runs with no ball sighting become dead",
        "lines": ["Live runs with", "no ball sighting", "become dead"],
        "kind": "process",
        "tag": "Process",
        "smart": False,
        "is_island": True,
    },
    {
        "num": "(8)",
        "label": "Submission file: H, A or D inside the graded halves, X outside",
        "lines": ["Submission file:", "H, A or D inside", "the graded halves,", "X outside"],
        "kind": "output",
        "tag": "Output",
        "smart": False,
        "is_island": False,
    },
]


def check_has_island_pass(code_content=None):
    """Check whether mtah/v2/decode.py contains the evidence-free island post-pass."""
    if code_content is not None:
        return "_label_evidence_free_islands_dead" in code_content
    try:
        out = subprocess.check_output(
            ["git", "show", "HEAD:mtah/v2/decode.py"],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        return "_label_evidence_free_islands_dead" in out
    except Exception:
        try:
            from mtah.v2 import decode
            return hasattr(decode, "_label_evidence_free_islands_dead")
        except Exception:
            return False


def get_pipeline_blocks(include_islands=None):
    """Return the ordered list of pipeline block dictionaries."""
    if include_islands is None:
        include_islands = check_has_island_pass()
    return [b for b in BLOCK_DEFINITIONS if include_islands or not b["is_island"]]


def draw_pipeline(ax=None, include_islands=None, figsize=(12.0, 2.6), dpi=200):
    """Draw the left-to-right possession pipeline on the given axes.

    Returns (fig, ax, blocks, arrows).
    """
    if include_islands is None:
        include_islands = check_has_island_pass()

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    fig = ax.get_figure()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    blocks = get_pipeline_blocks(include_islands=include_islands)
    n = len(blocks)

    left_m = 0.35
    right_m = 0.35
    arrow_gap = 0.24
    total_gaps = (n - 1) * arrow_gap
    box_w = (12.0 - left_m - right_m - total_gaps) / n
    box_h = 1.50
    box_y = 0.30

    # Main pipeline title
    ax.text(
        6.0, 2.35,
        "Possession pipeline (semi-Markov decoding)",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="#192832"
    )

    # Inputs on left, output on right markers
    ax.text(
        left_m, 2.05,
        "INPUTS  (tracking, metadata, Smart Tagging)",
        ha="left", va="center", fontsize=7.2, fontweight="bold", color="#52626d"
    )
    ax.text(
        12.0 - right_m, 2.05,
        "OUTPUT  (submission file)",
        ha="right", va="center", fontsize=7.2, fontweight="bold", color="#52626d"
    )

    arrows = []
    for i, b in enumerate(blocks):
        x = left_m + i * (box_w + arrow_gap)
        is_io = b["kind"] in ("input", "output")
        bg_col = COLOR_IO_BG if is_io else COLOR_PROCESS_BG
        border_col = COLOR_IO_BORDER if is_io else COLOR_PROCESS_BORDER
        head_bg = "#4e5b65" if is_io else "#1c555c"

        border_lw = 1.2
        if b["smart"]:
            border_col = COLOR_SMART_ACCENT_IO if is_io else COLOR_SMART_ACCENT_PROCESS
            border_lw = 1.6

        # Main block rectangle
        rect = patches.FancyBboxPatch(
            (x, box_y), box_w, box_h,
            boxstyle="round,pad=0.0,rounding_size=0.07",
            linewidth=border_lw, edgecolor=border_col, facecolor=bg_col, zorder=2
        )
        ax.add_patch(rect)

        # Header bar inside box
        head_h = 0.22
        head_y = box_y + box_h - head_h - 0.07
        head_w = box_w * 0.88
        head_x = x + (box_w - head_w) / 2.0
        head_rect = patches.FancyBboxPatch(
            (head_x, head_y), head_w, head_h,
            boxstyle="round,pad=0.0,rounding_size=0.04",
            linewidth=0.5, edgecolor=border_col, facecolor=head_bg, zorder=3
        )
        ax.add_patch(head_rect)
        head_text = "{}  {}".format(b["num"], b["tag"].upper())
        ax.text(
            head_x + head_w / 2.0, head_y + head_h / 2.0, head_text,
            ha="center", va="center", fontsize=6.5, fontweight="bold", color="#ffffff", zorder=4
        )

        # Attached tag under header chip if applicable
        tag_bottom = head_y
        if b["smart"]:
            tag_h = 0.15
            tag_w = box_w * 0.80
            tag_x = x + (box_w - tag_w) / 2.0
            tag_y = head_y - 0.04 - tag_h
            tag_rect = patches.FancyBboxPatch(
                (tag_x, tag_y), tag_w, tag_h,
                boxstyle="round,pad=0.0,rounding_size=0.03",
                linewidth=0.7, edgecolor=COLOR_SMART_BORDER, facecolor=COLOR_SMART_BG, zorder=3
            )
            ax.add_patch(tag_rect)
            ax.text(
                tag_x + tag_w / 2.0, tag_y + tag_h / 2.0,
                "Smart Tagging",
                ha="center", va="center", fontsize=5.3, fontweight="bold", color=COLOR_SMART_TEXT, zorder=4
            )
            tag_bottom = tag_y

        # Label text vertically centered in the body area
        content_top = tag_bottom if b["smart"] else head_y
        content_bot = box_y + 0.07
        text_y = (content_top + content_bot) / 2.0
        full_text = "\n".join(b["lines"])

        ax.text(
            x + box_w / 2.0, text_y, full_text,
            ha="center", va="center", fontsize=6.4, color=COLOR_TEXT,
            linespacing=1.28, zorder=4
        )

        # Thin dark-grey arrow to next block
        if i < n - 1:
            arrow_y = box_y + box_h / 2.0
            arrow = patches.FancyArrowPatch(
                (x + box_w + 0.035, arrow_y),
                (x + box_w + arrow_gap - 0.035, arrow_y),
                arrowstyle="-|>",
                color=COLOR_ARROW,
                linewidth=1.2,
                mutation_scale=10,
                zorder=5,
            )
            ax.add_patch(arrow)
            arrows.append(arrow)

    return fig, ax, blocks, arrows


def make_pipeline_figure(out_path, include_islands=None, dpi=200, figsize=(12.0, 2.6)):
    """Generate and save the pipeline figure deterministically, returning (path, sha256)."""
    target_path = out_path
    if os.path.isdir(target_path) or not target_path.lower().endswith(".png"):
        os.makedirs(target_path, exist_ok=True)
        target_path = os.path.join(target_path, "pipeline.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    fig, ax, blocks, arrows = draw_pipeline(
        include_islands=include_islands, figsize=figsize, dpi=dpi
    )
    fig.savefig(target_path, dpi=dpi, metadata=SAVEFIG_METADATA)
    plt.close(fig)

    with open(target_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()

    return target_path, digest


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline diagram figure.")
    parser.add_argument("--out", default="docs/out/figures", help="Output directory or PNG file path")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI (default: 200)")
    parser.add_argument("--islands", dest="islands", action="store_true", default=None,
                        help="Force include the live island suppression block")
    parser.add_argument("--no-islands", dest="islands", action="store_false",
                        help="Force exclude the live island suppression block")
    args = parser.parse_args()

    saved_path, digest = make_pipeline_figure(args.out, include_islands=args.islands, dpi=args.dpi)
    print("Saved pipeline figure to {}".format(saved_path))
    print("sha256: {}".format(digest))


if __name__ == "__main__":
    main()
