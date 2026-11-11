"""Label-free ablation report for the possession pipeline.

    python -m analysis.ablation --csv-out-dir <dir>

Loads the real tracking (cached) and computes the labels of:
  (0) Shipped default (semi-Markov duration decoder)
  (a) No Smart Tagging kickoff rule (tracking with kickoffs={})
  (b) No evidence-free live island rule (decode._label_evidence_free_islands_dead as no-op)
  (c) Minimum-stoppage Viterbi alternative (--variant champion)
  (d) Touch-only alternative (--variant touch_only)
  (e) No warm-up-touch reset (emissions without pre-kickoff touch suppression)

Reports label-free structure statistics defined in analysis.real_checks (possessions per
half and total, mean duration, share under 3 s, direct flicker per minute, D share, and
percentage of graded frames differing from default). Reads no ground truth and writes
nothing under submissions/.
"""

import argparse
import hashlib
import io
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.real_checks import shape  # noqa: E402
from mtah.v2 import decode, fallback, tracking  # noqa: E402

SHIPPED_SHA256 = "55fd9ef525637467ee6ef405c96bd9821b41d478afc8a5d488e1831a2f464b5f"
DEFAULT_SCRATCH_DIR = os.path.join(
    tempfile.gettempdir(), "claude", "73eaa64e68baf154", "010cc8d5-afad-4b4f-88e0-1970f3e4c9da", "scratchpad", "agyX"
)


def compute_sha256(labels):
    """Compute SHA-256 of the CSV representation of labels."""
    buf = io.StringIO()
    buf.write("frame,label\n")
    buf.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()


def write_csv(path, labels):
    """Write labels to CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def compute_metrics(labels, default_labels, blocks, fps):
    """Compute label-free structure statistics using analysis.real_checks.shape.

    Returns dict with possessions (h1, h2, total), mean_s, lt3, flicker, d_share,
    diff_count, and diff_pct.
    """
    labels = np.asarray(labels)
    default_labels = np.asarray(default_labels)

    graded_mask = np.zeros(len(labels), dtype=bool)
    for _, lo, hi in blocks:
        graded_mask[lo:hi + 1] = True
    n_graded = int(graded_mask.sum())

    diff_count = int(np.sum((labels != default_labels) & graded_mask))
    diff_pct = (100.0 * diff_count / float(n_graded)) if n_graded > 0 else 0.0

    s_tot = shape(labels, blocks, fps)
    h_shapes = [shape(labels, [(h, lo, hi)], fps) for h, lo, hi in blocks]

    h1_n = h_shapes[0]["n"] if len(h_shapes) > 0 else 0
    h2_n = h_shapes[1]["n"] if len(h_shapes) > 1 else 0

    return {
        "h1_n": h1_n,
        "h2_n": h2_n,
        "total_n": s_tot["n"],
        "mean_s": s_tot["mean_s"],
        "lt3": s_tot["lt3"],
        "flicker": s_tot["flicker"],
        "d_share": s_tot["d_share"],
        "diff_count": diff_count,
        "diff_pct": diff_pct,
    }


def format_markdown_table(rows):
    """Format row metrics into a Markdown table string."""
    headers = [
        "Component / Variant",
        "Poss (H1 / H2 / Total)",
        "Mean (s)",
        "Share < 3 s",
        "Flicker (/min)",
        "D Share",
        "Graded Diff (%)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name, m in rows:
        poss_str = "{:d} / {:d} / {:d}".format(m["h1_n"], m["h2_n"], m["total_n"])
        diff_str = "{:.2f}%".format(m["diff_pct"]) if m["diff_count"] > 0 else "0.00%"
        cols = [
            name,
            poss_str,
            "{:.1f} s".format(m["mean_s"]),
            "{:.3f}".format(m["lt3"]),
            "{:.2f} /min".format(m["flicker"]),
            "{:.3f}".format(m["d_share"]),
            diff_str,
        ]
        lines.append("| " + " | ".join(cols) + " |")
    return "\n".join(lines)


def format_text_table(rows):
    """Format row metrics into a plain-text aligned table string."""
    header = "{:<40} | {:>21} | {:>8} | {:>11} | {:>14} | {:>7} | {:>15}".format(
        "Component / Variant", "Poss (H1 / H2 / Total)", "Mean (s)", "Share < 3 s",
        "Flicker (/min)", "D Share", "Graded Diff"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for name, m in rows:
        poss_str = "{:d} / {:d} / {:d}".format(m["h1_n"], m["h2_n"], m["total_n"])
        diff_str = "{:.2f}% ({:d})".format(m["diff_pct"], m["diff_count"]) if m["diff_count"] > 0 else "0.00% (0)"
        line = "{:<40} | {:>21} | {:>7.1f}s | {:>11.3f} | {:>10.2f}/min | {:>7.3f} | {:>15}".format(
            name, poss_str, m["mean_s"], m["lt3"], m["flicker"], m["d_share"], diff_str
        )
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def run_ablations(trk):
    """Compute labels and metrics for shipped default and all ablation variants."""
    blocks = trk["blocks"]
    fps = trk["fps"]

    # (0) Shipped default
    labels_0 = fallback.decode(trk, "semi_markov")

    # (a) No Smart Tagging kickoff rule: build tracking with kickoffs={}
    trk_a = dict(trk, kickoffs={})
    labels_a = fallback.decode(trk_a, "semi_markov")

    # (b) No evidence-free live island rule: patch to a no-op
    orig_island = decode._label_evidence_free_islands_dead
    decode._label_evidence_free_islands_dead = lambda labels, bx: labels
    try:
        labels_b = fallback.decode(trk, "semi_markov")
    finally:
        decode._label_evidence_free_islands_dead = orig_island

    # (c) Minimum-stoppage Viterbi alternative decoder (--variant champion)
    labels_c = decode.decode(trk, decode.emissions(trk))

    # (d) Touch-only alternative (--variant touch_only)
    labels_d = fallback.decode(trk, "touch_only")

    # (e) No warm-up-touch reset: emissions without pre-kickoff touch suppression
    orig_emissions = fallback.emissions
    fallback.emissions = lambda trk_arg: orig_emissions(dict(trk_arg, kickoffs={}))
    try:
        labels_e = fallback.decode(trk, "semi_markov")
    finally:
        fallback.emissions = orig_emissions

    variants = [
        ("(0) Shipped default (semi-Markov)", labels_0),
        ("(a) No Smart Tagging kickoff rule", labels_a),
        ("(b) No live island rule", labels_b),
        ("(c) Min-stoppage Viterbi (champion)", labels_c),
        ("(d) Touch-only alternative", labels_d),
        ("(e) No warm-up-touch reset", labels_e),
    ]

    metrics_rows = []
    for name, labels in variants:
        m = compute_metrics(labels, labels_0, blocks, fps)
        metrics_rows.append((name, m))

    label_dict = {
        "default": labels_0,
        "no_kickoff": labels_a,
        "no_island": labels_b,
        "champion": labels_c,
        "touch_only": labels_d,
        "no_warmup": labels_e,
    }

    return label_dict, metrics_rows


def main():
    ap = argparse.ArgumentParser(description="Label-free ablation report for possession pipeline.")
    ap.add_argument("--csv-out-dir", default=None,
                    help="Directory to write ablation CSV files (writes nothing under submissions/)")
    ap.add_argument("--scratch-dir", default=DEFAULT_SCRATCH_DIR,
                    help="Directory to write Markdown ablation table")
    ap.add_argument("--data-dir", default=None, help="Tracking data directory")
    args = ap.parse_args()

    if args.csv_out_dir:
        sub_dir = os.path.realpath(os.path.join(ROOT, "submissions"))
        out_real = os.path.realpath(args.csv_out_dir)
        if out_real == sub_dir or out_real.startswith(sub_dir + os.sep):
            ap.error("refusing to write under submissions/")

    trk = tracking.from_atd(data_dir=args.data_dir, refresh=False)
    try:
        tracking.check_speed_unit(trk)
    except ValueError as err:
        sys.exit("error: {}".format(err))

    # Verify row (0) first
    labels_0 = fallback.decode(trk, "semi_markov")
    digest_0 = compute_sha256(labels_0)
    print("Row (0) computed SHA-256:", digest_0)
    if digest_0 != SHIPPED_SHA256:
        sys.exit("ERROR: Row (0) SHA-256 {!r} != shipped SHA-256 {!r}".format(digest_0, SHIPPED_SHA256))
    print("Row (0) reproduces shipped SHA-256 exactly.")

    label_dict, rows = run_ablations(trk)

    txt_table = format_text_table(rows)
    md_table = format_markdown_table(rows)

    print("\n" + txt_table + "\n")

    if args.scratch_dir:
        os.makedirs(args.scratch_dir, exist_ok=True)
        md_path = os.path.join(args.scratch_dir, "ablation_table.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(md_table + "\n")
        print("Wrote Markdown table to", md_path)

    if args.csv_out_dir:
        os.makedirs(args.csv_out_dir, exist_ok=True)
        for key, labels in label_dict.items():
            path = os.path.join(args.csv_out_dir, "{}.csv".format(key))
            write_csv(path, labels)
            print("Wrote ablation CSV to", path)


if __name__ == "__main__":
    main()
