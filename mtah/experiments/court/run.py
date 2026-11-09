"""The court: runs every candidate on synthetic matches (scored against known truth) and,
with --real, on the real tracking (UNLABELED diagnostics only; no ground truth is read here).

    python -m mtah.experiments.court.run [--cand slug ...] [--seeds 1,2,3] [--real] [--rules-lint]
"""

import argparse
import glob
import importlib
import json
import os
import re
import sys
import time

import numpy as np

from . import incumbent, synth, trk as trk_mod
from .trk import n_frames
from evaluation import possession_metrics as pm  # noqa: E402  (scoring functions only)
from mtah.decode import PARAMS as V1, viterbi  # noqa: E402

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
OVERRIDABLE = ("switch_cost", "dead_cost")
LETTERS = np.array(["H", "A", "D"])
FPS = trk_mod.FPS

# -- rules lint -------------------------------------------------------------------------------
FORBIDDEN = (
    (r"evaluation[/\\]+gt", "reads the ground-truth folder"),
    (r"gt_public|possession_gt", "names a ground-truth file"),
    (r"possession_metrics|(?:from|import)\s+evaluation", "imports the scorer (labels are not for candidates)"),
    (r"\b[A-Za-z]:\\|\b[A-Za-z]:/(?!/)|/Users/|/home/", "absolute path"),
    (r"smart_tagging", "Smart Tagging needs a 'lint-allow: smart_tagging <why>' note"),
)
SMART_OK = re.compile(r"lint-allow:\s*smart_tagging\s+\S+")


def lint(path):
    """List of 'line N: reason' violations found in a candidate source file."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    allowed = any(SMART_OK.search(line) for line in lines)
    found = []
    for i, line in enumerate(lines, 1):
        for pattern, why in FORBIDDEN:
            if re.search(pattern, line, re.IGNORECASE) and not (allowed and "smart_tagging" in pattern):
                found.append("line {}: {}".format(i, why))
    return found


# -- candidates -------------------------------------------------------------------------------

def discover(slugs):
    """Candidate records (incumbent first) for cand_*.py files matching any slug (all if none)."""
    paths = sorted(glob.glob(os.path.join(EXP_DIR, "cand_*.py")))
    wanted = [(os.path.basename(p)[:-3], p) for p in paths
              if not slugs or any(s in os.path.basename(p) for s in slugs)]
    records = [{"name": incumbent.NAME, "mod": incumbent, "error": None}]
    for name, path in wanted:
        rec = {"name": name, "mod": None, "error": None}
        problems = lint(path)
        if problems:
            rec["error"] = "rules-lint: " + "; ".join(problems[:3])
        else:
            try:
                rec["mod"] = importlib.import_module("mtah.experiments." + name)
                rec["name"] = getattr(rec["mod"], "NAME", name)
            except Exception as err:  # a broken candidate must not stop the court
                rec["error"] = "import: {}: {}".format(type(err).__name__, err)
        records.append(rec)
    return records


def decode_default(trk, e, params):
    p = dict(V1, **{k: v for k, v in params.items() if k in OVERRIDABLE})
    labels = np.full(n_frames(trk), "X", dtype="<U1")
    for _, lo, hi in trk["blocks"]:
        labels[lo:hi + 1] = LETTERS[viterbi(e[lo:hi + 1], p)]
    return labels


def run_candidate(mod, trk):
    """(labels, seconds). Labels are H/A/D inside the blocks and X outside, whatever the candidate did."""
    n = n_frames(trk)
    t0 = time.perf_counter()
    e = np.asarray(mod.emissions(trk), dtype=np.float64)
    if e.shape != (n, 3) or np.isnan(e).any() or (e == np.inf).any():
        raise ValueError("emissions must be finite-or--inf (n,3); got shape {} / NaN or +inf".format(e.shape))
    if hasattr(mod, "decode"):
        labels = np.asarray(mod.decode(trk, e)).astype("<U1")
        if labels.shape != (n,) or not set(np.unique(labels)) <= set("HADX"):
            raise ValueError("decode must return an (n,) array of H/A/D/X")
    else:
        labels = decode_default(trk, e, getattr(mod, "PARAMS", {}))
    inside = np.zeros(n, bool)
    for _, lo, hi in trk["blocks"]:
        inside[lo:hi + 1] = True
    labels = np.where(inside, labels, "X")
    return labels, time.perf_counter() - t0


# -- statistics -------------------------------------------------------------------------------

def shape_stats(labels, blocks):
    """Unlabeled description of a label sequence inside `blocks` (X forward-filled like the scorer)."""
    runs = [(lo, hi + 1) for _, lo, hi in blocks]
    codes = pm.label_codes(labels)
    filled = pm.fill_abstentions(codes, runs)
    spells = pm.possession_spells(filled, runs)
    dur = np.array([b - a for a, b, _ in spells], dtype=float) / FPS
    minutes = sum(b - a for a, b in runs) / FPS / 60.0
    live_switches = sum(1 for (_, b, c), (a2, _, c2) in zip(spells, spells[1:]) if b == a2 and c != c2)
    frames = np.concatenate([codes[a:b] for a, b in runs])
    return {
        "n": len(dur),
        "mean_s": float(dur.mean()) if len(dur) else 0.0,
        "lt3": float((dur < 3).mean()) if len(dur) else 0.0,
        "lt5": float((dur < 5).mean()) if len(dur) else 0.0,
        "longest_s": float(dur.max()) if len(dur) else 0.0,
        "flicker": live_switches / minutes,
        "share": {k: float((frames == i).mean()) for i, k in enumerate("HADX")},
    }


def score_synth(labels, truth_codes, truth_stats, blocks):
    res = pm.evaluate(truth_codes, pm.label_codes(labels))
    st = shape_stats(labels, blocks)
    return {"spell": res["spell_f1_mean"], "macro": res["macro_f1"], "n_pred": st["n"],
            "n_true": res["n_gt_spells"], "mean_s": st["mean_s"], "true_mean_s": truth_stats["mean_s"],
            "lt3": st["lt3"], "flicker": st["flicker"], "true_flicker": truth_stats["flicker"]}


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


# -- sections ---------------------------------------------------------------------------------

def synthetic_section(records, seeds, minutes):
    games = {}
    for diff in ("easy", "realistic"):
        for seed in seeds:
            trk, truth = synth.make(seed, minutes, diff)
            codes = pm.label_codes(truth)
            games[diff, seed] = (trk, codes, shape_stats(truth, trk["blocks"]))
    detail, summary, dq = [], [], []
    for rec in records:
        if rec["error"]:
            dq.append((rec["name"], rec["error"]))
            continue
        rows = {"easy": [], "realistic": []}
        try:
            for (diff, seed), (trk, codes, tstats) in games.items():
                labels, secs = run_candidate(rec["mod"], trk)
                row = score_synth(labels, codes, tstats, trk["blocks"])
                row.update(secs=secs, seed=seed)
                rows[diff].append(row)
                detail.append([rec["name"], diff, seed, "{:.4f}".format(row["spell"]),
                               "{:.4f}".format(row["macro"]), "{}/{}".format(row["n_pred"], row["n_true"]),
                               "{:.1f}/{:.1f}".format(row["mean_s"], row["true_mean_s"]),
                               "{:.2f}".format(row["lt3"]),
                               "{:.2f}/{:.2f}".format(row["flicker"], row["true_flicker"]),
                               "{:.1f}".format(secs)])
        except Exception as err:  # DISQUALIFIED, never a crash
            dq.append((rec["name"], "error: {}: {}".format(type(err).__name__, err)))
            continue
        r, e = rows["realistic"], rows["easy"]
        summary.append({"cand": rec["name"], "spell_r": mean(r, "spell"), "spell_sd": float(np.std([x["spell"] for x in r])),
                        "macro_r": mean(r, "macro"), "spell_e": mean(e, "spell"), "macro_e": mean(e, "macro"),
                        "poss": "{:.0f}/{:.0f}".format(mean(r, "n_pred"), mean(r, "n_true")),
                        "dur": "{:.1f}/{:.1f}".format(mean(r, "mean_s"), mean(r, "true_mean_s")),
                        "lt3": mean(r, "lt3"), "flicker": mean(r, "flicker"), "secs": mean(r + e, "secs")})
    summary.sort(key=lambda s: (-s["spell_r"], -s["macro_r"]))
    table = [[i + 1, s["cand"], "{:.4f} +/- {:.3f}".format(s["spell_r"], s["spell_sd"]), "{:.4f}".format(s["macro_r"]),
              "{:.4f}".format(s["spell_e"]), "{:.4f}".format(s["macro_e"]), s["poss"], s["dur"],
              "{:.2f}".format(s["lt3"]), "{:.2f}".format(s["flicker"]), "{:.1f}".format(s["secs"])]
             for i, s in enumerate(summary)]
    table += [["-", n, "DISQUALIFIED({})".format(why)] + [""] * 8 for n, why in dq]
    text = "## Synthetic court (seeds {}, {} min, mean over seeds; ranked by realistic spellF1)\n\n".format(
        ",".join(map(str, seeds)), minutes)
    text += md(["rank", "candidate", "spellF1 realistic", "macroF1 realistic", "spellF1 easy", "macroF1 easy",
                "poss pred/true", "mean dur s pred/true", "share <3s", "flicker/min", "s/run"], table)
    text += "\n\n### Per seed\n\n" + md(["candidate", "difficulty", "seed", "spellF1", "macroF1", "poss pred/true",
                                         "mean dur s pred/true", "share <3s", "flicker/min pred/true", "s"], detail)
    return text, {"summary": summary, "disqualified": dq}


def real_section(records):
    try:
        trk = trk_mod.from_atd()
    except FileNotFoundError as err:
        return "## Real data\n\nSKIPPED: {}".format(err), {"skipped": str(err)}
    rows, dq = [], []
    for rec in records:
        if rec["error"]:
            dq.append((rec["name"], rec["error"]))
            continue
        try:
            labels, secs = run_candidate(rec["mod"], trk)
        except Exception as err:
            dq.append((rec["name"], "error: {}: {}".format(type(err).__name__, err)))
            continue
        for block in trk["blocks"]:
            s = shape_stats(labels, [block])
            rows.append([rec["name"], "half {}".format(block[0]), s["n"], "{:.1f}".format(s["mean_s"]),
                         "{:.2f}".format(s["lt3"]), "{:.2f}".format(s["lt5"]), "{:.0f}".format(s["longest_s"]),
                         "/".join("{:.2f}".format(s["share"][k]) for k in "HADX"), "{:.2f}".format(s["flicker"]),
                         "{:.1f}".format(secs)])
    rows += [[n, "DISQUALIFIED({})".format(why)] + [""] * 8 for n, why in dq]
    text = "## Real data (unlabeled diagnostics only; no ground truth is read)\n\n" + md(
        ["candidate", "half", "possessions", "mean dur s", "share <3s", "share <5s", "longest s",
         "H/A/D/X share", "flicker/min", "s (whole match)"], rows)
    return text, {"rows": rows}


def save(name, text, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name + ".md"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(OUT_DIR, name + ".json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", nargs="*", default=[], help="substrings of cand_*.py names (incumbent always runs)")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--minutes", type=int, default=20, help="synthetic match length")
    ap.add_argument("--real", action="store_true", help="also run unlabeled diagnostics on the real tracking")
    ap.add_argument("--rules-lint", action="store_true", help="only lint candidate sources and exit")
    args = ap.parse_args()

    if args.rules_lint:
        bad = 0
        for path in sorted(glob.glob(os.path.join(EXP_DIR, "cand_*.py"))):
            problems = lint(path)
            bad += bool(problems)
            print("{}: {}".format(os.path.basename(path), "OK" if not problems else "VIOLATION " + "; ".join(problems)))
        sys.exit(1 if bad else 0)

    records = discover(args.cand)
    text, payload = synthetic_section(records, [int(s) for s in args.seeds.split(",")], args.minutes)
    save("court_synth", text, payload)
    print(text)
    if args.real:
        text, payload = real_section(records)
        save("court_real", text, payload)
        print("\n" + text)


if __name__ == "__main__":
    main()
