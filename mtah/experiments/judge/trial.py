"""Judge's hidden-seed trial: every eligible candidate, the judge's stacks and the incumbent, in ONE
process, on HIDDEN synthetic seeds 101-110 (realistic and easy, 20 min), through the court's own
machinery (court.run.run_candidate / score_synth, X forward-filled exactly as the scorer does).

    python -m mtah.experiments.judge.trial [--exclude cand_xx ...] [--boot 10000]

Nothing is tuned here: every constant, stack weight and decoder choice was fixed before this ran.

PRE-REGISTERED DECISION RULE (written before any hidden-seed score existed)
  1. Eligible = passes the judge's rules audit and is not struck.
  2. Supported = realistic spellF1 paired mean difference vs the incumbent over seeds 101-110 whose
     bootstrap 95% interval (seeds resampled, 10,000 draws, fixed rng) lies entirely above 0.
  3. Champion = highest realistic mean spellF1 among supported entries; but a stack only beats its
     own single base if the stack-minus-base paired interval also lies above 0 (no extra moving
     parts without proven gain), else the base is preferred.
  4. Runner-up = the best remaining supported entry, preferring a different mechanism from the
     champion so it is a real fallback.
  5. Sanity: an entry below the incumbent on easy spellF1 (paired mean) is not promoted.
  If nothing is supported: say so and recommend the minimal safe set.
"""

import argparse
import glob
import importlib
import json
import os

import numpy as np

from mtah.experiments.court import incumbent, run as court, synth

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
HIDDEN = tuple(range(101, 111))
MINUTES = 20
REF = "cand_02_touch_events"          # base of every stack: stacks are also paired against it
RNG_SEED = 20271108


def entries(exclude):
    paths = sorted(glob.glob(os.path.join(EXP_DIR, "cand_*.py"))) + sorted(
        glob.glob(os.path.join(EXP_DIR, "stack_*.py")))
    out = [("incumbent", incumbent)]
    for p in paths:
        name = os.path.basename(p)[:-3]
        if name in exclude or court.lint(p):
            continue
        out.append((name, importlib.import_module("mtah.experiments." + name)))
    return out


def boot_ci(d, rng, b):
    idx = rng.integers(0, len(d), (b, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    games = {}
    for diff in ("realistic", "easy"):
        for s in HIDDEN:
            trk, truth = synth.make(s, MINUTES, diff)
            games[diff, s] = (trk, court.pm.label_codes(truth), court.shape_stats(truth, trk["blocks"]))

    res = {}
    for name, mod in entries(set(args.exclude)):
        rows = {}
        try:
            for (diff, s), (trk, codes, st) in games.items():
                labels, secs = court.run_candidate(mod, trk)
                r = court.score_synth(labels, codes, st, trk["blocks"])
                r["secs"] = secs
                rows[diff, s] = r
        except Exception as err:  # reported, never fatal
            print("ERROR {}: {}: {}".format(name, type(err).__name__, err))
            continue
        res[name] = rows
        print("done {} ({:.1f} s)".format(name, sum(r["secs"] for r in rows.values())), flush=True)

    rng = np.random.default_rng(RNG_SEED)
    get = lambda name, diff, key: np.array([res[name][diff, s][key] for s in HIDDEN], dtype=float)
    summary = []
    for name in res:
        row = {"cand": name}
        for diff in ("realistic", "easy"):
            sp, mf = get(name, diff, "spell"), get(name, diff, "macro")
            d = sp - get("incumbent", diff, "spell")
            row[diff] = {"spell": float(sp.mean()), "spell_sd": float(sp.std()), "macro": float(mf.mean()),
                         "macro_sd": float(mf.std()), "n_pred": float(get(name, diff, "n_pred").mean()),
                         "n_true": float(get(name, diff, "n_true").mean()),
                         "lt3": float(get(name, diff, "lt3").mean()),
                         "flicker": float(get(name, diff, "flicker").mean()),
                         "true_flicker": float(get(name, diff, "true_flicker").mean()),
                         "d_inc": float(d.mean()), "won": int((d > 1e-9).sum()), "lost": int((d < -1e-9).sum()),
                         "ci_inc": boot_ci(d, rng, args.boot), "per_seed": sp.tolist()}
            if REF in res and name.startswith("stack_"):
                dr = sp - get(REF, diff, "spell")
                row[diff].update(d_ref=float(dr.mean()), won_ref=int((dr > 1e-9).sum()),
                                 ci_ref=boot_ci(dr, rng, args.boot))
        row["secs"] = float(np.mean([r["secs"] for r in res[name].values()]))
        summary.append(row)
    summary.sort(key=lambda r: -r["realistic"]["spell"])

    f4 = "{:.4f}".format
    ci = lambda c: "[{:+.4f}, {:+.4f}]".format(*c)
    head = ["rank", "candidate", "spellF1 real (mean +/- sd)", "macroF1 real", "poss pred/true", "share <3s",
            "flicker pred/true", "d spell vs inc", "won/lost (of 10)", "bootstrap 95% CI", "supported"]
    text = "## Hidden-seed trial: seeds {}-{}, {} min, REALISTIC\n\n".format(HIDDEN[0], HIDDEN[-1], MINUTES)
    text += court.md(head, [[i + 1, r["cand"], "{} +/- {:.3f}".format(f4(r["realistic"]["spell"]), r["realistic"]["spell_sd"]),
                             "{} +/- {:.3f}".format(f4(r["realistic"]["macro"]), r["realistic"]["macro_sd"]),
                             "{:.1f}/{:.1f}".format(r["realistic"]["n_pred"], r["realistic"]["n_true"]),
                             "{:.3f}".format(r["realistic"]["lt3"]),
                             "{:.2f}/{:.2f}".format(r["realistic"]["flicker"], r["realistic"]["true_flicker"]),
                             "{:+.4f}".format(r["realistic"]["d_inc"]),
                             "{}/{}".format(r["realistic"]["won"], r["realistic"]["lost"]), ci(r["realistic"]["ci_inc"]),
                             "YES" if r["realistic"]["ci_inc"][0] > 0 else "no"]
                            for i, r in enumerate(summary)])
    easy = sorted(summary, key=lambda r: -r["easy"]["spell"])
    text += "\n\n## Hidden-seed trial: EASY\n\n" + court.md(
        ["candidate", "spellF1 easy", "macroF1 easy", "poss pred/true", "share <3s", "d spell vs inc", "won/lost",
         "bootstrap 95% CI"],
        [[r["cand"], "{} +/- {:.3f}".format(f4(r["easy"]["spell"]), r["easy"]["spell_sd"]), f4(r["easy"]["macro"]),
          "{:.1f}/{:.1f}".format(r["easy"]["n_pred"], r["easy"]["n_true"]), "{:.3f}".format(r["easy"]["lt3"]),
          "{:+.4f}".format(r["easy"]["d_inc"]), "{}/{}".format(r["easy"]["won"], r["easy"]["lost"]),
          ci(r["easy"]["ci_inc"])] for r in easy])
    stacks = [r for r in summary if "d_ref" in r["realistic"]]
    if stacks:
        text += "\n\n## Stacks vs their base ({})\n\n".format(REF) + court.md(
            ["stack", "real d vs 02", "won (of 10)", "real 95% CI", "easy d vs 02", "easy 95% CI"],
            [[r["cand"], "{:+.4f}".format(r["realistic"]["d_ref"]), r["realistic"]["won_ref"], ci(r["realistic"]["ci_ref"]),
              "{:+.4f}".format(r["easy"]["d_ref"]), ci(r["easy"]["ci_ref"])] for r in stacks])
    text += "\n\n## Per seed, realistic spellF1\n\n" + court.md(
        ["candidate"] + [str(s) for s in HIDDEN],
        [[r["cand"]] + [f4(v) for v in r["realistic"]["per_seed"]] for r in summary])
    text += "\n\nmean s per 20-min match (emissions + decode): " + ", ".join(
        "{} {:.1f}".format(r["cand"], r["secs"]) for r in summary)
    print("\n" + text)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "trial.md"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(OUT_DIR, "trial.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
