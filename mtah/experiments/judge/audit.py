"""Judge's rules audit: lint, determinism, NaN/+inf, trk mutation, runtime on a tiled 199,319-frame
Trk, and DEV-seed (1-3) sensitivity probes for constants that coincide with synth.py constants.

    python -m mtah.experiments.judge.audit [--no-probes]

Hidden seeds (101-110) are never touched here. No ground truth of the real match is read; the
probes score synthetic truth only, through the court's own scorer.
"""

import argparse
import glob
import hashlib
import importlib
import os
import time

import numpy as np

from mtah.experiments.court import incumbent, run as court, synth
from mtah.experiments.court.trk import build

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TILED = 199319
TILED_BLOCKS = [(1, 2000, 2000 + 79000 - 1), (2, 100000, 100000 + 79857 - 1)]   # 158,857 graded


def modules():
    paths = sorted(glob.glob(os.path.join(EXP_DIR, "cand_*.py"))) + sorted(
        glob.glob(os.path.join(EXP_DIR, "stack_*.py")))
    out = [("incumbent", None, incumbent)]
    for p in paths:
        name = os.path.basename(p)[:-3]
        out.append((name, p, importlib.import_module("mtah.experiments." + name)))
    return out


def tiled_trk(seed=1):
    trk, _ = synth.make(seed, 20, "realistic")
    reps = -(-N_TILED // len(trk["bx"]))
    tile = lambda a: np.concatenate([a] * reps, axis=0)[:N_TILED]
    return build(tile(trk["px"]), tile(trk["py"]), tile(trk["ps"]), trk["team"], tile(trk["bx"]),
                 tile(trk["by"]), trk["pitch"], TILED_BLOCKS)


def digest(trk):
    h = hashlib.sha256()
    for k in ("px", "py", "ps", "team", "bx", "by"):
        h.update(trk[k].tobytes())
    h.update(repr((trk["pitch"], trk["fps"], trk["blocks"])).encode())
    return h.hexdigest()[:16]


def audit_one(name, path, mod, trk):
    row = {"cand": name, "lint": "OK" if path is None or not court.lint(path) else "; ".join(court.lint(path))}
    before = digest(trk)
    t0 = time.perf_counter()
    e1 = np.asarray(mod.emissions(trk), dtype=np.float64)
    row["emit_s"] = time.perf_counter() - t0
    e2 = np.asarray(mod.emissions(trk), dtype=np.float64)
    row["shape_ok"] = e1.shape == (N_TILED, 3)
    row["nan"] = int(np.isnan(e1).sum())
    row["pinf"] = int((e1 == np.inf).sum())
    row["ninf"] = int((e1 == -np.inf).sum())
    row["emit_det"] = e1.tobytes() == e2.tobytes()
    t0 = time.perf_counter()
    l1, _ = court.run_candidate(mod, trk)
    row["total_s"] = time.perf_counter() - t0
    l2, _ = court.run_candidate(mod, trk)
    row["label_det"] = bool(np.array_equal(l1, l2))
    row["mutated"] = digest(trk) != before or any(trk[k].flags.writeable for k in ("px", "bx", "team"))
    return row


def probe(mod, attr, values, seeds=(1, 2, 3)):
    """Mean realistic spellF1 on DEV seeds for each value of a module constant (restored after)."""
    orig = getattr(mod, attr)
    games = []
    for s in seeds:
        trk, truth = synth.make(s, 20, "realistic")
        games.append((trk, court.pm.label_codes(truth), court.shape_stats(truth, trk["blocks"])))
    res = []
    try:
        for v in [orig] + list(values):
            setattr(mod, attr, v)
            sc = [court.score_synth(court.run_candidate(mod, t)[0], c, st, t["blocks"])["spell"]
                  for t, c, st in games]
            res.append((v, float(np.mean(sc))))
    finally:
        setattr(mod, attr, orig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-probes", action="store_true")
    args = ap.parse_args()
    mods = modules()
    trk = tiled_trk()
    print("tiled Trk: {} frames, blocks {}".format(N_TILED, TILED_BLOCKS))
    rows = [audit_one(n, p, m, trk) for n, p, m in mods]
    print(court.md(["candidate", "lint", "shape", "NaN", "+inf", "-inf", "emissions det.", "labels det.",
                    "trk mutated", "emissions s", "emissions+decode s"],
                   [[r["cand"], r["lint"], r["shape_ok"], r["nan"], r["pinf"], r["ninf"], r["emit_det"],
                     r["label_det"], r["mutated"], "{:.2f}".format(r["emit_s"]), "{:.2f}".format(r["total_s"])]
                    for r in rows]))
    if args.no_probes:
        return
    by = {n: m for n, _, m in mods}
    print("\nDEV-seed probes (seeds 1-3 realistic, mean spellF1; first value = as filed)")
    for cand, attr, vals, why in (
            ("cand_01_incumbent_plus", "GLITCH_M", [1e9], "glitch filter sized to synth's 2-6 frame glitches; 1e9 = off"),
            ("cand_08_dead_ball", "HSR_MPS", [5.0, 6.0], "equals synth chaser speed 5.5 m/s"),
            ("cand_08_dead_ball", "WALK_MPS", [1.8, 2.2], "sits just above synth dead-ball walk cap 1.6 m/s"),
            ("cand_12_semi_markov", "LIVE_MEDIAN_S", [6.0, 15.0], "equals synth's stated ~10 s possession median"),
            ("cand_02_touch_events", "STILL_R", [0.5, 2.0], "still-ball radius vs synth's 0.2 m ball noise"),
            ("cand_03_pass_chain", "TOUCH_M", [1.2, 2.0], "justified by synth's 0.7 m carry offset")):
        print("  {}.{} ({}): {}".format(cand, attr, why, ", ".join(
            "{:g} -> {:.4f}".format(v, s) for v, s in probe(by[cand], attr, vals))))


if __name__ == "__main__":
    main()
