"""Robustness of the champion chain (tracking.from_atd -> decode.emissions -> decode.decode) to the
ways REAL ATD data may differ from the synthetic data the pipeline has only ever seen.

    python -m mtah.v2.tests.test_robust

Every case writes a small ATD bundle into a temp dir (two short graded halves; tracking.games is
patched to those blocks for the duration of a case), reads it back through the real parser and
decodes it. Hard checks (FAIL): no exception, emissions without NaN/+inf, labels in H/A/D inside the
blocks and X outside, identical results on a second run (including a re-parse and a cache load),
and the two fallback variants (v2.fallback) not crashing and keeping the label alphabet.
Suspicious behaviour (WARN, not a failure): a whole half in one label, D share > 0.6 or ~0, -inf
emissions. Case 7 is an informational probe. Case 9 is the 199,319-frame wall-time smoke on the
tiled synthetic Trk. Seeded; writes only inside the temp dir.
"""

import os
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from mtah.v2 import decode, run, tracking  # noqa: E402

try:
    from mtah.v2 import fallback  # noqa: E402
except ImportError:                 # fallback.py is optional for this test
    fallback = None

from mtah.v2.tests.simbundle import (AWAY_SLOTS, BLOCKS, FPS, HOME_SLOTS, L, N, NAN3, S,  # noqa: E402,F401
                                      W, copy_sim, make_sim, write_bundle)

TIME_LIMIT_S = 60.0


# ----------------------------------------------------------------------------- case machinery

def read_back(sim):
    """Write the bundle and parse it with the real tracking.from_atd (games patched to the mini
    blocks); also re-parse with refresh and from the npz cache to prove parse determinism."""
    g = tracking.games
    saved = (g.half_video_blocks, g.source_frames)
    g.half_video_blocks = lambda game, adjust=0: list(BLOCKS)
    g.source_frames = lambda source="tactical", game=None: N
    try:
        with tempfile.TemporaryDirectory() as tmp:
            write_bundle(tmp, sim)
            trk = tracking.from_atd(data_dir=tmp)
            trk2 = tracking.from_atd(data_dir=tmp, refresh=True)
            trk3 = tracking.from_atd(data_dir=tmp)       # npz cache path
    finally:
        g.half_video_blocks, g.source_frames = saved
    for k in ("px", "py", "ps", "team", "bx", "by"):
        assert np.array_equal(trk[k], trk2[k], equal_nan=True), "re-parse differs in " + k
        assert np.array_equal(trk[k], trk3[k], equal_nan=True), "cache load differs in " + k
    return trk


def evaluate(trk):
    """Hard checks + stats. Returns (stats dict, list of fail strings, list of warn strings)."""
    fails, warns = [], []
    n = len(trk["bx"])
    e = decode.emissions(trk)
    if e.shape != (n, 3):
        fails.append("emission shape {}".format(e.shape))
    if np.isnan(e).any():
        fails.append("NaN in emissions ({})".format(int(np.isnan(e).sum())))
    if (e == np.inf).any():
        fails.append("+inf in emissions")
    labels = decode.decode(trk, e)
    inside = np.zeros(len(labels), bool)
    for _, lo, hi in trk["blocks"]:
        inside[lo:hi + 1] = True
    if not np.isin(labels[inside], list("HAD")).all():
        fails.append("labels outside HAD inside blocks")
    if not (labels[~inside] == "X").all():
        fails.append("label other than X outside blocks")
    if not (np.array_equal(labels, decode.decode(trk, decode.emissions(trk)))
            and np.array_equal(e, decode.emissions(trk))):
        fails.append("not deterministic")
    n_ninf = int(np.isneginf(e[inside]).sum())
    if n_ninf:
        warns.append("{} -inf emission entries inside blocks".format(n_ninf))
    if fallback is not None:
        for variant in fallback.VARIANTS:
            try:
                fl = fallback.decode(trk, variant)
                if not (np.isin(fl[inside], list("HAD")).all() and (fl[~inside] == "X").all()):
                    fails.append("fallback {} bad labels".format(variant))
            except Exception as exc:     # noqa: BLE001
                fails.append("fallback {} raised {}: {}".format(variant, type(exc).__name__, exc))
    st = {"labels": labels, "H": [], "A": [], "D": [], "poss": []}
    tot = {k: 0 for k in "HAD"}
    cnt = 0
    for h, lo, hi in trk["blocks"]:
        blk = labels[lo:hi + 1]
        for k in "HAD":
            tot[k] += int((blk == k).sum())
        cnt += len(blk)
        st["poss"].append(run.shape(blk, trk["fps"])[0])
        sh = {k: float((blk == k).mean()) for k in "HAD"}
        if max(sh.values()) > 0.999:
            warns.append("half {} decoded as ONE label ({})".format(h, max(sh, key=sh.get)))
        if sh["D"] > 0.6:
            warns.append("half {} D share {:.2f} > 0.6".format(h, sh["D"]))
        if sh["D"] < 0.005 and max(sh.values()) <= 0.999:
            warns.append("half {} D share ~0 ({:.3f})".format(h, sh["D"]))
    for k in "HAD":
        st[k] = tot[k] / float(max(cnt, 1))
    return st, fails, warns


RESULTS = []   # (name, status, warns, fails)


def report(name, st, fails, warns, informational=False):
    status = "FAIL" if fails else ("INFO" if informational else "PASS")
    if st is not None:
        print("{:<34s} H {:.2f} A {:.2f} D {:.2f} | poss {:>3d}/{:<3d} | {}".format(
            name, st["H"], st["A"], st["D"], st["poss"][0], st["poss"][1], status))
    else:
        print("{:<34s} {}".format(name, status))
    for f in fails:
        print("    FAIL: " + f)
    for w in warns:
        print("    WARN: " + w)
    RESULTS.append((name, status, warns, fails))


def case(name, sim, informational=False):
    try:
        trk = read_back(sim)
        st, fails, warns = evaluate(trk)
    except Exception as exc:     # noqa: BLE001 - the point is to catch anything
        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        report(name, None, ["EXCEPTION {}: {} at {}:{} in {}".format(
            type(exc).__name__, exc, os.path.basename(tb.filename), tb.lineno, tb.name)], [])
        return
    report(name, st, fails, warns, informational)


# ----------------------------------------------------------------------------- mutations

def in_blocks_ranges(rng, k, lo_len, hi_len):
    """k random (a, b) ranges inside the graded blocks."""
    out = []
    for _ in range(k):
        _, lo, hi = BLOCKS[int(rng.integers(0, 2))]
        ln = int(rng.integers(lo_len, hi_len))
        a = int(rng.integers(lo, hi - ln))
        out.append((a, a + ln))
    return out


def ball_off(sim, a, b):
    sim["BX"][a:b] = np.nan
    sim["BY"][a:b] = np.nan


def build_cases(base):
    rng = np.random.default_rng(1234)
    cases = []

    def add(name, fn, informational=False):
        s = copy_sim(base)
        fn(s)
        cases.append((name, s, informational))

    add("0 clean reference", lambda s: None)

    # 1. ball missing
    add("1a ball NaN 100%", lambda s: ball_off(s, 0, N))

    def ball_keep(s, frac):
        keep = rng.random(N) < frac
        s["BX"][~keep] = np.nan
        s["BY"][~keep] = np.nan
    add("1b ball NaN 95%", lambda s: ball_keep(s, 0.05))

    def one_in_25(s):
        keep = (np.arange(N) % 25) == 0
        s["BX"][~keep] = np.nan
        s["BY"][~keep] = np.nan
    add("1c ball on 1 frame in 25", one_in_25)

    # 2. speed field
    def spd_nan(s):
        s["S"][:] = np.nan
    add("2a speed NaN everywhere", spd_nan)

    def spd_half(s):
        s["S"][rng.random(s["S"].shape) < 0.5] = np.nan
    add("2b speed NaN on random 50%", spd_half)

    # 3. missing players
    def no_away(s):
        for a, b in in_blocks_ranges(rng, 8, 500, 1500):
            s["X"][a:b, AWAY_SLOTS] = np.nan
            s["Y"][a:b, AWAY_SLOTS] = np.nan
    add("3a n_away=0 stretches", no_away)

    def one_team_always(s):
        s["X"][:, AWAY_SLOTS] = np.nan
        s["Y"][:, AWAY_SLOTS] = np.nan
    add("3b away never tracked", one_team_always)

    def no_players(s):
        for a, b in in_blocks_ranges(rng, 8, 100, 1500):
            s["X"][a:b, :] = np.nan
            s["Y"][a:b, :] = np.nan
    add("3c zero players stretches", no_players)

    def empty_player_field(s):
        def hook(lines):
            out = []
            for i, ln in enumerate(lines):
                if i % 7 == 0:
                    p = ln.split(":")
                    ln = p[0] + "::" + p[2]
                out.append(ln)
            return out
        s["line_hook"] = hook
    add("3d empty player field lines", empty_player_field)

    # 4. metadata mismatch
    def meta_short(s):
        s["teams"] = s["teams"][:30]           # away slots 30..33 are data but not in metadata
    add("4a slots beyond metadata list", meta_short)

    def meta_23(s):
        s["teams"] = s["teams"][:23]           # no away slot in metadata at all
    add("4b metadata lists home only", meta_23)

    def team_missing(s):
        for i in (3, 5, 8, 25, 27, 30):
            s["teams"][i] = None               # teamId attribute absent
    add("4c teamId absent on 6 slots", team_missing)

    def extra_groups(s):
        s["n_groups"] = 52
    add("4d 52 groups per line (>46)", extra_groups)

    def meta_long(s):
        s["teams"] = s["teams"] + ["T2"] * 14   # 60 metadata players
    add("4e metadata lists 60 players", meta_long)

    # 5. absent frames
    def gaps(s):
        s["present"][BLOCKS[0][1] + 2000:BLOCKS[0][1] + 2025] = False     # 1 s
        s["present"][BLOCKS[1][1] + 1500:BLOCKS[1][1] + 3000] = False     # 60 s
    add("5a absent frames 1 s + 60 s", gaps)

    def ends_early(s):
        s["present"][BLOCKS[1][2] - 2000:] = False
    add("5b ATD ends 80 s before block", ends_early)

    def ends_in_half1(s):
        s["present"][BLOCKS[0][2] - 100:] = False
    add("5c ATD ends inside half 1", ends_in_half1)

    # 6. glitches
    def glitch(s):
        for a, _ in in_blocks_ranges(rng, 40, 5, 6):
            k = int(rng.integers(1, 4))
            d = rng.uniform(30, 80) * rng.choice([-1, 1])
            ang = rng.uniform(0, 6.28)
            s["BX"][a:a + k] = np.where(np.isnan(s["BX"][a:a + k]), 0.5, s["BX"][a:a + k]) + \
                d * np.cos(ang) / L
            s["BY"][a:a + k] = np.where(np.isnan(s["BY"][a:a + k]), 0.5, s["BY"][a:a + k]) + \
                d * np.sin(ang) / W
    add("6a ball jumps 30-80 m x1-3 fr", glitch)

    def ball_far(s):
        for a, b in in_blocks_ranges(rng, 20, 50, 300):
            s["BX"][a:b] = rng.choice([-30.0 / L, 1 + 30.0 / L])
            s["BY"][a:b] = rng.choice([-30.0 / W, 1 + 30.0 / W])
    add("6b ball +-30 m outside pitch", ball_far)

    def ball_far_all(s):
        s["BX"][:] = 1.0 + 30.0 / L
        s["BY"][:] = 0.5
    add("6c ball 30 m outside all frames", ball_far_all)

    def players_out(s):
        m = rng.random(s["X"].shape) < 0.05
        s["X"][m] = rng.uniform(-0.5, 1.5, int(m.sum()))
        s["Y"][m] = rng.uniform(-0.5, 1.5, int(m.sum()))
        m2 = rng.random(s["X"].shape) < 0.001
        s["X"][m2] = 1000.0
    add("6d players outside [0,1]", players_out)

    # 7. unit probes
    def spd_norm(s):
        s["S"] = s["S"] / L
    add("7a speed pitch-normalised", spd_norm, True)

    def spd_kmh(s):
        s["S"] = s["S"] * 3.6
    add("7b speed in km/h (probe)", spd_kmh, True)

    # 8. half edges
    def start_gap(s):
        for _, lo, _hi in BLOCKS:
            ball_off(s, lo, lo + 90 * FPS)
    add("8a half starts 90 s no ball", start_gap)

    def end_gap(s):
        for _, _lo, hi in BLOCKS:
            ball_off(s, hi - 500, hi + 1500)
    add("8b half ends mid ball gap", end_gap)

    def both_gap(s):
        start_gap(s)
        end_gap(s)
    add("8c both edge gaps", both_gap)

    # 10. malformed text (extra)
    def trunc_last(s):
        def hook(lines):
            return lines[:-1] + [lines[-1][:len(lines[-1]) // 2]]
        s["line_hook"] = hook
    add("10a last line truncated", trunc_last)

    def junk_lines(s):
        def hook(lines):
            return ["", "# header"] + lines[:100] + ["", "garbage"] + lines[100:]
        s["line_hook"] = hook
    add("10b blank/junk lines", junk_lines)

    def lower_nan(s):
        def hook(lines):
            return [ln.replace("NaN", "nan") if i % 3 == 0 else ln for i, ln in enumerate(lines)]
        s["line_hook"] = hook
    add("10c lowercase nan tokens", lower_nan)

    def dup_unsorted(s):
        def hook(lines):
            return lines[:3000] + lines[2000:2100] + lines[3000:]   # repeated, out-of-order frames
        s["line_hook"] = hook
    add("10d repeated/out-of-order ids", dup_unsorted)

    return cases


# ----------------------------------------------------------------------------- case 9

def smoke_big():
    from mtah.experiments.court import synth   # tests may import the synthetic generator
    n = 199319
    trk0, _ = synth.make(1, 20, "realistic")
    reps = -(-n // len(trk0["bx"]))
    tile = lambda a: np.concatenate([a] * reps, axis=0)[:n]  # noqa: E731
    rng = np.random.default_rng(9)
    bx, by = tile(trk0["bx"]).copy(), tile(trk0["by"]).copy()
    drop = rng.random(n) < 0.6
    bx[drop], by[drop] = np.nan, np.nan
    blocks = [(1, 2000, 2000 + 79000 - 1), (2, 100000, 100000 + 79857 - 1)]
    big = tracking.build(tile(trk0["px"]), tile(trk0["py"]), tile(trk0["ps"]), trk0["team"],
                         bx, by, trk0["pitch"], blocks)
    t0 = time.perf_counter()
    e = decode.emissions(big)
    labels = decode.decode(big, e)
    dt = time.perf_counter() - t0
    fails = []
    if dt >= TIME_LIMIT_S:
        fails.append("emissions+decode {:.1f}s >= {:.0f}s".format(dt, TIME_LIMIT_S))
    if np.isnan(e).any() or (e == np.inf).any():
        fails.append("NaN/+inf in emissions")
    inside = np.zeros(n, bool)
    for _, lo, hi in blocks:
        inside[lo:hi + 1] = True
    if not (np.isin(labels[inside], list("HAD")).all() and (labels[~inside] == "X").all()):
        fails.append("bad label alphabet")
    sh = {k: float((labels[inside] == k).mean()) for k in "HAD"}
    poss = [run.shape(labels[lo:hi + 1], 25)[0] for _, lo, hi in blocks]
    st = {"H": sh["H"], "A": sh["A"], "D": sh["D"], "poss": poss}
    print("    (199,319 frames, ~60%+ ball missing: emissions+decode {:.1f}s, limit {:.0f}s)".format(
        dt, TIME_LIMIT_S))
    report("9 wall-time smoke 199,319 fr", st, fails, [])


# ----------------------------------------------------------------------------- main

def main():
    t0 = time.perf_counter()
    base = make_sim(7)
    for name, sim, info in build_cases(base):
        case(name, sim, info)
    try:
        smoke_big()
    except Exception as exc:     # noqa: BLE001
        report("9 wall-time smoke 199,319 fr", None,
               ["EXCEPTION {}: {}".format(type(exc).__name__, exc)], [])
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    warns = [r for r in RESULTS if r[2]]
    print()
    print("{} cases, {} FAIL, {} with WARN, {:.0f}s".format(
        len(RESULTS), len(fails), len(warns), time.perf_counter() - t0))
    if warns:
        print("suspicious (WARN): " + "; ".join(r[0] for r in warns))
    if fails:
        print("ROBUST FAILED: " + "; ".join(r[0] for r in fails))
        sys.exit(1)
    print("ROBUST OK")


if __name__ == "__main__":
    main()
