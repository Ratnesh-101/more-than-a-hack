"""Pipeline plausibility check on SYNTHETIC tracking, so the code is proven before the real data arrives.

    python -m mtah.selfcheck                     # validate + plausibility
    python -m mtah.selfcheck --keep out.csv      # also keep the synthetic submission

Builds a fake bundle in a temp dir with mtah.v2.tests.simbundle (22 players that move smoothly, a
ball passed between them with ~35% of frames lost in occlusion runs, ~15 s possessions alternating
H/A, 25% of spells stoppages) covering BOTH graded halves at the real 199,319-frame length, runs the
champion chain (tracking.from_atd -> emissions -> decode -> write), validates the file with the
official checker and asserts PLAUSIBILITY against the simulator's own truth:

  - file valid, 199,319 rows, X exactly outside the two graded blocks, only H/A/D inside;
  - both H and A present in every half;
  - D share per half in [0.05, 0.40]: the simulator makes 25% of spells stoppages;
  - possessions per half within 0.5x-2x of the simulator's count of live spells;
  - share of possessions < 3 s <= 0.15: every simulated possession lasts >= 6 s;
  - a second run (re-parse with refresh) gives byte-identical labels.

Uses no real data and no ground truth. Seeded, so two runs give byte-identical files.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks  # noqa: E402
from mtah.v2 import decode, tracking  # noqa: E402
from mtah.v2 import run  # noqa: E402
from mtah.v2.tests import simbundle  # noqa: E402

N_FRAMES = 199319
D_RANGE = (0.05, 0.40)    # simulator truth: 25% of spells are stoppages
POSS_RATIO = (0.5, 2.0)   # predicted / true possessions
MAX_SHORT = 0.15          # share of possessions < 3 s; simulated possessions last >= 6 s
SEED = 0


def make_bundle(folder, rng, halves=(1, 2)):
    """Write a synthetic ATD bundle (real graded blocks). Returns {half: true H/A possessions}.
    Kept for mtah/v2/tests/test_e2e.py: the seed is drawn from rng."""
    blocks = [b for b in half_video_blocks("BAR-ATH") if b[0] in halves]
    sim = simbundle.make_sim(int(rng.integers(0, 2 ** 31 - 1)), N_FRAMES, blocks)
    simbundle.write_bundle(folder, sim)
    return dict(sim["n_true"])


def plausibility(labels, blocks, n_true, dead_true):
    """List of failure strings (empty = plausible) for labels against the simulator's truth."""
    bad = []
    if len(labels) != N_FRAMES:
        bad.append("{} rows, expected {}".format(len(labels), N_FRAMES))
    inside = np.zeros(len(labels), bool)
    for _, lo, hi in blocks:
        inside[lo:hi + 1] = True
    if not (labels[~inside] == "X").all():
        bad.append("label other than X outside the graded blocks")
    if not np.isin(labels[inside], list("HAD")).all():
        bad.append("label outside H/A/D inside the graded blocks")
    for half, lo, hi in blocks:
        blk = labels[lo:hi + 1]
        poss, _, short, _ = run.shape(blk, 25)
        share = {k: float((blk == k).mean()) for k in "HAD"}
        if share["H"] == 0 or share["A"] == 0:
            bad.append("half {}: H or A missing {}".format(half, share))
        if not D_RANGE[0] <= share["D"] <= D_RANGE[1]:
            bad.append("half {}: D share {:.2f} outside {} (simulator truth {:.2f})".format(
                half, share["D"], D_RANGE, dead_true[half]))
        if not POSS_RATIO[0] * n_true[half] <= poss <= POSS_RATIO[1] * n_true[half]:
            bad.append("half {}: {} possessions vs {} true (allowed {}x-{}x)".format(
                half, poss, n_true[half], *POSS_RATIO))
        if short > MAX_SHORT:
            bad.append("half {}: {:.2f} of possessions < 3 s (max {})".format(half, short, MAX_SHORT))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=None, help="also copy the synthetic submission here")
    args = ap.parse_args()

    blocks = half_video_blocks("BAR-ATH")
    t0 = time.perf_counter()
    sim = simbundle.make_sim(SEED, N_FRAMES, blocks)
    with tempfile.TemporaryDirectory() as tmp:
        simbundle.write_bundle(tmp, sim)
        t_sim = time.perf_counter() - t0
        t0 = time.perf_counter()
        trk = tracking.from_atd(data_dir=tmp)
        t_trk = time.perf_counter() - t0
        t0 = time.perf_counter()
        labels = decode.decode(trk, decode.emissions(trk))
        t_dec = time.perf_counter() - t0
        out = os.path.join(tmp, "selfcheck.csv")
        run.write(out, labels)
        rc = subprocess.call([sys.executable, os.path.join(ROOT, "evaluation",
                                                           "validate_submission.py"), out])
        print()
        for half, lo, hi in trk["blocks"]:
            n_pred, mean, short, _ = run.shape(labels[lo:hi + 1], trk["fps"])
            sh = {k: float(np.mean(labels[lo:hi + 1] == k)) for k in "HAD"}
            print("synthetic half {}: true possessions {}, predicted {} (mean {:.1f}s, share<3s "
                  "{:.2f}); H {:.2f} A {:.2f} D {:.2f} (true D {:.2f})".format(
                      half, sim["n_true"][half], n_pred, mean, short, sh["H"], sh["A"], sh["D"],
                      sim["dead_share"][half]))
        bad = plausibility(labels, trk["blocks"], sim["n_true"], sim["dead_share"])
        t0 = time.perf_counter()
        again = decode.decode(*(lambda t: (t, decode.emissions(t)))(
            tracking.from_atd(data_dir=tmp, refresh=True)))
        if not np.array_equal(labels, again):
            bad.append("second run (re-parse) differs: not deterministic")
        t_again = time.perf_counter() - t0
        print("build {:.1f}s, tracking.from_atd {:.1f}s, emissions+decode {:.2f}s, "
              "second run {:.1f}s".format(t_sim, t_trk, t_dec, t_again))
        for b in bad:
            print("IMPLAUSIBLE:", b)
        if args.keep:
            os.makedirs(os.path.dirname(os.path.abspath(args.keep)), exist_ok=True)
            shutil.copyfile(out, args.keep)
            print("kept", args.keep)
        ok = rc == 0 and not bad
        print("SELFCHECK", "OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
