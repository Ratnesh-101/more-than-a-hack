"""Pipeline smoke test on SYNTHETIC tracking, so the code is proven before the real data arrives.

    python -m mtah.selfcheck                     # validate only
    python -m mtah.selfcheck --keep out.csv      # also keep the synthetic submission (determinism)

Builds a fake bundle in a temp dir (22 players, a ball that alternates between teams in ~15 s
spells with stoppages) covering BOTH halves, runs the champion pipeline (mtah.v2 tracking.from_atd
-> emissions -> decode -> write), validates the file with the official checker and prints the
recovered possession count per half. Uses no real data and no
ground truth. Seeded, so two runs must produce byte-identical files.
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

META = """<?xml version="1.0"?>
<Root><Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
<Players>{}</Players></Root>"""


def make_bundle(folder, rng, halves=(1, 2)):
    """Write a synthetic ATD bundle. Returns {half: number of true H/A possessions}."""
    tac = os.path.join(folder, "tactical")
    os.makedirs(tac)
    players = "".join('<Player id="P{0}" teamId="{1}"><ShirtNumber>{0}</ShirtNumber></Player>'
                      .format(i + 1, "T1" if i < 11 else "T2") for i in range(22))
    with open(os.path.join(tac, "BAR-ATH_tactical_atd.xml"), "w") as fh:
        fh.write(META.format(players))

    n_true = {}
    with open(os.path.join(tac, "BAR-ATH_tactical_atd.txt"), "w") as fh:
        for half, lo, hi in half_video_blocks("BAR-ATH"):
            if half not in halves:
                continue
            truth = []
            t, state = lo, half - 1   # 2nd half kicked off by the other side
            while t <= hi:
                dur = int(rng.integers(150, 600))
                dead = rng.random() < 0.25
                for vf in range(t, min(t + dur, hi + 1)):
                    pos = rng.random((22, 2))
                    speed = rng.random(22) * (0.5 if dead else 4)
                    carrier = (0 if state == 0 else 11) + int(rng.integers(0, 11))
                    if dead or rng.random() < 0.4:   # 40% missing ball, like the real data
                        ball = "NaN,NaN,NaN"
                    else:
                        bx, by = pos[carrier] + rng.normal(0, 0.005, 2)
                        ball = "{:.4f},{:.4f},0".format(bx, by)
                    groups = ";".join("{:.4f},{:.4f},{:.2f}".format(x, y, s)
                                      for (x, y), s in zip(pos, speed))
                    fh.write("{}:{}:{}\n".format(vf + 1, groups, ball))
                truth.append("D" if dead else "HA"[state])
                t += dur
                if not dead:
                    state = 1 - state
            n_true[half] = sum(1 for x in truth if x != "D")
    return n_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=None, help="also copy the synthetic submission here")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        n_true = make_bundle(tmp, rng)
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
            n_pred, mean, _, _ = run.shape(labels[lo:hi + 1], trk["fps"])
            print("synthetic half {}: true possessions {}, predicted {} (mean {:.1f}s)".format(
                half, n_true[half], n_pred, mean))
        print("tracking.from_atd {:.1f}s, emissions+decode {:.2f}s on both synthetic halves".format(
            t_trk, t_dec))
        if args.keep:
            os.makedirs(os.path.dirname(os.path.abspath(args.keep)), exist_ok=True)
            shutil.copyfile(out, args.keep)
            print("kept", args.keep)
        print("SELFCHECK", "OK" if rc == 0 and len(labels) == 199319 else "FAILED")


if __name__ == "__main__":
    main()
