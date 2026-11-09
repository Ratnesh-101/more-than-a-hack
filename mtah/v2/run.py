"""End-to-end v2: data/ -> submissions/<TEAM>.csv.

    python -m mtah.v2.run --team TEAMNAME            # stream/cached tracking, write submission
    python -m mtah.v2.run --team TEAMNAME --refresh  # rebuild the tracking cache
    python -m mtah.v2.run --team TEAMNAME --variant touch_only   # fallback: candidate 02 alone
    python -m mtah.v2.run --team TEAMNAME --variant semi_markov  # fallback: stack_1 (02 + cand 12)
    python -m mtah.v2.run --team TEAMNAME --data-dir D --out OTHER_FOLDER
    python evaluation/validate_submission.py submissions/TEAMNAME.csv

Deterministic: same data in, byte-identical file out (RULES.md rule 3). Prints only label-free
shape statistics of the result; no ground truth is read anywhere.
"""

import argparse
import hashlib
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.v2 import decode, fallback, tracking  # noqa: E402

TEAM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *("COM%d" % i for i in range(1, 10)),
                    *("LPT%d" % i for i in range(1, 10))}


def safe_output(folder, team):
    """<folder>/<team>.csv, refusing any team name that could escape the folder."""
    if not TEAM_RE.match(team or "") or team.upper() in WINDOWS_RESERVED:
        raise ValueError("invalid team name {!r}: use letters, digits, '_' or '-' "
                         "(max 64, start with a letter or digit, no Windows device names)"
                         .format(team))
    base = os.path.realpath(folder)
    path = os.path.realpath(os.path.join(base, team + ".csv"))
    if os.path.dirname(path) != base:
        raise ValueError("refusing to write outside {}: {}".format(base, path))
    return path


def write(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def shape(block, fps):
    """(possessions, mean duration s, share < 3 s, direct H<->A switches per minute) of one half.
    A possession is a maximal run of one team's label; D (or X) ends it."""
    change = np.flatnonzero(block[1:] != block[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(block)]))
    runs = [(block[s], e - s) for s, e in zip(starts, ends) if block[s] in ("H", "A")]
    dur = np.array([d for _, d in runs], dtype=float) / fps
    flicker = sum(1 for s, e in zip(starts, ends) if e < len(block)
                  and block[s] in ("H", "A") and block[e] in ("H", "A"))
    return (len(runs), float(dur.mean()) if len(dur) else 0.0,
            float((dur < 3.0).mean()) if len(dur) else 0.0, flicker / (len(block) / fps / 60.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="exact team name used at registration")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--variant", choices=("champion",) + fallback.VARIANTS, default="champion",
                    help="champion (default) or a fallback if real data kills its dead-ball premise")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions"),
                    help="folder for <team>.csv (default: submissions/)")
    args = ap.parse_args()
    try:
        safe_output(args.out, args.team)   # fail before the slow part
    except ValueError as err:
        ap.error(str(err))

    trk = tracking.from_atd(data_dir=args.data_dir, refresh=args.refresh)
    if args.variant == "champion":
        labels = decode.decode(trk, decode.emissions(trk))
    else:
        labels = fallback.decode(trk, args.variant)
    out = safe_output(args.out, args.team)
    write(out, labels)

    with open(out, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print("variant", args.variant)
    print("wrote", out)
    print("sha256", digest)
    print("labels", {k: int(np.sum(labels == k)) for k in "HADX"})
    for half, lo, hi in trk["blocks"]:
        n, mean, lt3, flick = shape(labels[lo:hi + 1], trk["fps"])
        print("half {}: possessions {}  mean {:.1f}s  share<3s {:.2f}  flicker {:.2f}/min  "
              "D share {:.2f}".format(half, n, mean, lt3, flick,
                                      float(np.mean(labels[lo:hi + 1] == "D"))))


if __name__ == "__main__":
    main()
