"""Real-data acceptance gate for a possession CSV, from UNLABELED statistics only.

    python -m analysis.real_checks --csv submissions/nirakshak.csv [--compare other.csv [--compare-is-v1]]

Implements the acceptance criteria of mtah/experiments/briefs/VERDICT.md section 5 item 4 for each
graded half (games.half_video_blocks) plus sanity checks of the dead-ball premise on the tracking.
It reads the submission CSV, an optional comparison CSV and the tracking (court.trk.from_atd). It
NEVER reads evaluation/gt or any ground truth, public or private, so nothing printed here is
tuned on labels. The tracking statistics (walking share, ball step size) are consistency checks
against our own labels, not accuracy.

Status: PASS / FAIL for the VERDICT gates, WARN for the heuristic sanity thresholds defined below
(they are this script's, not VERDICT's), INFO for numbers with no gate. Exit code 0 only if no FAIL.
A compact JSON of all rows is written to analysis/out/<csv name>_real_checks.json (git-ignored).

--synthetic-demo SEED is a test hook: it swaps the real tracking for the synthetic Trk
mtah.experiments.court.synth.make(SEED) (imported lazily), so the script can run without the match data.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from analysis.unlabeled_stats import valley  # noqa: E402
from mtah.v2 import touch, tracking  # noqa: E402  (the stamped cache the submission itself uses)

WALK_MPS = 2.0          # a slot walks below this speed (WALK_MPS of cand_08)
WALK_SHARE = 0.9        # a frame is "all walking" when this share of known-team slots walks
MIN_SLOTS = 4           # fewer valid known-team slots than this: the frame has no walking share
SPEED_LAG = 2           # frames either side for the position-derived speed
KICK_MPS = 5.0          # DV_KICK of the touch-event design
VALLEY_BIN = 0.05       # histogram bin width for the walking-share valley (share is in 0..1)
JITTER_WARN = 0.05      # WARN above this share of ball steps over 1 m
MISSING_WARN = 0.5      # WARN above this share of D-labelled frames without a ball
MIN_WALK_FRAC = 0.01    # FAIL when fewer frames than this reach the walking share
KICK_RUN = 13           # consecutive visible-ball frames the kick detection needs (touch K)
RUN_WARN, RUN_FAIL = 0.6, 0.3   # lead guesses for the share of visible frames in runs >= KICK_RUN
PS_SLOTS = 10           # frames with at least this many tracked slots enter the speed-unit check
PS_MEDIAN_MIN, PS_P95_MIN, PS_P95_MAX = 0.3, 2.0, 15.0   # m/s bounds (expected median 0.5-3, p95 4-9)
ONE_TEAM_WARN = 0.8     # WARN when one team holds more than this share of a half's live frames


def read_labels(path, n):
    """Letter array (n,) from a frame,label CSV; frames without a valid letter are X."""
    labels = np.full(n, "X", dtype="<U1")
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            frame = int(row["frame"])
            letter = row["label"].strip().upper()
            if 0 <= frame < n and letter in ("H", "A", "D"):
                labels[frame] = letter
    return labels


def rle(seg):
    """(letters, lengths) of the maximal runs of a letter array."""
    cut = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [len(seg)]))
    return seg[cut[:-1]], np.diff(cut)


def shape(labels, blocks, fps):
    """Possession and D statistics of a label array inside `blocks` [(half, lo, hi)]."""
    dur, switches, d_runs, d_start, x_frames, n_frames = [], 0, [], 0.0, 0, 0
    kick = ""
    for _, lo, hi in blocks:
        letters, lengths = rle(labels[lo:hi + 1])
        live = np.isin(letters, ("H", "A"))
        dur += list(lengths[live] / float(fps))
        switches += int(np.sum(live[1:] & live[:-1]))
        d_runs += list(lengths[letters == "D"] / float(fps))
        d_start = max(d_start, lengths[0] / float(fps) if letters[0] == "D" else 0.0)
        kick = kick or "".join(letters[letters != "D"][:1])
        x_frames += int(lengths[letters == "X"].sum())
        n_frames += hi - lo + 1
    dur = np.array(dur)
    d_frames = sum(int(np.sum(labels[lo:hi + 1] == "D")) for _, lo, hi in blocks)
    return {"n": len(dur), "mean_s": float(dur.mean()) if len(dur) else 0.0,
            "lt3": float(np.mean(dur < 3)) if len(dur) else 0.0,
            "max_s": float(dur.max()) if len(dur) else 0.0,
            "flicker": switches / (n_frames / float(fps) / 60.0),
            "d_share": d_frames / float(n_frames), "d_longest_s": max(d_runs, default=0.0),
            "d_start_s": d_start, "kick": kick, "x": x_frames}


def speeds(trk):
    """Per-slot speed: the tracker's ps, else a central difference of positions."""
    px, py, ps, k = trk["px"], trk["py"], trk["ps"], SPEED_LAG
    der = np.full(ps.shape, np.nan, dtype=np.float32)
    der[k:-k] = np.hypot(px[2 * k:] - px[:-2 * k], py[2 * k:] - py[:-2 * k]) * (trk["fps"] / (2.0 * k))
    return np.where(np.isnan(ps), der, ps)


def walking_share(trk):
    """Per frame share of known-team slots with speed < WALK_MPS (NaN below MIN_SLOTS slots)."""
    v = speeds(trk)[:, trk["team"] >= 0]
    ok = np.isfinite(v)
    n = ok.sum(axis=1)
    share = np.where(n >= MIN_SLOTS, (ok & (v < WALK_MPS)).sum(axis=1) / np.maximum(n, 1), np.nan)
    return share


class Table:
    """Collected check rows: (status, scope, check, value, gate)."""

    def __init__(self):
        self.rows = []

    def gate(self, scope, name, value, lo, hi, fmt="{:.3f}"):
        ok = (lo is None or value >= lo) and (hi is None or value <= hi)
        span = "{}..{}".format("" if lo is None else fmt.format(lo), "" if hi is None else fmt.format(hi))
        self.rows.append(("PASS" if ok else "FAIL", scope, name, fmt.format(value), span))

    def warn(self, scope, name, value, bad, gate, fmt="{:.3f}"):
        self.rows.append(("WARN" if bad else "PASS", scope, name, fmt.format(value), gate))

    def info(self, scope, name, value):
        self.rows.append(("INFO", scope, name, value, ""))

    def print(self):
        w = [max(len(str(r[i])) for r in self.rows) for i in (1, 2, 3)]
        for s, scope, name, value, g in self.rows:
            print("{:<5} {:<{}}  {:<{}}  {:>{}}  {}".format(
                s, scope, w[0], name, w[1], value, w[2], g).rstrip())


def acceptance(tab, scope, st, other, v1):
    """VERDICT section 5 item 4 gates for one scope; `other` is the comparison shape or None."""
    if other is not None:
        tab.gate(scope, "possessions / compare", st["n"] / float(max(other["n"], 1)), 0.7, 1.3, "{:.2f}")
        if v1:
            tab.gate(scope, "possessions - v1 count", st["n"] - other["n"], 0, None, "{:.0f}")
        tab.gate(scope, "D share - compare D share", st["d_share"] - other["d_share"], -0.1, 0.1)
    tab.info(scope, "possessions", "{}".format(st["n"]))
    tab.gate(scope, "mean possession s", st["mean_s"], 8, 25, "{:.1f}")
    tab.gate(scope, "share < 3 s", st["lt3"], None, 0.10)
    tab.info(scope, "max possession s", "{:.1f}".format(st["max_s"]))
    tab.gate(scope, "flicker H<->A /min", st["flicker"], 0.5, 3, "{:.2f}")
    tab.info(scope, "D share", "{:.3f}".format(st["d_share"]))
    tab.info(scope, "longest D run s", "{:.1f}".format(st["d_longest_s"]))
    if scope == "half 1":   # graded from the kickoff itself
        tab.gate(scope, "D run at half start s", st["d_start_s"], None, 60, "{:.1f}")
    else:
        tab.info(scope, "D run at half start s", "{:.1f}".format(st["d_start_s"]))
    if scope == "half 2":
        tab.rows.append(("INFO", scope, "head-of-half note", "", "pre-kickoff head of half 2 is expected "
                         "to be dead per FAQ.md (~117 s); this is a consistency check, never a rule: "
                         "nothing in the pipeline may hard-code it"))
    if scope != "all":
        tab.info(scope, "first live label after head D (kickoff team)", st["kick"] or "none")
    tab.gate(scope, "X frames", st["x"], 0, 0, "{:.0f}")


def tracking_checks(tab, trk, labels):
    """Dead-ball premise and ball-jitter sanity from tracking and our labels (no truth)."""
    fps, blocks = trk["fps"], trk["blocks"]
    inside = np.zeros(len(labels), dtype=bool)
    for _, lo, hi in blocks:
        inside[lo:hi + 1] = True
    share = walking_share(trk)
    s_in = share[inside]
    s_ok = s_in[np.isfinite(s_in)]
    frac = float(np.mean(s_ok >= WALK_SHARE)) if len(s_ok) else 0.0
    tab.rows.append(("FAIL" if frac < MIN_WALK_FRAC else "PASS", "tracking",
                     "frames with walking share >= {:g}".format(WALK_SHARE), "{:.3f}".format(frac),
                     ">= {:g}".format(MIN_WALK_FRAC)))
    v = valley(s_ok, VALLEY_BIN) if len(s_ok) else None
    tab.warn("tracking", "walking-share bimodal", 0.0 if v is None else 1.0, v is None, "valley found",
             "{:.0f}")
    if v is not None:
        tab.info("tracking", "valley / modes", "{:.2f} / {:.2f} {:.2f}".format(*v))
    walk = np.isfinite(share) & (share >= WALK_SHARE) & inside
    dead = (labels == "D") & np.isfinite(share)
    tab.info("tracking", "labelled D with walking share >= 0.9",
             "{:.3f}".format(float(np.mean(walk[dead]))) if dead.any() else "n/a")
    tab.info("tracking", "walking share >= 0.9 labelled D",
             "{:.3f}".format(float(np.mean(labels[walk] == "D"))) if walk.any() else "n/a")

    steps = []
    for _, lo, hi in blocks:
        x, y = trk["bx"][lo:hi + 1], trk["by"][lo:hi + 1]
        both = np.isfinite(x[1:]) & np.isfinite(x[:-1])
        steps.append(np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])[both])
    steps = np.concatenate(steps)
    if len(steps):
        tab.info("ball", "step m median / p95", "{:.3f} / {:.3f}".format(*np.percentile(steps, (50, 95))))
        tab.info("ball", "steps over {:g} m/s".format(KICK_MPS),
                 "{:.3f}".format(float(np.mean(steps > KICK_MPS / fps))))
        over = float(np.mean(steps > 1.0))
        tab.warn("ball", "steps over 1 m", over, over > JITTER_WARN, "<= {:g}".format(JITTER_WARN))
    d_frames = (labels == "D") & inside
    if d_frames.any():
        miss = float(np.mean(~np.isfinite(trk["bx"][d_frames])))
        tab.warn("ball", "ball missing in D frames", miss, miss > MISSING_WARN, "<= {:g}".format(MISSING_WARN))
    roughness = ball_roughness_stats(trk["bx"], trk["by"], blocks)
    tab.info("ball", "roughness x m median / p99", "{:.3f} / {:.3f}".format(*roughness["x"]))
    tab.info("ball", "roughness y m median / p99", "{:.3f} / {:.3f}".format(*roughness["y"]))


def visibility_checks(tab, trk):
    """Ball visibility per graded half: run lengths, share of visible frames in runs >= KICK_RUN."""
    for half, lo, hi in trk["blocks"]:
        scope = "half {}".format(half)
        vis = np.isfinite(trk["bx"][lo:hi + 1])
        letters, lengths = rle(np.where(vis, "V", "M"))
        on, off = lengths[letters == "V"], lengths[letters == "M"]
        if not len(on):
            tab.rows.append(("FAIL", scope, "ball visible frames", "0", "> 0"))
            continue
        tab.info(scope, "visible run frames p5/25/50/75/95", " / ".join(
            "{:.0f}".format(v) for v in np.percentile(on, (5, 25, 50, 75, 95))))
        share = float(on[on >= KICK_RUN].sum()) / on.sum()
        tab.rows.append(("FAIL" if share < RUN_FAIL else "WARN" if share < RUN_WARN else "PASS", scope,
                         "visible frames in runs >= {}".format(KICK_RUN), "{:.3f}".format(share),
                         "warn < {:g}, fail < {:g} (lead guesses)".format(RUN_WARN, RUN_FAIL)))
        tab.info(scope, "longest ball-missing run s",
                 "{:.1f}".format(off.max() / float(trk["fps"]) if len(off) else 0.0))


def speed_unit_check(tab, trk):
    """Median and p95 of the tracker's own ps on frames with enough tracked slots (unit sanity)."""
    ps = trk["ps"][np.isfinite(trk["px"]).sum(axis=1) >= PS_SLOTS]
    ps = ps[np.isfinite(ps)]
    if not len(ps):
        tab.info("tracking", "ps speed unit", "ps all NaN, skipped")
        return
    med, p95 = np.percentile(ps, (50, 95))
    ok = med >= PS_MEDIAN_MIN and PS_P95_MIN <= p95 <= PS_P95_MAX
    tab.rows.append(("PASS" if ok else "FAIL", "tracking", "ps median / p95 m/s",
                     "{:.2f} / {:.2f}".format(med, p95),
                     "median >= {:g}, p95 {:g}..{:g}".format(PS_MEDIAN_MIN, PS_P95_MIN, PS_P95_MAX)))


def team_balance(tab, labels, blocks, fps):
    """Longest single-team run and the larger team's share of live frames, per half."""
    for half, lo, hi in blocks:
        letters, lengths = rle(labels[lo:hi + 1])
        h, a = (int(lengths[letters == t].sum()) for t in "HA")
        scope = "half {}".format(half)
        longest = max([0.0] + [l / float(fps) for t, l in zip(letters, lengths) if t in "HA"])
        tab.info(scope, "longest single-team run s", "{:.1f}".format(longest))
        top = max(h, a) / float(max(h + a, 1))
        tab.warn(scope, "larger team share of live frames", top, top > ONE_TEAM_WARN,
                 "<= {:g}".format(ONE_TEAM_WARN))


def second_differences_1d(z, blocks=None):
    """Absolute second differences |z[t+2] - 2*z[t+1] + z[t]| from consecutive-visible triples."""
    if blocks is None:
        segments = [z]
    else:
        segments = [z[lo:hi + 1] for _, lo, hi in blocks]
    diffs = []
    for seg in segments:
        if len(seg) < 3:
            continue
        d2 = seg[2:] - 2.0 * seg[1:-1] + seg[:-2]
        ok = np.isfinite(d2)
        if ok.any():
            diffs.append(np.abs(d2[ok]))
    return np.concatenate(diffs) if diffs else np.empty(0, dtype=np.float64)


def ball_roughness_stats(bx, by, blocks=None):
    """Median and p99 of absolute second differences for bx and by in metres."""
    out = {}
    for axis, arr in (("x", bx), ("y", by)):
        d2 = second_differences_1d(arr, blocks)
        if len(d2):
            out[axis] = (float(np.median(d2)), float(np.percentile(d2, 99)))
        else:
            out[axis] = (0.0, 0.0)
    return out


def kick_enrichment(trk, thresholds=(3.0, 5.0, 8.0)):
    """Kick velocity-change event rates and player-proximity shares inside blocks.

    Reads event definitions and constants from mtah.v2.touch.
    """
    fps = trk["fps"]
    bx, by = trk["bx"].astype(np.float64), trk["by"].astype(np.float64)
    vis = np.isfinite(bx) & np.isfinite(by)
    with np.errstate(invalid="ignore"):
        jump = np.hypot(np.diff(bx), np.diff(by)) * fps > touch.V_MAX
    good = vis.copy()
    good[1:] &= ~jump
    good[:-1] &= ~jump
    within_k = touch._all_within(good, touch.K)

    blocks = trk.get("blocks") or [(1, 0, len(bx) - 1)]
    inside = np.zeros(len(bx), dtype=bool)
    for _, lo, hi in blocks:
        inside[lo:hi + 1] = True

    usable = within_k & inside
    usable_minutes = float(usable.sum()) / (float(fps) * 60.0)

    vx_b, vx_f = touch._slope(bx, good)
    vy_b, vy_f = touch._slope(by, good)
    dv = np.where(usable, np.hypot(vx_f - vx_b, vy_f - vy_b) * fps, 0.0)
    prev = np.concatenate([[0.0], dv[:-1]])
    nxt = np.concatenate([dv[1:], [0.0]])

    d_h, d_a = (np.where(vis, d, np.inf) for d in touch._team_dist(trk))
    near = np.minimum(d_h, d_a)
    player_near = near < touch.R_KICK
    baseline_share = float(np.mean(player_near[usable])) if usable.any() else 0.0

    rows = []
    for thresh in thresholds:
        ev = (dv > thresh) & (dv >= prev) & (dv > nxt) & inside
        count = int(ev.sum())
        per_min = count / usable_minutes if usable_minutes > 0 else 0.0
        near_share = float(np.mean(player_near[ev])) if count > 0 else 0.0
        rows.append({
            "thresh": thresh,
            "count": count,
            "per_min": per_min,
            "near_share": near_share,
        })
    return {
        "usable_minutes": usable_minutes,
        "baseline_share": baseline_share,
        "rows": rows,
    }


def kick_checks(tab, trk):
    """Kick enrichment table: rate and player proximity across DV_KICK thresholds."""
    res = kick_enrichment(trk)
    base = res["baseline_share"]
    tab.info("kick", "player within 2 m (all usable frames)", "{:.3f}".format(base))
    for row in res["rows"]:
        tab.info(
            "kick",
            "DV_KICK {:g} m/s (/min, near share vs all)".format(row["thresh"]),
            "{:.2f} /min, {:.3f} vs {:.3f}".format(row["per_min"], row["near_share"], base),
        )


def team_tag_to_side(team_str):
    """Map a Smart Tagging Team string to 'H' or 'A'."""
    if not team_str:
        return None
    s = str(team_str).strip().lower()
    if "barcelona" in s or s in ("h", "home"):
        return "H"
    if "atletic" in s or "athletic" in s or s in ("a", "away"):
        return "A"
    return None


def check_half_kickoff_alternation(labels, events):
    """First live label (H or A) after the kickoff of each half; returns (letters, alternates)."""
    if not events:
        return [], False
    start_tags = [
        e for e in events
        if (e.get("code") or "").strip().lower() == "set pieces"
        and (e.get("tags", {}).get("Type") or "").strip().lower() == "kick off (start)"
    ]
    start_tags.sort(key=lambda e: e.get("start_frame", 0))
    letters = []
    for tag in start_tags:
        sf = tag.get("start_frame", 0)
        sub = labels[sf:]
        live = sub[np.isin(sub, ("H", "A"))]
        letters.append(str(live[0]) if len(live) else "none")
    alternates = (
        len(letters) == 2
        and letters[0] in ("H", "A")
        and letters[1] in ("H", "A")
        and letters[0] != letters[1]
    )
    return letters, alternates


def check_goal_kickoffs(labels, events):
    """For each GOALS tag, verify the first live label after following kickoff is opposite team."""
    if not events:
        return 0, 0
    goals = [e for e in events if (e.get("code") or "").strip().lower() == "goals"]
    kickoffs = [
        e for e in events
        if (e.get("code") or "").strip().lower() == "set pieces"
        and "kick off" in (e.get("tags", {}).get("Type") or "").strip().lower()
    ]
    holds = 0
    total = len(goals)
    for g in goals:
        scorer = team_tag_to_side(g.get("tags", {}).get("Team"))
        if scorer is None:
            continue
        expected = "A" if scorer == "H" else "H"
        after_kos = [k for k in kickoffs if k.get("start_frame", 0) >= g.get("start_frame", 0)]
        if not after_kos:
            continue
        ko = after_kos[0]
        sub = labels[ko.get("start_frame", 0):]
        live = sub[np.isin(sub, ("H", "A"))]
        first_live = str(live[0]) if len(live) else None
        if first_live == expected:
            holds += 1
    return holds, total


def kickoff_checks(tab, labels, events):
    """Kickoff-side checks from Smart Tagging and the submission CSV."""
    if not events:
        tab.info("kickoff", "half 1/2 first live label alternates", "skipped (no Smart Tagging)")
        tab.info("kickoff", "post-goal kickoff side opposite to scorer", "skipped (no Smart Tagging)")
        return
    letters, alternates = check_half_kickoff_alternation(labels, events)
    if len(letters) == 2:
        val = "{}, {} ({})".format(letters[0], letters[1], "alternates" if alternates else "no alternation")
    else:
        val = " / ".join(letters) if letters else "none"
    tab.info("kickoff", "half 1/2 first live label alternates", val)

    holds, total = check_goal_kickoffs(labels, events)
    tab.info("kickoff", "post-goal kickoff side opposite to scorer", "{}/{} holds".format(holds, total))


def load_smart_tagging_events():
    """Load Smart Tagging events if available, else return None."""
    try:
        import load_smart_tagging as st
        return st.load()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Unlabeled real-data acceptance gate for a possession CSV.")
    ap.add_argument("--csv", required=True, help="submission frame,label CSV to check")
    ap.add_argument("--compare", help="comparison frame,label CSV (e.g. 02 alone or the v1 file)")
    ap.add_argument("--compare-is-v1", action="store_true",
                    help="the comparison is the v1 file: possessions must not fall below its count")
    ap.add_argument("--synthetic-demo", type=int, metavar="SEED", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.compare_is_v1 and not args.compare:
        ap.error("--compare-is-v1 needs --compare")

    if args.synthetic_demo is not None:
        from mtah.experiments.court import synth  # noqa: E402  (test hook only)
        trk = synth.make(args.synthetic_demo)[0]
    else:
        try:
            trk = tracking.from_atd()
        except OSError as err:
            sys.exit("real_checks: match tracking not found ({}). Unpack the bundle into data/ or set "
                     "MTAH_DATA_DIR (see README.md, 'Reproduce').".format(" ".join(str(err).split())))
    n, fps, blocks = len(trk["bx"]), trk["fps"], trk["blocks"]
    try:
        labels = read_labels(args.csv, n)
        other = read_labels(args.compare, n) if args.compare else None
    except (OSError, KeyError, ValueError) as err:
        sys.exit("real_checks: cannot read a label CSV: {}".format(err))

    tab = Table()
    scopes = [("half {}".format(h), [(h, lo, hi)]) for h, lo, hi in blocks] + [("all", blocks)]
    for scope, bl in scopes:
        acceptance(tab, scope, shape(labels, bl, fps), None if other is None else shape(other, bl, fps),
                   args.compare_is_v1)
    tracking_checks(tab, trk, labels)
    visibility_checks(tab, trk)
    speed_unit_check(tab, trk)
    team_balance(tab, labels, blocks, fps)
    kick_checks(tab, trk)
    st_events = load_smart_tagging_events() if args.synthetic_demo is None else None
    kickoff_checks(tab, labels, st_events)

    tab.print()
    out = os.path.join(ROOT, "analysis", "out")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, os.path.splitext(os.path.basename(args.csv))[0] + "_real_checks.json")
    with open(path, "w") as fh:
        json.dump([dict(zip(("status", "scope", "check", "value", "gate"), r)) for r in tab.rows], fh, indent=1)
    failed = sum(r[0] == "FAIL" for r in tab.rows)
    print("\n{} FAIL, {} WARN; wrote {}".format(failed, sum(r[0] == "WARN" for r in tab.rows),
                                               os.path.relpath(path, ROOT)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
