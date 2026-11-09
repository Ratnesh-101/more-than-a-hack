"""Data-arrival audit: does the bundle, and OUR parse of it, match what DATA.md publishes?

    python -m mtah.audit                 # after unpacking the bundle into data/
    python -m mtah.audit --data-dir DIR

Run this before trusting any score. Every expected number below is copied from DATA.md / FAQ.md
(published facts about the provided files), never from the ground truth. A FAIL means our reading
of the data is wrong somewhere -- frame offset, coordinate scaling, team mapping -- and every label
downstream inherits it. Exit code 1 on any FAIL.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames  # noqa: E402
from mtah import features  # noqa: E402

# Published in DATA.md (tactical source). Exact counts are checked exactly; means within a band.
EXPECT = {
    "atd_records": 199319,
    "slots": 46,
    "home_name": "barcelona",          # DATA.md: "Home is Barcelona (H), away is Athletic (A)"
    "ball_frames_all": 113488,         # 56.9 %
    "ball_frames_graded": 94214,       # 59.3 % of 158,857
    "graded_frames": 158857,
    "mean_tracked_graded": 20.5,       # "the mean over graded frames is 20.5"
    "smart_tagging_instances": 638,
    "team_tags": {"Barcelona": 314, "Atletic Club": 288, "N/A": 36},
    "homography_valid_pct": 100.0,
}


class Report:
    def __init__(self):
        self.rows, self.fails, self.warns = [], 0, 0

    def check(self, name, ok, got, want="", warn_only=False):
        status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
        self.fails += status == "FAIL"
        self.warns += status == "WARN"
        print("  {:<4}  {:<34} got {:<22} {}".format(
            status, name, str(got), "want {}".format(want) if want != "" else ""))


def graded_mask(n):
    m = np.zeros(n, dtype=bool)
    for _, lo, hi in half_video_blocks(GAME):
        m[lo:hi + 1] = True
    return m


def audit_atd(r, data_dir, refresh):
    from baseline import team_names, team_of_each_track  # official parsers
    paths = asset_paths(GAME, data_dir)
    print("\nATD metadata")
    if not os.path.isfile(paths["atd_meta"]):
        r.check("atd_meta present", False, "missing", paths["atd_meta"])
        return None
    team = team_of_each_track(paths["atd_meta"])
    home, away = team_names(paths["atd_meta"])
    r.check("track slots", len(team) == EXPECT["slots"], len(team), EXPECT["slots"])
    r.check("home team (H) is Barcelona", EXPECT["home_name"] in (home or "").lower(),
            "H={} A={}".format(home, away), "H=Barcelona")
    r.check("slots per team (home/away)", True,
            "{}/{}".format(int((team == 0).sum()), int((team == 1).sum())))

    print("\nATD tracking -> features (cached at data/cache/)")
    if not os.path.isfile(paths["atd_raw"]):
        r.check("atd_raw present", False, "missing", paths["atd_raw"])
        return None
    f = features.load(GAME, data_dir, refresh=refresh)
    n = source_frames("tactical", GAME)
    g = graded_mask(n)
    records = int(np.isfinite(f["n_home"]).sum())
    vis = f["ball_vis"] > 0.5
    r.check("ATD records parsed", records == EXPECT["atd_records"], records,
            EXPECT["atd_records"])
    r.check("graded frames", int(g.sum()) == EXPECT["graded_frames"], int(g.sum()),
            EXPECT["graded_frames"])
    r.check("ball visible, all frames", int(vis.sum()) == EXPECT["ball_frames_all"],
            "{} ({:.1f}%)".format(int(vis.sum()), 100 * vis.mean()), EXPECT["ball_frames_all"])
    r.check("ball visible, graded frames", int(vis[g].sum()) == EXPECT["ball_frames_graded"],
            "{} ({:.1f}%)".format(int(vis[g].sum()), 100 * vis[g].mean()),
            EXPECT["ball_frames_graded"])
    tracked = np.nan_to_num(f["n_home"][g]) + np.nan_to_num(f["n_away"][g])
    mt = float(tracked.mean())
    r.check("mean tracked objects, graded", abs(mt - EXPECT["mean_tracked_graded"]) < 0.3,
            "{:.2f}".format(mt), "~{}".format(EXPECT["mean_tracked_graded"]))

    # scale sanity: metres, inside a slightly padded pitch almost always
    length, width = GAMES[GAME]["pitch"]
    bx, by = f["ball_x"][vis], f["ball_y"][vis]
    inside = np.mean((bx > -5) & (bx < length + 5) & (by > -5) & (by < width + 5))
    r.check("ball within pitch +5 m", inside > 0.98, "{:.1%}".format(inside), "> 98%")
    out = float(np.nanmean(f["ball_out"][g & vis]))
    r.check("ball_out share (seen, graded)", out < 0.2, "{:.1%}".format(out), "< 20%",
            warn_only=True)
    dmin = np.fmin(f["d_home"], f["d_away"])[g & vis]
    med = float(np.nanmedian(dmin))
    r.check("median ball->nearest player (m)", 0.2 < med < 5.0, "{:.2f}".format(med),
            "0.2 .. 5 (metres, not 0-1 units)")
    ms = float(np.nanmedian(f["mean_speed"][g]))
    r.check("median mean player speed (m/s)", 0.5 < ms < 5.0, "{:.2f}".format(ms),
            "0.5 .. 5", warn_only=True)
    # live play should be inside the graded halves; the tracking runs only there
    outside = int(np.isfinite(f["n_home"][~g]).sum())
    r.check("ATD records outside graded halves", True, outside)

    if "pos" in f:
        r.check("player positions cached", True, "{} {}".format(f["pos"].shape, f["pos"].dtype))
    root = data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
    cache = os.path.join(root, "cache", "{}_features.npz".format(GAME))
    if os.path.isfile(cache):
        mb = os.path.getsize(cache) / 1e6
        r.check("feature cache size (MB)", mb < 300, "{:.1f}".format(mb), "< 300")
    return f


def audit_v2_tracking(r, data_dir):
    """The SUBMISSION parser (mtah.v2.tracking) is separate from features.py: check it too.

    The v2 dead-ball model's walking-share premise assumes the tracker's speed column is m/s. The
    unit check here does not trust that: it compares the tracker speed with the speed derived from
    the positions themselves (which are scaled to metres from the 0-1 coordinates). Ratio ~1 means
    m/s; ~3.6 means km/h; ~0.01 means normalised units -- any of those breaks WALK_MPS = 2.
    """
    print("\nv2 tracking (the parser the submission uses)")
    try:
        from mtah.v2 import tracking
        trk = tracking.from_atd(data_dir=data_dir)
    except (IOError, OSError) as err:
        r.check("v2 tracking loads", False, type(err).__name__)
        return
    n = len(trk["bx"])
    g = graded_mask(n)
    vis = np.isfinite(trk["bx"])
    r.check("v2 ball visible, all frames", int(vis.sum()) == EXPECT["ball_frames_all"],
            int(vis.sum()), EXPECT["ball_frames_all"])
    r.check("v2 ball visible, graded", int(vis[g].sum()) == EXPECT["ball_frames_graded"],
            int(vis[g].sum()), EXPECT["ball_frames_graded"])
    real = int((trk["team"] >= 0).sum())
    r.check("v2 slots with a team", real == EXPECT["slots"], real, EXPECT["slots"])

    known = trk["team"] >= 0
    px, py, ps = trk["px"][:, known], trk["py"][:, known], trk["ps"][:, known]
    present = np.isfinite(px[g])
    with_speed = float(np.isfinite(ps[g])[present].mean()) if present.any() else 0.0
    r.check("tracker speed present (of tracked)", with_speed > 0.5, "{:.1%}".format(with_speed),
            "> 50% (else dead ball runs on noisy derived speed)", warn_only=True)

    lag, fps = 2, trk["fps"]
    der = np.full(ps.shape, np.nan, dtype=np.float32)
    der[lag:-lag] = np.hypot(px[2 * lag:] - px[:-2 * lag],
                             py[2 * lag:] - py[:-2 * lag]) * (fps / (2.0 * lag))
    both = g[:, None] & np.isfinite(ps) & np.isfinite(der)
    if both.sum() < 1000:
        r.check("speed unit check (tracker vs positions)", False, "too few samples", "",
                warn_only=True)
        return
    med_ps, med_der = float(np.median(ps[both])), float(np.median(der[both]))
    ratio = med_ps / med_der if med_der > 0 else float("inf")
    r.check("median tracker speed (m/s)", 0.3 < med_ps < 4.0, "{:.2f}".format(med_ps),
            "0.3 .. 4 (football, m/s)")
    r.check("tracker / position-derived speed", 0.7 < ratio < 1.4, "{:.2f}".format(ratio),
            "~1.0 (3.6 = km/h)")
    p99 = float(np.percentile(ps[both], 99))
    r.check("p99 tracker speed (m/s)", p99 < 12.0, "{:.1f}".format(p99),
            "< 12 (human sprint ~10.5)", warn_only=True)


def audit_smart_tagging(r, data_dir):
    print("\nSmart Tagging")
    import load_smart_tagging as st   # official loader
    try:
        events = st.load(GAME, data_dir)
    except IOError:
        r.check("smart_tagging present", False, "missing")
        return
    r.check("instances", len(events) == EXPECT["smart_tagging_instances"], len(events),
            EXPECT["smart_tagging_instances"])
    teams = {}
    for e in events:
        t = e["tags"].get("Team")
        teams[t] = teams.get(t, 0) + 1
    r.check("Team tag counts", teams == EXPECT["team_tags"], teams, EXPECT["team_tags"],
            warn_only=True)
    tb = st.check_time_base(events)
    r.check("time base (kickoffs contained)", tb["agrees"],
            [row["contains_kickoff"] for row in tb["halves"]], "[True, True]")


def audit_homography(r, data_dir):
    print("\nHomography")
    import load_homography as hg   # official loader
    try:
        rep = hg.check_alignment(GAME, data_dir)
        _, _, valid = hg.load(GAME, data_dir)
    except IOError:
        r.check("homography present", False, "missing")
        return
    r.check("one row per video frame", rep["agrees"], rep["n_rows"], rep["n_video_frames"])
    pct = 100.0 * valid.mean()
    r.check("valid %", abs(pct - EXPECT["homography_valid_pct"]) < 0.05, "{:.2f}".format(pct),
            EXPECT["homography_valid_pct"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--refresh", action="store_true", help="rebuild the feature cache first")
    args = ap.parse_args()

    print("audit of", os.path.join(args.data_dir or os.environ.get("MTAH_DATA_DIR")
                                   or os.path.join(ROOT, "data"), "tactical"))
    r = Report()
    audit_atd(r, args.data_dir, args.refresh)
    audit_v2_tracking(r, args.data_dir)
    audit_smart_tagging(r, args.data_dir)
    audit_homography(r, args.data_dir)
    print("\nAUDIT {}  ({} fail, {} warn)".format("OK" if not r.fails else "FAILED",
                                                  r.fails, r.warns))
    sys.exit(1 if r.fails else 0)


if __name__ == "__main__":
    main()
