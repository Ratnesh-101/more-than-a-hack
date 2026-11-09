"""Synthetic ATD world shared by mtah.selfcheck and mtah.v2.tests.test_robust (SYNTHETIC only).

Smooth world: 22 players random-walk at plausible speeds (live 2-6 m/s, stoppage 0-1.2 m/s), the
ball is passed between players of the possessing team (hold 1.6-4.4 s, ~0.8 s linear flight) and
sits still on a player during a stoppage; ~35% of ball frames are lost in occlusion RUNS (mean ~40
frames). Spells last 6-24 s, 25% of them are stoppages, live spells alternate H, A, H, ...

make_sim(seed)                         the 12,500-frame mini world used by test_robust
make_sim(seed, n, blocks=real blocks)  only the graded frames are simulated (compact), with the
                                       simulator's own truth in sim["n_true"] / sim["dead_share"]
write_bundle(folder, sim)              writes <folder>/tactical/BAR-ATH_tactical_atd.{xml,txt}
"""

import os

import numpy as np

N = 12500
BLOCKS = [(1, 300, 5299), (2, 7000, 11999)]
S = 46
FPS = 25
L, W = 105.3, 68.0
HOME_SLOTS = list(range(0, 11))
AWAY_SLOTS = list(range(23, 34))
NAN3 = "NaN,NaN,NaN"


def make_sim(seed, n=N, blocks=None):
    """Return the sim dict. With blocks=None every frame 0..n-1 is simulated (one segment).
    With blocks=[(half, lo, hi)] only those frames are simulated (arrays are compact, sim["frames"]
    maps rows to video frames); each half starts its own spell sequence (half 1 kicked off by H,
    half 2 by A) and sim["n_true"][half] counts its live spells = true possessions."""
    rng = np.random.default_rng(seed)
    if blocks is None:
        frames = np.arange(n)
        segs = [(0, n, 0, None)]
    else:
        frames = np.concatenate([np.arange(lo, hi + 1) for _, lo, hi in blocks])
        segs, a = [], 0
        for half, lo, hi in blocks:
            segs.append((a, a + hi - lo + 1, half - 1, half))
            a += hi - lo + 1
    M = len(frames)
    X = np.full((M, S), np.nan)
    Y = np.full((M, S), np.nan)
    SP = np.full((M, S), np.nan)
    BX, BY = np.full(M, np.nan), np.full(M, np.nan)
    state = np.zeros(M, int)       # 0 H, 1 A, 2 dead
    spells = []
    n_true, dead_share = {}, {}
    for a, b, cur, half in segs:
        t = a
        while t < b:
            dur = int(rng.integers(150, 600))
            dead = rng.random() < 0.25
            st = 2 if dead else cur
            e = min(b, t + dur)
            state[t:e] = st
            spells.append((t, e, st))
            t += dur
            if not dead:
                cur = 1 - cur
        if half is not None:
            n_true[half] = sum(1 for s0, s1, st in spells if a <= s0 < b and st != 2)
            dead_share[half] = float((state[a:b] == 2).mean())
    slots = HOME_SLOTS + AWAY_SLOTS
    m = len(slots)
    u = rng.uniform(2.0, 6.0, (M, m))
    d = rng.uniform(0.0, 1.2, (M, m))
    spd = np.where((state == 2)[:, None], d, u)                       # m/s
    theta = np.cumsum(rng.normal(0, 0.08, (M, m)), axis=0) + rng.uniform(0, 6.28, m)
    vx = spd * np.cos(theta) / FPS / L
    vy = spd * np.sin(theta) / FPS / W
    px = np.cumsum(vx, axis=0) + rng.uniform(0.2, 0.8, m)
    py = np.cumsum(vy, axis=0) + rng.uniform(0.2, 0.8, m)
    px, py = (1.0 - np.abs(np.mod(p, 2.0) - 1.0) for p in (px, py))   # reflect into [0, 1]
    for j, s in enumerate(slots):
        X[:, s], Y[:, s], SP[:, s] = px[:, j], py[:, j], spd[:, j]
    last = (px[0, 0], py[0, 0])
    for a, b, st in spells:
        if st == 2:
            c = int(rng.integers(0, m))
            BX[a:b], BY[a:b] = px[a, c], py[a, c]
            last = (px[b - 1, c], py[b - 1, c])
            continue
        t = a
        while t < b:     # hold, then a pass: ~0.8 s linear flight to the next carrier of the team
            c = int(rng.integers(0, 11)) + (0 if st == 0 else 11)
            fl = min(20, b - t)
            w = (np.arange(fl) + 1.0) / 20.0
            BX[t:t + fl] = last[0] + w * (px[t + fl - 1, c] - last[0])
            BY[t:t + fl] = last[1] + w * (py[t + fl - 1, c] - last[1])
            t += fl
            hold = min(int(rng.integers(40, 110)), b - t)
            BX[t:t + hold], BY[t:t + hold] = px[t:t + hold, c], py[t:t + hold, c]
            if hold:
                last = (px[t + hold - 1, c], py[t + hold - 1, c])
            t += hold
        BX[a:b] += rng.normal(0, 0.15 / L, b - a)
        BY[a:b] += rng.normal(0, 0.15 / W, b - a)
    miss = np.zeros(M, bool)     # ball lost in runs (mean ~40 frames, ~35% of frames), as occlusions are
    t = 0
    while t < M:
        t += int(rng.geometric(1 / 70.0))
        ln = int(rng.geometric(1 / 40.0))
        miss[t:t + ln] = True
        t += ln
    BX[miss] = np.nan
    BY[np.isnan(BX)] = np.nan
    return {"X": X, "Y": Y, "S": SP, "BX": BX, "BY": BY, "present": np.ones(M, bool),
            "frames": frames, "n_true": n_true, "dead_share": dead_share,
            "teams": ["T1"] * 23 + ["T2"] * 23, "n_groups": S, "line_hook": None}


def copy_sim(sim):
    return {k: (v.copy() if isinstance(v, np.ndarray) else (list(v) if isinstance(v, list) else v))
            for k, v in sim.items()}


def write_bundle(folder, sim):
    tac = os.path.join(folder, "tactical")
    os.makedirs(tac)
    players = "".join('<Player id="P{0}"{1}><ShirtNumber>{0}</ShirtNumber></Player>'.format(
        i + 1, "" if tm is None else ' teamId="{}"'.format(tm)) for i, tm in enumerate(sim["teams"]))
    meta = ('<?xml version="1.0"?>\n<Root><Teams><Team id="T1"><Name>Home</Name></Team>'
            '<Team id="T2"><Name>Away</Name></Team></Teams><Players>' + players + '</Players></Root>')
    with open(os.path.join(tac, "BAR-ATH_tactical_atd.xml"), "w") as fh:
        fh.write(meta)
    cols = np.flatnonzero(np.isfinite(sim["X"]).any(axis=0) & np.isfinite(sim["Y"]).any(axis=0))
    cols = [int(c) for c in cols if c < sim["n_groups"]]
    X, Y, SP, BX, BY = (sim[k].tolist() for k in ("X", "Y", "S", "BX", "BY"))
    frames = sim["frames"].tolist()
    ng = sim["n_groups"]
    lines = []
    for i in range(len(frames)):
        if not sim["present"][i]:
            continue
        xs, ys, ss = X[i], Y[i], SP[i]
        grp = [NAN3] * ng
        for s in cols:
            if xs[s] == xs[s] and ys[s] == ys[s]:
                grp[s] = "{:.4f},{:.4f},{}".format(xs[s], ys[s], "NaN" if ss[s] != ss[s]
                                                   else "{:.4f}".format(ss[s]))
        ball = NAN3 if BX[i] != BX[i] or BY[i] != BY[i] else "{:.4f},{:.4f},0".format(BX[i], BY[i])
        lines.append("{}:{}:{}".format(frames[i] + 1, ";".join(grp), ball))
    if sim["line_hook"]:
        lines = sim["line_hook"](lines)
    with open(os.path.join(tac, "BAR-ATH_tactical_atd.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
