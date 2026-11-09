"""Fallbacks of the champion, one flag away (`python -m mtah.v2.run --variant ...`).

Real tracking may kill the champion's premise (the walking share of the players reaching 0.9 in a
stoppage, see VERDICT section 4). Two self-contained ports of the court's next-best entries:

  touch_only   candidate 02 alone: its touch-event team evidence AND its own dead model (the v1
               dead cues plus STILL_BONUS / KICK_LIVE), decoded by the v1 Viterbi with 02's costs.
  semi_markov  stack_1: the same emissions decoded by candidate 12's semi-Markov segment DP, with
               02's costs (the runner-up of the court).

The touch logic shared with the champion (constants, helpers) comes from touch.py, which is
bit-identical to cand_02's; everything else is ported verbatim with its original constants.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mtah.v2 import touch
from mtah.v2.decode import decode_half
from mtah.v2.touch import CLIP, DV_KICK, HOLD, K, MARGIN, R_CTRL, R_KICK, TAU, V_MAX

VARIANTS = ("touch_only", "semi_markov")
LETTERS = np.array(["H", "A", "D"])

# switch_cost 30: after a contested touch the new team needs ~1.5 s of uninterrupted touching
# (lead grows 1 nat/s) before possession flips; this penalizes fleeting deflections, though
# loose-ball evidence can still exceed the nominal 60-nat round trip.
# dead_cost 20 > switch_cost / 2 so an H->A hand-over is never routed through a fake stoppage.
SWITCH_COST = 30.0
DEAD_COST = 20.0
MIN_DEAD_FRAMES = 25   # v1: a real stoppage (out, foul, restart) lasts >= ~1 s at 25 fps

STILL_R = 1.0      # m: a ball that moves < 1 m in 1 s is resting (on the spot), not in play
STILL_BONUS = 3.0  # log-odds: a resting ball is strong (not certain: a keeper may hold it) dead evidence
KICK_LIVE = 3.0    # log-odds: a fresh kick near a player is strong evidence the ball is in play
LIVE_S = 2.0       # s: a kick keeps play "proven live" this long (one pass flight)

# v1 dead model (mtah.decode.PARAMS, via court/incumbent), unchanged: nothing re-tuned.
SLOW_SPEED = 1.5      # mean player speed (m/s) below which play looks stopped
SLOW_WIDTH = 0.35
DEAD_BIAS = -2.5      # prior log-odds of dead ball on a normal frame
DEAD_OUT_BONUS = 5.0  # ball outside the lines -> strongly dead
DEAD_SLOW_BONUS = 3.0  # everyone slow -> likely stoppage
OUT_MARGIN_M = 0.5    # ball-tracking error near a line
EPS = 1e-6

# Candidate 12 (semi-Markov): duration priors in seconds, lognormal(median, sigma) in [min, max].
CELL = 5                 # DP grid = 0.2 s at 25 fps; boundaries are refined to the frame after
LIVE_MEDIAN_S = 10.0     # possessions in elite football are typically 5-25 s; geometric middle ~10 s
LIVE_SIGMA = 0.8         # ln(25/10) ~ 0.9 and ln(10/5) ~ 0.7: the 5-25 s band is about +/- 1 sigma
LIVE_MIN_S = 2.0         # a run shorter than a few seconds is a touch/deflection (keeps previous team)
LIVE_MAX_S = 240.0       # a 4-min unbroken possession does not occur in a match; bounds the DP
DEAD_MEDIAN_S = 12.0     # throw-ins ~8 s, goal kicks / free kicks ~20-30 s; geometric middle ~12 s
DEAD_SIGMA = 1.0         # stoppages range from a quick throw-in to a goal celebration (5 s-90 s)
DEAD_MIN_S = 1.0         # a restart takes >= ~1 s
DEAD_MAX_S = 600.0       # treatment / VAR stoppages can run minutes; 10 min bounds the DP
DUR_WEIGHT = CELL        # emissions are near-duplicates within ~0.2 s: one prior look per cell


# -- candidate 02 emissions ---------------------------------------------------------------------

def _spread(z, k):
    """(n, 2k+1) offsets of z over t-k..t+k from z[t] (NaN outside / where missing)."""
    pad = np.concatenate([np.full(k, np.nan), z, np.full(k, np.nan)])
    return sliding_window_view(pad, 2 * k + 1) - z[:, None]


def _touches(trk):
    """Frame masks: touch_home, touch_away, kick (any team in reach), ball resting."""
    fps = trk["fps"]
    bx, by = trk["bx"].astype(np.float64), trk["by"].astype(np.float64)
    vis = np.isfinite(bx) & np.isfinite(by)
    with np.errstate(invalid="ignore"):
        jump = np.hypot(np.diff(bx), np.diff(by)) * fps > V_MAX
    good = vis.copy()
    good[1:] &= ~jump
    good[:-1] &= ~jump
    vx_b, vx_f = touch._slope(bx, good)
    vy_b, vy_f = touch._slope(by, good)
    dv = np.where(touch._all_within(good, K), np.hypot(vx_f - vx_b, vy_f - vy_b) * fps, 0.0)
    prev = np.concatenate([[0.0], dv[:-1]])
    nxt = np.concatenate([dv[1:], [0.0]])
    kick = (dv > DV_KICK) & (dv >= prev) & (dv > nxt)   # one event per velocity change

    d_h, d_a = (np.where(vis, d, np.inf) for d in touch._team_dist(trk))
    near = np.minimum(d_h, d_a)
    reach_h = (d_h < R_KICK) & (d_h <= near + MARGIN)
    reach_a = (d_a < R_KICK) & (d_a <= near + MARGIN)
    held_h = touch._all_within((d_h < R_CTRL) & (d_h < d_a - MARGIN), HOLD)
    held_a = touch._all_within((d_a < R_CTRL) & (d_a < d_h - MARGIN), HOLD)
    t_h, t_a = touch._backdate((kick & reach_h) | held_h, (kick & reach_a) | held_a, vis)

    with np.errstate(invalid="ignore"):
        moved = np.hypot(_spread(bx, HOLD), _spread(by, HOLD)).max(axis=1)  # NaN if any gap
    return t_h, t_a, kick & (reach_h | reach_a), vis & (moved < STILL_R)


def _v1_log_dead(trk):
    """log P(dead) of the v1 model: ball beyond the lines + everybody slow, clipped to [EPS, 1-EPS]."""
    bx, by = trk["bx"], trk["by"]
    length, width = trk["pitch"]
    vis = ~np.isnan(bx)
    present = ~np.isnan(trk["ps"])
    mean_speed = np.where(present.any(axis=1),
                          np.where(present, trk["ps"], 0.0).sum(axis=1) / np.maximum(present.sum(axis=1), 1),
                          SLOW_SPEED)
    slow = 1.0 / (1.0 + np.exp(-(SLOW_SPEED - mean_speed) / SLOW_WIDTH))
    out = vis & ((bx < -OUT_MARGIN_M) | (bx > length + OUT_MARGIN_M)
                 | (by < -OUT_MARGIN_M) | (by > width + OUT_MARGIN_M))
    p_dead = 1.0 / (1.0 + np.exp(-(DEAD_BIAS + DEAD_OUT_BONUS * out + DEAD_SLOW_BONUS * slow)))
    return np.log(np.clip(p_dead, EPS, 1 - EPS))


def emissions(trk):
    """(n, 3) float64 log-likelihoods for H, A, D (candidate 02)."""
    n, fps = len(trk["bx"]), trk["fps"]
    t_h, t_a, kick, still = _touches(trk)
    r = np.zeros(n)
    recent = np.zeros(n, bool)
    for _, lo, hi in trk["blocks"]:
        lead = (touch._since(t_a, lo, hi) - touch._since(t_h, lo, hi)).astype(np.float64)
        r[lo:hi + 1] = np.clip(lead / (TAU * fps), -CLIP, CLIP)
        recent[lo:hi + 1] = touch._since(kick, lo, hi) < LIVE_S * fps
    log_dead = _v1_log_dead(trk)
    logit = log_dead - np.log1p(-np.exp(log_dead)) + STILL_BONUS * still - KICK_LIVE * recent
    e = np.empty((n, 3), dtype=np.float64)
    e[:, 2] = -np.logaddexp(0.0, -logit)
    log_live = -np.logaddexp(0.0, logit)
    e[:, 0] = -np.logaddexp(0.0, -r) + log_live
    e[:, 1] = -np.logaddexp(0.0, r) + log_live
    return e


# -- touch_only decoder: v1 Viterbi ---------------------------------------------------------------

def _viterbi(e, sw, dc, kmin):
    """Most likely H/A/D path for one contiguous block (v1 mtah.decode.viterbi, K = kmin >= 2).

    Exact Viterbi over (H, A, D1..DK), O(1) per frame: dead ball is a forced chain, so arriving in
    DK from the chain at t is "entered at t-K+1" plus the summed D emissions (a prefix sum). The
    path may start in dead ball and end mid-chain. Ties are broken towards staying in the current
    state, so flat evidence carries the previous team instead of drifting to home.
    """
    n = len(e)
    eH, eA, eD = (e[:, j].tolist() for j in range(3))
    cD = np.concatenate(([0.0], np.cumsum(e[:, 2]))).tolist()   # cD[t+1] = sum eD[0..t]
    ninf = -np.inf
    enter = [ninf] * n            # score of being in D1 at t (just entered dead ball)
    enter_from = [0] * n          # H(0) or A(1) at t-1
    bH = bytearray(n)             # predecessor of H at t: 0=H 1=A 2=DK
    bA = bytearray(n)             # predecessor of A at t: 0=H 1=A 2=DK
    bK = bytearray(n)             # DK at t: 0 = stayed in DK, 1 = arrived from the chain
    # Starting in DK dominates starting at any earlier phase (same emissions, freer exit).
    sH, sA, sK = eH[0], eA[0], eD[0]
    for t in range(1, n):
        if sA > sH:
            enter[t], enter_from[t] = sA - dc + eD[t], 1
        else:
            enter[t], enter_from[t] = sH - dc + eD[t], 0
        t0 = t - kmin + 1
        kch = enter[t0] + (cD[t] - cD[t0 + 1]) if t0 >= 1 else ninf
        a, b, c = sH, sA - sw, sK - dc
        if a >= b and a >= c:
            nH, bH[t] = a, 0
        elif b >= c:
            nH, bH[t] = b, 1
        else:
            nH, bH[t] = c, 2
        a, b, c = sA, sH - sw, sK - dc
        if a >= b and a >= c:
            nA, bA[t] = a, 1
        elif b >= c:
            nA, bA[t] = b, 0
        else:
            nA, bA[t] = c, 2
        if sK >= kch:
            nK, bK[t] = sK, 0
        else:
            nK, bK[t] = kch, 1
        sH, sA, sK = nH + eH[t], nA + eA[t], nK + eD[t]

    best, state, t0_end = sH, 0, -1     # end: H, A, DK, or mid-chain (entered at t0 > n-K)
    if sA > best:
        best, state = sA, 1
    if sK > best:
        best, state = sK, 2
    for t0 in range(max(1, n - kmin + 1), n):
        v = enter[t0] + (cD[n] - cD[t0 + 1])
        if v > best:
            best, state, t0_end = v, 3, t0

    path = np.empty(n, dtype=np.int8)
    t = n - 1
    if state == 3:                      # ended mid-chain
        path[t0_end:] = 2
        t, state = t0_end - 1, enter_from[t0_end]
    while t >= 0:
        if state == 0:
            path[t] = 0
            prev = bH[t]
        elif state == 1:
            path[t] = 1
            prev = bA[t]
        else:                           # DK
            if t == 0:
                path[0] = 2
                break
            if bK[t] == 0:
                path[t] = 2
                prev = 2
            else:                       # chain entered at t0 = t-K+1 covers t0..t
                t0 = t - kmin + 1
                path[t0:t + 1] = 2
                t, state = t0 - 1, enter_from[t0]
                continue
        if t == 0:
            break
        state = prev
        t -= 1
    return path


def _decode_viterbi(trk, e):
    labels = np.full(len(e), "X", dtype="<U1")
    for half, lo, hi in trk["blocks"]:
        labels[lo:hi + 1] = decode_half(
            trk, half, lo, hi,
            lambda s, h: LETTERS[_viterbi(e[s:h + 1], SWITCH_COST, DEAD_COST, MIN_DEAD_FRAMES)])
    return labels


# -- semi_markov decoder: candidate 12's segment DP, costs of 02 ----------------------------------

def _log_dur(median_s, sigma, min_s, max_s, cell_s):
    """Shape-only log duration prior for k = 0..kmax cells: weighted lognormal log-mass minus its
    maximum (0 at the mode, negative elsewhere), -inf below the minimum."""
    kmax = int(round(max_s / cell_s))
    d = (np.arange(kmax + 1) + 0.5) * cell_s
    z = (np.log(d) - np.log(median_s)) / sigma
    logp = -0.5 * z * z - np.log(d * sigma * np.sqrt(2 * np.pi)) + np.log(cell_s)
    logp = DUR_WEIGHT * (logp - logp.max())
    logp[:int(np.ceil(min_s / cell_s))] = -np.inf
    return logp


def _segment_dp(cum, logdur, trans):
    """Exact semi-Markov Viterbi on the cell grid.

    cum: (3, m+1) prefix sums of cell emissions; logdur: per-state arrays log P(k cells);
    trans: (3, 3) log transition scores (row = from, column = to). The first and last segments are
    clipped by the block edge, so their duration prior is dropped but their minimum is kept.
    Returns a list of (state, start_cell, end_cell).
    """
    m = cum.shape[1] - 1
    ninf = -np.inf
    F = np.full((3, m + 1), ninf)        # best score of [0, t) with a segment of state s ending at t
    arg = np.zeros((3, m + 1), dtype=np.int64)    # its start cell
    prev = np.full((3, m + 1), -1, dtype=np.int64)  # state of the segment before it (-1: first)
    kmin = [int(np.argmax(np.isfinite(ld))) for ld in logdur]
    for t in range(1, m + 1):
        for s in range(3):
            lo_u, hi_u = max(0, t - len(logdur[s]) + 1), t - kmin[s]
            if hi_u < lo_u:
                if t < m:
                    continue
                lo_u = hi_u = 0                              # block shorter than the minimum
            u = np.arange(lo_u, hi_u + 1)
            G = F[:, u] + trans[:, s][:, None]               # enter s at u from another state
            src = np.argmax(G, axis=0)
            first = u == 0                                   # segment starts at the block edge
            g = np.where(first, 0.0, G[src, np.arange(len(u))])
            w = np.where(first | (t == m), 0.0, logdur[s][t - u])
            val = g + w + cum[s, t] - cum[s, u]
            j = int(np.argmax(val))
            F[s, t], arg[s, t] = val[j], u[j]
            prev[s, t] = -1 if first[j] else src[j]
    segs, s, t = [], int(np.argmax(F[:, m])), m
    while t > 0:
        u = int(arg[s, t])
        segs.append((s, u, t))
        s, t = int(prev[s, t]), u
    return segs[::-1]


def _refine(e, cut, a, b, lo, hi):
    """Frame of the a->b boundary in [lo, hi] that maximises emissions (a before, b after)."""
    cand = np.arange(lo, hi + 1)
    gain = np.concatenate(([0.0], np.cumsum(e[lo:hi, a] - e[lo:hi, b])))
    return int(cand[np.argmax(gain)]) if len(cand) else cut


def _semi_markov_block(e, fps):
    """H/A/D letters for one contiguous block of emissions e (n, 3)."""
    n = len(e)
    m = -(-n // CELL)
    pad = np.zeros((m * CELL, 3))
    pad[:n] = e
    cells = pad.reshape(m, CELL, 3).sum(axis=1)
    cum = np.concatenate([np.zeros((3, 1)), np.cumsum(cells.T, axis=1)], axis=1)
    cs = CELL / float(fps)
    live = _log_dur(LIVE_MEDIAN_S, LIVE_SIGMA, LIVE_MIN_S, LIVE_MAX_S, cs)
    dead = _log_dur(DEAD_MEDIAN_S, DEAD_SIGMA, DEAD_MIN_S, DEAD_MAX_S, cs)
    # Diagonal = chaining two segments of the same state: the output merges them into ONE run.
    trans = np.array([[-SWITCH_COST, -SWITCH_COST, -DEAD_COST],
                      [-SWITCH_COST, -SWITCH_COST, -DEAD_COST],
                      [-DEAD_COST, -DEAD_COST, -DEAD_COST]])
    segs = _segment_dp(cum, [live, live, dead], trans)
    out = np.empty(n, dtype=np.int8)
    starts = [0]
    for (sa, _, cut), (sb, _, _) in zip(segs, segs[1:]):
        c = cut * CELL                    # boundary moves at most half a cell either way
        starts.append(_refine(e, c, sa, sb, max(starts[-1] + 1, c - CELL // 2),
                              min(n - 1, c + CELL // 2)))
    starts.append(n)
    for (s, _, _), lo, hi in zip(segs, starts, starts[1:]):
        out[lo:hi] = s
    return LETTERS[out]


def _decode_semi_markov(trk, e):
    labels = np.full(len(e), "X", dtype="<U1")
    for half, lo, hi in trk["blocks"]:
        labels[lo:hi + 1] = decode_half(
            trk, half, lo, hi,
            lambda s, h: _semi_markov_block(e[s:h + 1], trk["fps"]))
    return labels


def decode(trk, variant):
    """(n,) H/A/D labels inside the graded halves, X outside, for one of VARIANTS."""
    if variant not in VARIANTS:
        raise ValueError("unknown fallback variant {!r}; use one of {}".format(variant, VARIANTS))
    e = emissions(trk)
    return (_decode_viterbi if variant == "touch_only" else _decode_semi_markov)(trk, e)
