"""Synthetic match with known possession truth: a stress test, not random noise.

make(seed, minutes=20, difficulty="realistic") -> (Trk, truth)
  truth: array of 'H' 'A' 'D' per frame, 'X' outside the two blocks.

Layer 1 (the game; identical for every difficulty of a seed): 22 players follow loose formations
that shift toward the ball; a possession is a chain of carries and passes of lognormal length
(median ~10 s, ~12% under 4 s) that ends in a tackle + scramble, an intercepted pass, an aerial
duel, the ball going out, a foul or a goal. Inside a possession there are also scrambles and aerial
duels that the SAME team wins (nearest-player evidence flips, truth does not). A contested ball
stays with the last controller until someone controls it. Stoppages: the ball sits (out of play or
on the spot), players walk, a clear team restarts.

Layer 2 (observation, "realistic" only): 46 slot tracklets for 22 players that start, end and
vanish, sit in a wrong-team slot (~1%) or duplicate a player; position/speed noise; the ball is
missing for ~41% of in-block frames in bursty log-normal gaps, more often around tackles,
interceptions and aerials (so some gaps swallow hand-overs), plus short 20+ m glitches.
"easy" observes layer 1 perfectly: full ball, no noise, one tracklet per player.
"""

import numpy as np

from .trk import FPS, GAME, GAMES, N_SLOTS, build

L, W = GAMES[GAME]["pitch"]
DT = 1.0 / FPS
LETTERS = np.array(["H", "A", "D"])
TARGET_MISSING = 0.41
LEAD, MID, TAIL = 300, 3000, 200  # frames before half 1, between the halves, after half 2

# (depth along the attack direction, width) of 11 players, goalkeeper first
_FORM = np.array([[.04, .5], [.25, .12], [.22, .38], [.22, .62], [.25, .88], [.45, .25],
                  [.42, .5], [.45, .75], [.65, .15], [.7, .5], [.65, .85]])
_PULL = np.array([.15] + [.4] * 4 + [.5] * 6)
_END = (("tackle", .33), ("intercept", .27), ("aerial", .18), ("out", .14), ("foul", .06),
        ("goal", .02))
_CENTRE = np.array([L / 2, W / 2])
_TOUCH = np.array([2.0, 2.0]), np.array([L - 2, W - 2])


def _frames(rng, median_s, sigma, lo_s, hi_s):
    return int(np.clip(rng.lognormal(np.log(median_s), sigma), lo_s, hi_s) * FPS)


class _Half:
    """One half of play simulated frame by frame (positions, ball, truth, occlusion windows)."""

    def __init__(self, rng, flip, first_team):
        self.rng, self.t = rng, 0
        self.dir = np.repeat([1.0, -1.0], 11) * (-1.0 if flip else 1.0)
        form = _FORM + rng.normal(0, .02, _FORM.shape)
        self.base = np.stack([L / 2 + self.dir * (np.tile(form[:, 0], 2) - .5) * .95 * L,
                              np.tile(form[:, 1], 2) * W * .9 + .05 * W], axis=1)
        self.pull = np.tile(_PULL, 2)
        self.phase = rng.uniform(0, 6.3, (22, 2))
        self.period = rng.uniform(15, 40, (22, 2)) * FPS
        self.P, self.V = self.base.copy(), np.zeros((22, 2))
        self.ball, self.bv = _CENTRE.copy(), np.zeros(2)  # bv in metres per frame
        self.windows, self.win_t0, self.T = [], None, first_team
        self._enter_dead("kickoff", first_team, _CENTRE, 8 * FPS)

    # -- helpers ------------------------------------------------------------------------------
    def _near(self, team, k, point=None):
        point = self.ball if point is None else point
        d = np.hypot(*(self.P[team * 11:team * 11 + 11] - point).T)
        return team * 11 + np.argpartition(d, k)[:k]

    def _touch(self, p):
        """Where the ball sits relative to its carrier: just ahead of his running direction."""
        v = self.V[p]
        s = np.hypot(*v)
        return self.P[p] + .7 * (v / s if s > .3 else np.array([self.dir[p], 0.0]))

    def _close_window(self):
        if self.win_t0 is not None:
            self.windows.append((self.win_t0, self.t))
            self.win_t0 = None

    # -- state transitions --------------------------------------------------------------------
    def _begin(self, team, owner):
        """`team` gains control with `owner`: a new possession starts (truth switches here)."""
        self._close_window()
        self.T, self.owner, self.mode = team, owner, "carry"
        self.send = self.t + _frames(self.rng, 6.5, 1.1, 1.0, 90)
        self.carry_v = self.rng.uniform(2.5, 5.0)
        self._new_hold()

    def _continue(self, owner):
        self._close_window()
        self.owner, self.mode = owner, "carry"
        self._new_hold()

    def _new_hold(self):
        self.hold = min(_frames(self.rng, 1.8, .5, .5, 6), max(1, self.send - self.t))

    def _enter_dead(self, kind, restart_team, dest, dur):
        self.mode, self.R, self.dest = "dead", restart_team, np.asarray(dest, float)
        self.dead_t0, self.dead_len, self.from_ball = self.t, dur, self.ball.copy()
        self.restarter = int(self._near(restart_team, 1, self.dest)[0])
        self.win_t0 = None

    def _go_loose(self, win_team, kick_mps, winner=None):
        rng = self.rng
        self.mode, self.l_t0, self.l_min = "loose", self.t, int(rng.integers(12, 45))
        ang = rng.uniform(0, 2 * np.pi)
        self.bv = kick_mps * DT * np.array([np.cos(ang), np.sin(ang)])
        self.win_team = win_team
        self.winner = int(rng.choice(self._near(win_team, 3)) if winner is None else winner)

    def _start_pass(self, receiver, interceptor=False):
        if interceptor:  # an opponent standing near the pass lane takes it
            lane = self.ball + .65 * (self.P[receiver] - self.ball)
            receiver = int(self._near(1 - self.T, 1, lane)[0])
        dist = np.hypot(*(self.P[receiver] - self.ball))
        self.mode, self.recv = "pass", receiver
        self.p_end = self.t + max(8, int(dist / self.rng.uniform(11, 20) * FPS))
        self.recv_goal = self.P[receiver] + self.V[receiver] * (self.p_end - self.t) * DT

    def _start_aerial(self, win_team):
        rng = self.rng
        dest = self.ball + np.array([self.dir[self.owner] * rng.uniform(22, 38), rng.uniform(-12, 12)])
        self.a_dest = np.clip(dest, _TOUCH[0] + 1, _TOUCH[1] - 1)
        dist = np.hypot(*(self.a_dest - self.ball))
        self.a_end = self.t + max(20, int(dist / rng.uniform(13, 17) * FPS))
        self.win_team, self.mode, self.win_t0 = win_team, "aerial", self.t
        self.bv = (self.a_dest - self.ball) / (self.a_end - self.t)

    def _pick_receiver(self):
        o = self.owner
        mates = np.array([p for p in range(self.T * 11, self.T * 11 + 11) if p != o])
        dist = np.hypot(*(self.P[mates] - self.ball).T)
        fwd = (self.P[mates, 0] - self.ball[0]) * self.dir[o]
        w = np.exp(-np.abs(dist - 16) / 9) * np.where(fwd > -2, 1.6, .6)
        w[mates % 11 == 0] *= .2
        return int(self.rng.choice(mates, p=w / w.sum()))

    def _end_possession(self):
        rng, o = self.rng, self.owner
        kind = _END[rng.choice(len(_END), p=[p for _, p in _END])][0]
        if kind in ("tackle", "intercept", "aerial"):
            self.win_t0 = self.t  # the ball is likely hidden by the scrum / in the air
        if kind == "tackle":
            self.mode, self.tackler, self.k_t0 = "tackle", int(self._near(1 - self.T, 1)[0]), self.t
        elif kind == "intercept":
            self._start_pass(self._pick_receiver(), interceptor=True)
        elif kind == "aerial":
            self._start_aerial(1 - self.T if rng.random() < .6 else self.T)
        elif kind == "foul":
            self.ball = self._touch(o)
            self.bv[:] = 0
            self._enter_dead("foul", 1 - self.T if rng.random() < .7 else self.T, self.ball,
                             _frames(rng, 10, .5, 6, 40))
        else:  # the ball flies over a line
            d = self.dir[o]
            self.out_kind = "goal" if kind == "goal" else ("endline" if rng.random() < .35 else "throw")
            if self.out_kind == "throw":
                y = -rng.uniform(.8, 2.5) if self.ball[1] < W / 2 else W + rng.uniform(.8, 2.5)
                dest = np.array([np.clip(self.ball[0] + d * rng.uniform(5, 20), 3, L - 3), y])
            else:
                x = L + rng.uniform(.8, 2.5) if d > 0 else -rng.uniform(.8, 2.5)
                dest = np.array([x, rng.uniform(30, 38) if kind == "goal" else rng.uniform(8, 60)])
            self.mode = "outflight"
            self.bv = DT * rng.uniform(12, 18) * (dest - self.ball) / np.hypot(*(dest - self.ball))

    def _cross_line(self):
        """The ball has left the pitch: play stops; pick the restart."""
        rng, ball, d = self.rng, self.ball, self.dir[self.owner]
        if self.out_kind == "goal":
            self._enter_dead("goal", 1 - self.T, _CENTRE, _frames(rng, 45, .2, 30, 70))
        elif self.out_kind == "endline":
            end_x = L if d > 0 else 0.0
            corner = rng.random() < .3
            dest = [end_x, 0.0 if ball[1] < W / 2 else W] if corner else [end_x - d * 5.5, W / 2]
            self._enter_dead("endline", self.T if corner else 1 - self.T, dest,
                             _frames(rng, 12, .5, 6, 40))
        else:
            dest = [np.clip(ball[0], 1, L - 1), 0.0 if ball[1] < W / 2 else W]
            self._enter_dead("throw", 1 - self.T if rng.random() < .8 else self.T, dest,
                             _frames(rng, 9, .5, 6, 40))

    # -- one frame ----------------------------------------------------------------------------
    def step(self, t):
        """Advance the world one frame; returns the truth code (0 H, 1 A, 2 D) of this frame."""
        self.t = t
        rng, P, V, ball, mode = self.rng, self.P, self.V, self.ball, self.mode
        ref = self.dest if mode == "dead" else ball
        tgt = self.base + self.pull[:, None] * (ref - _CENTRE) * [1.0, .35]
        tgt = tgt + 2.0 * np.sin(2 * np.pi * t / self.period + self.phase)
        vmax = np.full(22, 3.0)

        def chase(players, point, speed):
            tgt[players], vmax[players] = point, speed

        if mode == "dead":
            frac = (t - self.dead_t0) / max(1, self.dead_len)
            if np.hypot(*self.bv) > .001:
                ball += self.bv
                self.bv *= .9
                self.from_ball = ball.copy()
            ball[:] = self.from_ball + (self.dest - self.from_ball) * np.clip((frac - .3) / .4, 0, 1)
            vmax[:] = 1.6
            chase(self.restarter, self.dest, 4.0)
            if frac >= 1.0:
                r = self.restarter
                if np.hypot(*(P[r] - self.dest)) > 3:
                    P[r] = self.dest + rng.normal(0, .5, 2)
                self._begin(self.R, r)
        elif mode == "carry":
            o = self.owner
            lateral = 3 * np.sin(t / 60.0 + o)
            chase(o, np.clip(P[o] + [self.dir[o] * 8, lateral], *_TOUCH), self.carry_v)
            for j, p in enumerate([p for p in self._near(self.T, 3) if p != o][:2]):
                chase(p, ball + [self.dir[o] * 6, 5.0 * (-1) ** j], 4.5)
            chase(self._near(1 - self.T, 2), ball, 5.5)
            self.hold -= 1
            if self.hold <= 0:
                u = rng.random()
                if t >= self.send:
                    self._end_possession()
                elif u < .08:
                    self._go_loose(self.T, 4.0)
                elif u < .14:
                    self._start_aerial(self.T)
                else:
                    self._start_pass(self._pick_receiver())
        elif mode == "pass":
            r, remaining = self.recv, self.p_end - t
            chase(self._near(1 - self.T, 1), ball, 5.5)
            chase(r, self.recv_goal, 6.0)
            if remaining <= 1:
                ball[:] = self._touch(r)
                if r // 11 == self.T:
                    self._continue(r)
                else:
                    self._begin(1 - self.T, r)
            else:
                ball += (P[r] + V[r] * remaining * DT - ball) / remaining
        elif mode == "tackle":
            o, q = self.owner, self.tackler
            chase(o, np.clip(P[o] + [self.dir[o] * 8, 0], *_TOUCH), 3.5)
            chase(q, P[o], 7.0)
            if np.hypot(*(P[q] - P[o])) < 1.4 or t - self.k_t0 > 90:
                self._go_loose(1 - self.T, 5.0, q if rng.random() < .7 else None)
        elif mode == "aerial":
            for team in (0, 1):
                chase(self._near(team, 3, self.a_dest), self.a_dest, 6.0)
            ball += self.bv
            if t >= self.a_end:
                self._go_loose(self.win_team, 1.5)
        elif mode == "loose":
            w, age = self.winner, t - self.l_t0
            for team in (0, 1):
                chase(self._near(team, 3), ball, 6.5)
            chase(w, ball, 7.0)
            ball += self.bv
            self.bv *= .92
            if age >= self.l_min:
                ball += (P[w] - ball) * .2
            if (age >= self.l_min and np.hypot(*(ball - P[w])) < .9) or age > 120:
                ball[:] = self._touch(w)
                if self.win_team == self.T:
                    self._continue(w)
                else:
                    self._begin(self.win_team, w)
        else:  # outflight
            ball += self.bv
            for team in (0, 1):
                chase(self._near(team, 1), ball, 5.0)
            if not (0 <= ball[0] <= L and 0 <= ball[1] <= W):
                self._cross_line()

        d = tgt - P
        dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        want = d / dist[:, None] * np.minimum(vmax, dist * 1.2)[:, None]
        V += np.clip(want - V, -.18, .18) + rng.normal(0, .03, (22, 2))
        P += V * DT
        np.clip(P, [-.5, -.5], [L + .5, W + .5], out=P)
        if self.mode in ("carry", "tackle"):
            ball[:] = self._touch(self.owner)
        return 2 if self.mode == "dead" else self.T

    def run(self, n):
        pos, spd, ball = (np.empty(s, np.float32) for s in ((n, 22, 2), (n, 22), (n, 2)))
        truth = np.empty(n, np.int8)
        for t in range(n):
            truth[t] = self.step(t)
            pos[t], spd[t], ball[t] = self.P, np.hypot(*self.V.T), self.ball
        return pos, spd, ball, truth, self.windows


def _play(rng, minutes):
    """Both halves laid out on one timeline; NaN/X outside the two blocks."""
    half = int(minutes * 60 * FPS / 2)
    blocks = [(1, LEAD, LEAD + half - 1), (2, LEAD + half + MID, LEAD + 2 * half + MID - 1)]
    n = blocks[-1][2] + 1 + TAIL
    pos, spd = np.full((n, 22, 2), np.nan, np.float32), np.full((n, 22), np.nan, np.float32)
    ball, code = np.full((n, 2), np.nan, np.float32), np.full(n, 3, np.int8)
    windows, first = [], int(rng.integers(2))
    for (h, lo, hi), team in zip(blocks, (first, 1 - first)):
        p, s, b, c, w = _Half(rng, h == 2, team).run(hi - lo + 1)
        pos[lo:hi + 1], spd[lo:hi + 1], ball[lo:hi + 1], code[lo:hi + 1] = p, s, b, c
        windows += [(a + lo, z + lo) for a, z in w]
    return pos, spd, ball, code, blocks, windows


# -- observation layer ------------------------------------------------------------------------

def _tracklets(rng, blocks, realistic):
    """(player, first, last, offset) pieces: a slot follows a player for a while, then is lost."""
    out = []
    for _, lo, hi in blocks:
        for player in range(22):
            t = lo
            while t <= hi:
                end = hi if not realistic else min(hi, t + _frames(rng, 60, 1.0, 4, 1e5))
                out.append((player, t, end, np.zeros(2)))
                if realistic and rng.random() < .01:  # tracker duplicates a player
                    ang = rng.uniform(0, 6.28)
                    out.append((player, t, min(end, t + 200),
                                rng.uniform(.3, 1.0) * np.array([np.cos(ang), np.sin(ang)])))
                t = end + 1 + (_frames(rng, 1.5, 1.0, .1, 20) if realistic and rng.random() < .5 else 0)
    return sorted(out, key=lambda x: x[1])


def _slots(rng, tracklets, realistic):
    """Greedy slot allocation; a slot's team is fixed, ~1% of tracklets land in a wrong-team slot."""
    slot_team = rng.permutation(np.repeat([0, 1], N_SLOTS // 2)).astype(np.int8)
    free_at = np.zeros(N_SLOTS, int)
    placed = []
    for player, a, b, off in tracklets:
        want = player // 11 if not (realistic and rng.random() < .01) else 1 - player // 11
        free = np.flatnonzero((slot_team == want) & (free_at <= a))
        if len(free):
            s = int(rng.choice(free))
            free_at[s] = b + 1
            placed.append((s, player, a, b, off))
    return slot_team, placed


def _ball_missing(rng, n, blocks, windows):
    """Bursty ball dropouts reaching TARGET_MISSING of in-block frames."""
    miss = np.zeros(n, bool)
    inside = np.zeros(n, bool)
    for _, lo, hi in blocks:
        inside[lo:hi + 1] = True
    for a, z in windows:  # occlusions around contested events
        if rng.random() < .55:
            miss[max(0, a - int(rng.uniform(0, FPS))):z + _frames(rng, 1.5, 1.0, 0, 30)] = True
    starts = np.flatnonzero(inside)
    need = int(TARGET_MISSING * len(starts))
    while (miss & inside).sum() < need:
        gap = _frames(rng, 3.5, 1.2, .4, 90)
        a = int(rng.choice(starts))
        miss[a:a + min(gap, need - int((miss & inside).sum()) + FPS)] = True
    return miss & inside


def _observe(rng, pos, spd, ball, blocks, windows, realistic):
    n = len(ball)
    slot_team, placed = _slots(rng, _tracklets(rng, blocks, realistic), realistic)
    px, py, ps = (np.full((n, N_SLOTS), np.nan, np.float32) for _ in range(3))
    for s, player, a, b, off in placed:
        px[a:b + 1, s] = pos[a:b + 1, player, 0] + off[0]
        py[a:b + 1, s] = pos[a:b + 1, player, 1] + off[1]
        ps[a:b + 1, s] = spd[a:b + 1, player]
    bx, by = ball[:, 0].copy(), ball[:, 1].copy()
    if realistic:
        for arr, sd in ((px, .15), (py, .15), (ps, .25), (bx, .2), (by, .2)):
            arr += rng.normal(0, sd, arr.shape).astype(np.float32)
        np.maximum(ps, 0, out=ps)
        ps[rng.random(ps.shape) < .02] = np.nan
        miss = _ball_missing(rng, n, blocks, windows)
        visible = np.flatnonzero(~miss & np.isfinite(bx))
        for _ in range(int(len(visible) / FPS / 80)):  # ~one glitch per 80 s of visible ball
            a = int(rng.choice(visible))
            ang = rng.uniform(0, 6.28)
            jump = rng.uniform(20, 40) * np.array([np.cos(ang), np.sin(ang)])
            span = slice(a, a + int(rng.integers(2, 7)))
            bx[span] += jump[0]
            by[span] += jump[1]
        bx[miss], by[miss] = np.nan, np.nan
    return px, py, ps, slot_team, bx, by


def make(seed, minutes=20, difficulty="realistic"):
    """(Trk, truth) for one deterministic synthetic match."""
    if difficulty not in ("easy", "realistic"):
        raise ValueError("difficulty must be 'easy' or 'realistic', got {!r}".format(difficulty))
    pos, spd, ball, code, blocks, windows = _play(np.random.default_rng([seed, 0]), minutes)
    obs = _observe(np.random.default_rng([seed, 1]), pos, spd, ball, blocks, windows,
                   difficulty == "realistic")
    truth = np.where(code == 3, "X", LETTERS[np.minimum(code, 2)])
    return build(*obs, (L, W), blocks), truth
