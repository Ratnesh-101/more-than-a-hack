# 05 Ball reconstruction (advocate 05)

**Claim.** Reject physically impossible ball samples, bridge every gap with a straight line whose variance follows a random-velocity ball, and feed nearest-player evidence the *expected* distance to that uncertain ball, sqrt(d^2 + var). The ball's position in a gap becomes graded evidence that fades to flat as uncertainty grows, so the HMM places hand-overs inside gaps instead of at the reappearance, and never invents a team on a ball it cannot locate.

**Football rationale.** v1 sees nothing on the 41% of frames without a ball, so a hand-over hidden in a gap is only switched when the ball reappears. Over a short gap the ball keeps its line (pass or carry), so the endpoints locate it well. Over a long gap direction is lost after about one touch, and honest evidence must go flat. Glitches are rejected by a speed bound, not by pattern. Dead-ball and decoder are v1's, so the measured delta is due to the ball alone.

| constant | value | justification |
|---|---|---|
| BALL_VMAX_MPS | 35 m/s | hardest struck balls in elite football; anything faster vs its neighbourhood is not a ball |
| GLITCH_HALF_S | 0.4 s | median window wide enough that a multi-frame glitch stays a minority |
| SIGMA_OBS_M | 0.5 m | detector + homography noise of a seen ball |
| BALL_VRMS_MPS | 7 m/s | rms open-play ball speed: carries 2-5 m/s mixed with 12-20 m/s passes |
| BALL_TAU_S | 1.0 s | ball direction memory: a touch or pass every 1-2 s |
| OUT_MARGIN_M | 0.5 m | v1's line margin (out cue now from clean samples only) |
| PARAMS | {} | v1 switch 30 / dead 12; not touched |

**Dev results (seeds 1-5, 20 min, `court.run --seeds 1,2,3,4,5 --cand 05_ball`).**

| | spellF1 realistic | macroF1 realistic | spellF1 easy | macroF1 easy | share <3 s | poss pred/true |
|---|---|---|---|---|---|---|
| incumbent | 0.4955 +/- 0.044 | 0.7323 | 0.6765 | 0.8105 | 0.10 | 56/66 |
| **cand 05** | **0.5147 +/- 0.048** | 0.7386 | 0.6722 | 0.8076 | 0.07 | 56/66 |

Paired realistic delta per seed: +0.032, +0.008, -0.013, +0.025, +0.044 (mean +0.019, 4/5 up). That is below the ~0.04 real-match noise band, so it is a consistent small gain, not a proven one. Easy mode is -0.004 (no gaps there; the cost is the 0.5 m blur on seen balls). Runtime: 0.37 s for emissions on a 199,319-frame tiled synthetic Trk (v1 0.20 s), 231 MB peak, deterministic, trk unmodified.

Diagnostics (dev seeds): with the TRUE ball on every frame the incumbent reaches 0.646, so the missing ball is worth about 0.15 and this candidate recovers about 1/8 of it. Removing glitches alone changes nothing (0.494 vs 0.496): the switch cost already absorbs 2-6 frame jumps.

**Tried and failed.**
- **Player-swarm prior for long gaps** (mean shift over nearby players, fused by inverse variance). It cut median position error (16.5 to 11.5 m on long gaps) but lowered nearest-team accuracy (0.61 to 0.47-0.52 on 3-6 m gaps). The swarm centre sits between the carrier and his pressers, so its nearest player is a coin flip. Net spellF1 was +/-0.003. Removed.
- **Velocity-convergence point** (least-squares intersection of player running lines): no better than the bridge. Not used.
- **First variance model, linear D = 25 m^2/s** (v = 5 m/s x 1 s, mis-derived): 0.500. It was 4x under-dispersed against the dev-seed error, so it was replaced by the correlated-velocity form.

**Known weaknesses (disclosed).**
1. **v and tau were influenced by the synthetic.** The functional form came from physics, but I switched to it after a dev-seed calibration check (implied long-gap D of about 90-100 m^2/s; my constants give 98). The values are football-plausible, but real ball statistics may differ, and the sweep showed realistic spellF1 moves by about 0.02 across D = 10-100.
2. **Long gaps (about 70% of gap frames) carry no team information.** Their evidence goes flat by design, so hand-overs inside long gaps are still found late.
3. **The straight-line bridge ignores bounces, deflections and the ball leaving play inside a gap.** A gap containing an out-of-play is bridged as live, and the D state still relies on player slowness.
4. **Glitch rejection needs neighbours.** A glitch next to a gap, or one longer than about 10 frames, survives. On real data a long run of a wrong-object track (e.g. a second ball) would not be caught.
5. **One-sided edges.** At block starts and ends the last sample is held with a growing variance.
6. **No real-data check yet.** It has run only on synthetic data; the `--real` diagnostics are pending.
7. **The synthetic ball sits 0.7 m ahead of its carrier** (proximity-friendly). On real data the nearest player to an interpolated ball is less reliable, so the gain may shrink.
