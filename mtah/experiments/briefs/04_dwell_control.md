# 04 Dwell-time control

**Claim.** A team has possession when one of its players *stays* with the ball: near it and moving with it, for a sustained time. Per-team contact, integrated with a two-sided exponential decay that is frozen (not reset) through ball-missing gaps, gives team evidence that beats the incumbent on every dev seed (realistic +0.032 spellF1, easy +0.085).

**Football rationale.** CHALLENGE.md proposes "a player keeps the ball for a few seconds" as the proxy for control. A touch, deflection or a pass flying past a player is brief or fast relative to that player, so it adds little; a carry or a kept reception adds evidence on every frame. Summing a forward and a backward leaky integrator gives a symmetric kernel, so the hand-over is not delayed (a causal filter would be TAU*ln2 late, and 1 s late costs about 0.35 spellF1). When neither team has dwelt on the ball, the evidence fades to 50/50 and the decoder keeps the last possessor (the rule for a loose ball). The dead-ball model is the incumbent's, unchanged.

| constant | value | justification |
|---|---|---|
| CONTROL_RADIUS_M | 1.5 m | playing distance: the ball is reachable within ~1-2 m of a player's centre |
| REL_SPEED_MPS | 5 m/s | a hard dribble touch runs ~3-5 m/s ahead of the carrier; passes are >= 8-10 m/s |
| VEL_HALF_WIN | 4 frames | 0.32 s central difference: ~0.2 m position noise becomes ~1 m/s velocity noise, shorter than a touch cycle |
| TAU_S | 0.5 s | per-side decay; the two-sided kernel spans ~1-2 s, a "kept" reception |
| CTRL_SCALE_S | 0.25 s | a full exclusive dwell (~1 s integrated) gives per-frame odds ~e^4, about the incumbent's strongest evidence |
| CONF_DWELL_S | 0.5 s | integrated dwell at which team evidence is trusted 63%; below it the ball is loose |
| switch_cost | 15 (incumbent 30) | the emission is already a temporal integral, so less decoder smoothing is needed. **Chosen on dev seeds after 30 under-segmented** |
| CHUNK | 256 | numerical only: keeps a^-CHUNK finite in the vectorised integrator |

**Dev results (seeds 1-5, 20 min, mean +/- sd)**

| | spellF1 realistic | macroF1 realistic | spellF1 easy | macroF1 easy | poss pred/true |
|---|---|---|---|---|---|
| incumbent | 0.4955 +/- 0.044 | 0.7323 | 0.6765 | 0.8105 | 56/66 |
| **04 dwell** | **0.5272 +/- 0.043** | **0.7689** | **0.7611** | **0.8494** | 54/66 |
| control: incumbent at switch_cost 15 | 0.4644 +/- 0.048 | | 0.6462 | | |

The candidate wins all 5 paired seeds in both modes. The control row shows that the gain comes from the feature, not from the lower switch cost. Runtime is 0.5 s for emissions on a 199,319-frame tiled synthetic Trk (~0.6 s with Viterbi), with ~440 MB peak RSS.

**Known weaknesses**
1. The realistic gain (+0.032) is below the ~0.04 real-match noise band; it is consistent across seeds but small.
2. **The "moves with the ball" gate does not earn its keep on synthetic data.** Ablated (gate off), it gives 0.525 at sw15 and 0.519 at sw30, statistically the same or better. The synthetic ball sticks to the carrier and rarely passes close to a player, so the gate is kept on a football prior (deflections, passes past players, ball glitches) that the synthetic court cannot test.
3. switch_cost 15 was chosen on dev seeds; the mechanism argument came after the fact. The other constants were set as priors, and a dev sweep (TAU 0.5/1, CTRL_SCALE 0.125-0.5, CONF 0.25/0.5, radius 1.5/2.5) moved scores within ~0.01, except CTRL_SCALE 0.5 (-0.03).
4. Possessions are still under-counted (54/66): true possessions under 4 s are matched at IoU >= 0.5 only 14% of the time (same as the incumbent; 4-8 s: 0.38 vs 0.33), since dwell evidence needs time to build by design.
5. The dead-ball model is the incumbent's. About 12% of true-live frames are labelled D, which splits possessions; this is the largest remaining error and it is not addressed here. Trying "sustained control implies live" (bonus 2 or 4) dropped spellF1 to ~0.38-0.42, because restarters also dwell with the ball.
6. Freezing compresses the gap out of the timeline, so evidence from after a gap bleeds into the frame before it. A hand-over hidden inside a gap is placed at the gap edge chosen by the decoder, not at its true time.
7. Contact uses the nearest slot of each team. A tracklet in the wrong-team slot or a duplicate player near the ball gives false contact. There is no identity-level dwell (slots are tracklets).
8. The integrator is non-causal (offline only), which is fine for Phase 1.
9. It is not validated on real data (none available yet). REL_SPEED and the velocity window assume real ball position noise near ~0.2 m.
