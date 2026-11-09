# 10_movement: collective reaction to the ball (Advocate 10)

**Claim.** The team without the ball converges on it, and the team with the ball does not. So the difference between the two teams' mean closing speed to the ball is possession evidence that keeps working when the ball is hidden. Smoothed, read 0.5 s ahead and bounded, then added to the incumbent's nearest-player softmax, it raises realistic spellF1 from 0.496 to 0.551 on dev seeds 1-5 and brings the possession count from 56 to 65 (true count 66).

**Football rationale.** Defenders press the carrier, cover the passing lanes and shift toward the ball. Attackers stay level with the ball or run away from it to make space and offer passes. The ball also travels toward the defending team's goal. The read is closing speed, d/dt of the player-to-ball distance averaged over a whole team, and every one of these behaviours shows up in it. A team needs about 0.5-1.5 s to react to a hand-over, so the read is a slow confirmation. When the ball is hidden it carries possession across the gap (gaps are 41% of realistic frames). When the ball is visible it stops a brief proximity flip from turning into a new possession. Dead-ball evidence is the incumbent's, unchanged. Default decoder and costs.

| constant | value | justification |
|---|---|---|
| HALF_DIFF | 5 fr | central difference over +-0.2 s, long enough to beat ~0.15 m position jitter |
| MAX_SPEED | 12 m/s | above sprint speed, so it is a slot swap or ball glitch and is dropped |
| UNIT | 1 m/s | a team-wide closing gap of walking pace = one tanh unit |
| LAG_S | 0.5 s | ~0.3 s reaction + ~0.2 s to accelerate; the window then covers t-0.5 .. t+1.5 s |
| WIN_S | 2 s | a press or recovery run lasts a couple of seconds |
| NATS_PER_S | 10 | ~3 s of saturated team-wide press pays one switch_cost (30) |
| ball interp | no gap limit | the line between the last and next sightings is still the best estimate of where play is |

**Dev results** (court, seeds 1-5, 20 min, mean +/- sd over seeds)

| | spellF1 real. | macroF1 real. | spellF1 easy | poss pred/true | share <3 s | flicker/min |
|---|---|---|---|---|---|---|
| incumbent | 0.4955 +/- 0.044 | 0.7323 | 0.6765 | 56/66 | 0.10 | 0.78 |
| 10_movement | **0.5507 +/- 0.046** | 0.7437 | 0.7161 | 65/66 | 0.09 | 1.21 |

- Per-seed realistic delta: +0.057, +0.024, **-0.035 (seed 3)**, +0.145, +0.085. Easy delta: +0.005 to +0.073, so it helps even with the ball fully visible.
- Runtime on a 201k-frame tiled Trk: emissions 0.39 s, 154 MB peak (incumbent: 0.22 s, 112 MB).

**Known weaknesses**
1. **NATS_PER_S was chosen on the dev seeds.** I tried {3, 6, 10, 15, 25}, and 10 was best (3 gives 0.528, 25 gives 0.505, worse than 15). The "3 s pays a switch" argument came after the fact. The score is peaked in this constant.
2. **LAG_S and WIN_S do not matter on the synthetic data.** Lag 0-1 s and window 2-4 s all land within +-0.01. The synthetic chasers react on the hand-over frame, so the ~1 s real reaction lag is NOT validated anywhere.
3. **Parts of my brief failed.** Team velocity along the attack axis ("push") and the lateral ball-side shift of the team centroid both have AUC ~0.50 on the synthetic data, so I removed them. I do not use `ps` (speed has no direction; closing speed comes from positions).
4. **The signal is weak per frame.** Alone it has AUC 0.73-0.80 (smoothed, realistic). On the synthetic data it mainly comes from a few chasers and supporters plus the ball moving toward the defending team. Real football can turn it upside down: a low block retreating, a goalkeeper or back-line recycling the ball, both teams sprinting the same way on a counter, a team that wins the ball while still converging.
5. **No limit on ball gaps.** Across a 20-30 s gap the ball "position" is a straight line, so the read measures convergence on a made-up path.
6. **Flicker rises from 0.78 to 1.21 per minute** (truth is 2.3). This is more switches, still under the truth, but one seed (3) lost spellF1.
7. **Slot handling.** Goalkeepers are included in the team mean, which dilutes it. Unknown-team slots are ignored. Wrong-team slots (~1%) feed the wrong side, and duplicate tracklets count twice.
8. **Only synthetic evidence.** Not run on real tracking, and no `--real` sanity stats exist yet.
