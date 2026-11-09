# 09 territory: attack-direction and territory priors, self-calibrated

**Claim.** The incumbent plus a weak H/A prior built from two territory cues: *advance* (how fast the whole field drifts toward one goal, signed by the direction home attacks this half) and *zone* (how deep the ball, or the player centroid when the ball is lost, sits in home's attacking half). The cue weights come from a ridge-logistic fit, per half, against the match's own confident nearest-player frames (algorithmic pseudo-labels, no human ones), so no weight is set by hand, and a cue that carries no information in a match gets a weight near 0.

**Football rationale.** Each team's block sits in its own half, so the median x-gap between the two teams' players gives the attack direction for each half without labels. The team with the ball pushes play toward the other team's goal, and the defending team drops back the same way, so the sign of the field's x-velocity points at the team in possession. This works even with the ball missing (41% of frames) and during scrambles, and it flips sign at turnovers. Zone carries the "deep in your own third means defenders or a clearance" prior. The fit learns its sign from the data instead of me assuming one.

| constant | value | justification |
|---|---|---|
| HALF_WIN_S | 0.5 s | a +/-0.5 s centred displacement is one stride cycle, longer than tracking jitter |
| MAX_SPEED_MPS | 10 | elite sprint ceiling; faster per-slot motion is an identity jump inside the slot |
| ADV_SCALE_MPS | 1.5 | a field drift of ~1.5 m/s is a clear attacking push (jog is 2-3 m/s) |
| SURE_DIST_M / SURE_GAP_M | 1.5 / 3.0 m | pseudo-label only when a player is within playing distance and there is no duel |
| MIN_SURE | 500 fr | need at least 20 s of confident frames per half to fit two slopes, else no prior |
| RIDGE | 1.0 | unit Gaussian prior on the slopes, so a small fit cannot explode |
| NEWTON_ITERS | 12 | Newton converges in under 10 steps on a 3-parameter logistic |
| CORR_FRAMES | 25 | cues vary on a ~1 s scale, so one frame carries 1/25 of a second's evidence |
| CAP_LOGIT | 2.0 | a territory cue alone never claims more than ~88/12 per second |
Decoder, switch_cost and dead_cost are the incumbent's (PARAMS = {}).

**Dev results (seeds 1-5, 20 min).** Realistic spellF1 is **0.5419 +/- 0.047**, against 0.4955 +/- 0.044 for the incumbent (+0.046, better on 4/5 seeds; seed 3 is -0.003). Realistic macroF1 is 0.7650 vs 0.7323. Easy spellF1 is 0.6931 vs 0.6765. The predicted possession count is 57 (truth 66), the same as the incumbent's 56, so the gain comes from better hand-over placement and side choice, not from more possessions. Ablations on realistic: advance only scores 0.537 and zone only 0.506, so almost all of the gain is the advance cue. Emissions on a 199k-frame tiled synthetic Trk take 0.56 s (peak ~116 MB); emissions plus decode take 0.7 s.

**Known weaknesses.**
1. The synthetic generator makes advance unrealistically clean (carriers always run forward at 2.5-5 m/s; P(H | home advancing) is about 0.85). In real football, possession often goes backwards (recycling, goalkeeper build-up), so the real gain will be smaller. The self-fit should shrink the weight, but this is unproven on real data.
2. A stronger prior scores higher on synthetic data (CORR_FRAMES=12.5 gives 0.573). I did NOT take it, because it would over-trust that clean signal. My constant is the principled one, not the best-scoring one.
3. Zone is weak and unstable: its fitted slope changes sign between halves and seeds (pseudo-labels near goal are biased toward defenders clearing). It adds only ~+0.005, which is within noise.
4. The pseudo-labels inherit nearest-player bias: confident frames are ball-visible open play, not the scrambles where the prior is applied. It is a transfer assumption.
5. Attack direction is one sign per graded block. It assumes teams switch ends only at half time. Extra time or a block spanning a switch of ends would break it. There is no goalkeeper-slot fallback, because the median team separation was enough.
6. The prior is applied at the same weight whether or not the ball is visible. Weighting it toward ball-missing frames (the brief's "tie-breaker" idea) did not help on dev seeds (0.536-0.537), so I left it out.
7. It does nothing about extra short possessions caused by the nearest-player cue in duels. The count stays at the incumbent's level, below truth.
