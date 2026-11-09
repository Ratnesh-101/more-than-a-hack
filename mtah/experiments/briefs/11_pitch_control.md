# 11 pitch_control: time-to-intercept instead of distance

**Claim.** Team evidence comes from who can *reach* the ball first (reaction drift, bounded accel and top speed, worked in the ball's moving frame), turned into a home-control probability by Spearman's logistic in the arrival-time gap. On dev seeds 1-5 it beats v1 on macroF1 (+0.013 realistic, +0.018 easy) but NOT on spellF1 (-0.014 realistic, within noise; -0.043 easy).

**Football rationale.** Distance ignores motion: a defender 3 m away sprinting at the ball beats a stationary one at 2 m, and a carrier moving *with* the ball is closer in time than in metres. Arrival is computed relative to the ball's velocity, so a carrier dribbling with the ball, or a receiver the pass is travelling to, gets a short time. A player who already has the ball within reach reacts to nothing, so his time is just the step to it. Evidence fades to 50/50 when even the first arrival is more than about a reaction time away (loose ball: last team keeps it, through the v1 HMM). Dead-ball evidence is v1, unchanged.

| constant | value | justification |
|---|---|---|
| VEL_HALF_WINDOW | 5 frames | +/-0.2 s central difference: 0.15 m tracker jitter becomes ~0.5 m/s, not ~5 m/s |
| REACTION_S | 0.25 s | human visual reaction ~0.2-0.3 s; also the e-fold of the confidence fade |
| V_MAX | 7.5 m/s | pro sprint top speed 7-8 m/s |
| A_MAX | 3.5 m/s^2 | typical maximal footballer acceleration 3-4 m/s^2 |
| REACH_M | 1.0 m | about one stride / leg reach: playable without running |
| CONTROL_SIGMA_S | 0.45 s | Spearman 2017 control-time uncertainty (logistic scale sigma*sqrt3/pi) |
| MAX_PLAYER_SPEED | 12 m/s | above world-record speed: the finite difference is an id swap, velocity set to 0 |
| MAX_BALL_SPEED | 35 m/s | above the hardest shots: ball velocity set to 0 (position still used) |
| OUT_MARGIN_M, dead model | v1 | unchanged, to isolate the team cue |
| CHUNK | 20000 | memory bound only (no effect on results) |

**Dev results** (court, 20 min, seeds 1-5, mean +/- sd, PARAMS = {} i.e. v1 switch/dead costs):

| | spellF1 realistic | macroF1 realistic | spellF1 easy | macroF1 easy | poss pred/true (real.) |
|---|---|---|---|---|---|
| incumbent | 0.4955 +/- 0.044 | 0.7323 | 0.6765 | 0.8105 | 56/66 |
| pitch_control | 0.4819 +/- 0.037 | 0.7455 | 0.6338 | 0.8284 | 66/66 |

Per-seed realistic spellF1 cand/inc: .504/.531 .508/.498 .464/.515 .417/.410 .517/.523. Runtime: 0.6 s emissions, 156 MB peak on a 199k x 46 tiled synthetic Trk.

**Known weaknesses (honest).**
1. Does not beat the incumbent on the ranking metric; on easy it over-fragments (93 vs 66 true possessions, 15-36% under 3 s). Pitch control is *anticipatory*: a tackler sprinting in is credited before he controls the ball, while the labels follow *controlled* possession.
2. The design was revised after looking at dev seeds: ground-frame arrival (0.114), ball frame without the in-reach rule (0.336), confidence e-fold 1 s (0.405) -> tied to REACTION_S (0.482). The confidence decay is the most influential choice; I tied it to an existing prior instead of adding a free constant, but it was chosen with dev scores in view.
3. Ball-frame approximation: the speed cap applies to closing speed relative to the ball, not to ground speed; a fast pass "arrives" at its receiver early. Lateral velocity (turning cost) and ball height (aerials) are ignored.
4. Velocities are 0 for the first/last 5 frames of every tracklet and of every visible-ball run (ball treated as stationary next to gaps); a fresh tracklet looks stationary.
5. Inside 1 m everybody nearly ties (time = gap/7.5 m/s), so crowded duels give ~50/50 evidence; wrong-team slots and team -1 slots are trusted / ignored exactly as v1.
6. Sensitivity (dev seeds, realistic spell): V_MAX/A_MAX 7/3 -> 0.477, 8/4 -> 0.484; REACTION_S 0.2 -> 0.480; a doubled sigma lowers flicker but is not a literature value and was not adopted. Zeroing confidence on implausible ball speed gave 0.478 and was dropped.
