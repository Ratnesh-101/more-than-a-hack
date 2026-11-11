# Pipeline Parameters and Design Constants

## 1. Introduction

The pipeline has no learned parameters; every constant below is a fixed design value; none was fitted to the match or to any label.

---

## 2. Constants of the Shipped Decoder

The table below lists every fixed design constant in the shipped decoder (`--variant semi_markov`, executed via `mtah.v2.run`). Team evidence is computed in `mtah/v2/touch.py`, and emissions and segment-level dynamic programming duration decoding are computed in `mtah/v2/fallback.py`.

| Name | Value | Where (file:line at HEAD) | What it does | Basis |
|---|---|---|---|---|
| `K` | `6` | `mtah/v2/touch.py:16` | Velocity line-fit half-window in frames (0.24 s) to smooth tracker jitter | design choice, not validated against labels |
| `V_MAX` | `40.0` | `mtah/v2/touch.py:17` | Ball speed ceiling in m/s (144 km/h); faster frame steps are excluded as tracking teleports | physical limit |
| `DV_KICK` | `5.0` | `mtah/v2/touch.py:18` | Velocity vector change threshold in m/s (18 km/h) to classify a deliberate kick | design choice, not validated against labels |
| `R_KICK` | `2.0` | `mtah/v2/touch.py:19` | Maximum reach distance in meters from body-center player track to ball for kick attribution | physical limit |
| `R_CTRL` | `1.5` | `mtah/v2/touch.py:20` | Proximity threshold in meters from player to ball for carry and close dribbling control | heuristic, no support found |
| `MARGIN` | `0.5` | `mtah/v2/touch.py:21` | Team distance contest margin in meters; differences within margin indicate a contested duel | design choice, not validated against labels |
| `HOLD` | `12` | `mtah/v2/touch.py:22` | Half-window in frames (12 frames each side = 25 frames ~ 1.0 s) for continuous carry requirement | heuristic, no support found |
| `TAU` | `1.0` | `mtah/v2/touch.py:23` | Time scale converting touch-lead duration in seconds to log-odds nats (1 s lead = 1 nat) | design choice, not validated against labels |
| `CLIP` | `3.0` | `mtah/v2/touch.py:24` | Maximum magnitude cap on team log-odds evidence (clipped to [-3.0, 3.0]) | heuristic, no support found |
| `BIG` | `10 ** 9` | `mtah/v2/touch.py:25` | Sentinel index for frames prior to first recorded touch | heuristic, no support found |
| `SWITCH_COST` | `30.0` | `mtah/v2/fallback.py:29` | Transition penalty in nats between H and A states in semi-Markov segment DP | design choice, not validated against labels |
| `DEAD_COST` | `20.0` | `mtah/v2/fallback.py:30` | Transition penalty in nats between live (H/A) and dead (D) states in semi-Markov segment DP | design choice, not validated against labels |
| `MIN_DEAD_FRAMES` | `25` | `mtah/v2/fallback.py:31` | Minimum stoppage duration in frames (25 frames = 1.0 s) for touch_only fallback | physical limit |
| `STILL_R` | `1.0` | `mtah/v2/fallback.py:33` | Maximum ball displacement in meters over 1 s window for resting ball detection | design choice, not validated against labels |
| `STILL_BONUS` | `3.0` | `mtah/v2/fallback.py:34` | Dead logit bonus when ball is resting (< 1 m displacement in 1 s) | design choice, not validated against labels |
| `KICK_LIVE` | `3.0` | `mtah/v2/fallback.py:35` | Dead logit penalty (live evidence bonus) following a detected kick event | design choice, not validated against labels |
| `LIVE_S` | `2.0` | `mtah/v2/fallback.py:36` | Duration window in seconds that a kick event keeps play treated as live (one pass flight) | physical limit |
| `SLOW_SPEED` | `1.5` | `mtah/v2/fallback.py:39` | Mean player speed threshold in m/s below which play is flagged as stopped | design choice, not validated against labels |
| `SLOW_WIDTH` | `0.35` | `mtah/v2/fallback.py:40` | Sigmoid transition width for mean player speed dead probability | design choice, not validated against labels |
| `DEAD_BIAS` | `-2.5` | `mtah/v2/fallback.py:41` | Prior log-odds of dead ball on a normal frame when all cues are silent | design choice, not validated against labels |
| `DEAD_OUT_BONUS` | `5.0` | `mtah/v2/fallback.py:42` | Dead logit bonus when ball is outside pitch boundary lines | football law (Law 9) |
| `DEAD_SLOW_BONUS` | `3.0` | `mtah/v2/fallback.py:43` | Dead logit bonus when mean player speed is below SLOW_SPEED | design choice, not validated against labels |
| `OUT_MARGIN_M` | `0.5` | `mtah/v2/fallback.py:44` | Ball-tracking boundary line tolerance margin in meters | measured on the match without labels (signed distance distribution) |
| `EPS` | `1e-6` | `mtah/v2/fallback.py:45` | Numerical clipping probability floor to prevent log(0) in emissions | heuristic, no support found |
| `CELL` | `5` | `mtah/v2/fallback.py:48` | DP coarse time grid cell size in frames (0.2 s at 25 fps) for semi-Markov segment DP | design choice, not validated against labels |
| `LIVE_MEDIAN_S` | `10.0` | `mtah/v2/fallback.py:49` | Geometric median possession duration in seconds for lognormal duration prior | design choice, not validated against labels |
| `LIVE_SIGMA` | `0.8` | `mtah/v2/fallback.py:50` | Scale parameter (log-std) for lognormal live possession duration prior | physical limit |
| `LIVE_MIN_S` | `2.0` | `mtah/v2/fallback.py:51` | Hard minimum duration constraint in seconds for live possession segments | design choice, not validated against labels |
| `LIVE_MAX_S` | `240.0` | `mtah/v2/fallback.py:52` | Upper bound on live possession segment duration in seconds (4 minutes) | physical limit |
| `DEAD_MEDIAN_S` | `12.0` | `mtah/v2/fallback.py:53` | Geometric median stoppage duration in seconds for lognormal duration prior | physical limit |
| `DEAD_SIGMA` | `1.0` | `mtah/v2/fallback.py:54` | Scale parameter (log-std) for lognormal stoppage duration prior | physical limit |
| `DEAD_MIN_S` | `1.0` | `mtah/v2/fallback.py:55` | Hard minimum duration constraint in seconds for stoppage segments | physical limit |
| `DEAD_MAX_S` | `600.0` | `mtah/v2/fallback.py:56` | Upper bound on stoppage segment duration in seconds (10 minutes) | physical limit |
| `DUR_WEIGHT` | `5` | `mtah/v2/fallback.py:57` | Weight on shape-only duration log-prior in semi-Markov segment DP (equal to CELL) | design choice, not validated against labels |

---

## 2.1. Constants of the Documented Alternative Decoder (`--variant champion`)

For complete transparency, the table below documents the constants of the alternative champion pipeline (`--variant champion`), which evaluates the multi-cue dead-ball model in `mtah/v2/deadball.py` decoded by the minimum-stoppage Viterbi decoder in `mtah/v2/decode.py`.

| Name | Value | Where (file:line at HEAD) | What it does | Basis |
|---|---|---|---|---|
| `OUT_MARGIN_M` | `0.5` | `mtah/v2/deadball.py:17` | Tolerance distance in meters outside pitch boundary lines for ball-out detection | measured on the match without labels (signed distance distribution) |
| `WALK_MPS` | `2.0` | `mtah/v2/deadball.py:18` | Upper speed limit in m/s (7.2 km/h) for walking pace classification | design choice, not validated against labels |
| `HSR_MPS` | `5.5` | `mtah/v2/deadball.py:19` | High-speed running speed threshold in m/s (19.8 km/h); sprinting players veto dead ball | design choice, not validated against labels |
| `SLOW_FRAC` | `0.9` | `mtah/v2/deadball.py:20` | Fraction of tracked team slots walking (90%) required to trigger stoppage cue | design choice, not validated against labels |
| `SLOW_WIDTH` | `0.03` | `mtah/v2/deadball.py:21` | Sigmoid transition scale parameter around SLOW_FRAC (~1 player transition width) | heuristic, no support found |
| `SMOOTH_S` | `1.0` | `mtah/v2/deadball.py:22` | Time window in seconds (25 frames) for centred moving average smoothing of player cues | physical limit |
| `SPEED_LAG` | `2` | `mtah/v2/deadball.py:23` | Central difference half-window in frames (0.16 s) for position-derived player speeds | physical limit |
| `BALL_WIN_S` | `1.0` | `mtah/v2/deadball.py:24` | Ball displacement measurement time window in seconds (25 frames) | heuristic, no support found |
| `BALL_STILL_MPS` | `1.5` | `mtah/v2/deadball.py:25` | Ball speed upper threshold in m/s for stationary / resting ball detection | physical limit |
| `BALL_PASS_MPS` | `8.0` | `mtah/v2/deadball.py:26` | Lower ball speed threshold in m/s (28.8 km/h) for moving ground pass in play | physical limit |
| `PERSIST_S` | `2.0` | `mtah/v2/deadball.py:27` | Required duration window in seconds (50 frames) for ball cues to persist | physical limit |
| `PERSIST_FRAC` | `0.75` | `mtah/v2/deadball.py:28` | Minimum fraction of ball sightings in persistence window exhibiting cue (75%) | heuristic, no support found |
| `SPOT_R_M` | `2.0` | `mtah/v2/deadball.py:29` | Restart spot detection radius in meters around corner arc (1 m) and pitch centre spot | football law (Law 1, Law 17) |
| `LINE_M` | `1.0` | `mtah/v2/deadball.py:30` | Distance band width in meters along touchlines for throw-in restart detection | football law (Law 15) |
| `GOAL_AREA` | `(6.0, 10.0)` | `mtah/v2/deadball.py:31` | Depth (6.0 m) and half-width (10.0 m) for goal-area restart spot detection | football law (Law 1) |
| `OWN_HALF_FRAC` | `0.95` | `mtah/v2/deadball.py:32` | Fraction of team players required in defending half (95%) for kickoff geometry cue | football law (Law 8) |
| `DEAD_BIAS` | `-2.5` | `mtah/v2/deadball.py:35` | Baseline log-odds of dead ball on a normal frame when all cues are silent | heuristic, no support found |
| `W_SLOW` | `3.0` | `mtah/v2/deadball.py:36` | Dead logit bonus when nearly all players are walking | heuristic, no support found |
| `W_HSR` | `-2.0` | `mtah/v2/deadball.py:37` | Dead logit penalty per high-speed runner (speed > 5.5 m/s), capped at 2 runners | physical limit |
| `W_STILL` | `1.5` | `mtah/v2/deadball.py:38` | Dead logit bonus when ball is motionless for 2 s | design choice, not validated against labels |
| `W_OUT` | `5.0` | `mtah/v2/deadball.py:39` | Dead logit bonus when ball is beyond boundary lines for 2 s | football law (Law 9) |
| `W_SPOT` | `2.0` | `mtah/v2/deadball.py:40` | Dead logit bonus when ball is stationary on a restart spot | design choice, not validated against labels |
| `W_KICKOFF` | `3.0` | `mtah/v2/deadball.py:41` | Dead logit bonus when kickoff geometry cue is satisfied | design choice, not validated against labels |
| `W_PASS` | `-2.0` | `mtah/v2/deadball.py:42` | Dead logit penalty when ball moves at pass speed (> 8.0 m/s) inside pitch | design choice, not validated against labels |
| `SWITCH_COST` | `30.0` | `mtah/v2/decode.py:12` | Transition penalty in nats between H and A states in Viterbi decoder | design choice, not validated against labels |
| `DEAD_COST` | `12.0` | `mtah/v2/decode.py:13` | Transition penalty in nats between live (H/A) and dead (D) states in Viterbi decoder | design choice, not validated against labels |
| `MIN_DEAD_S` | `3.0` | `mtah/v2/decode.py:14` | Minimum stoppage duration constraint (3.0 s = 75 frames at 25 fps) in Viterbi decoder | physical limit |
| `EPS` | `1e-6` | `mtah/v2/decode.py:15` | Numerical clipping probability floor for p_dead to prevent log(0) | heuristic, no support found |

---

## 2.2. Official Pipeline Constants and Documented Rules

The table below catalogs official data constants, tracking parameters, and fixed algorithmic rules from `tracking.py`, `run.py`, and decoder boundary logic.

| Rule / Parameter | Expression / Value | Where (file:line at HEAD) | What it does | Basis |
|---|---|---|---|---|
| `N_SLOTS` | `46` | `mtah/v2/tracking.py:27` | Minimum slot dimension for tracking array allocation | official document (DATA.md:15) |
| `_HALVES` | `{"1st Half": 1, "2nd Half": 2}` | `mtah/v2/tracking.py:28` | Mapping from Smart Tagging Half strings to integer half identifiers | official document (DATA.md:155) |
| `FPS` | `25` | `mtah/v2/tracking.py:29` | Video and tracking sampling rate in frames per second | official document (DATA.md:11) |
| `CACHE_VERSION` | `"v2-trk-1"` | `mtah/v2/tracking.py:31` | On-disk cache version tag to invalidate stale array caches | heuristic, no support found |
| `SPEED_MIN_SLOTS` | `10` | `mtah/v2/tracking.py:105` | Minimum tracked player slots required to include frame in speed unit check | heuristic, no support found |
| `SPEED_MEDIAN_MIN` | `0.3` | `mtah/v2/tracking.py:106` | Lower bound on median player speed in m/s to guard against pitch-normalized coordinates | heuristic, no support found |
| `SPEED_P95_RANGE` | `(2.0, 15.0)` | `mtah/v2/tracking.py:107` | Acceptable range for 95th percentile player speed in m/s | physical limit |
| ATD frame index 1-based offset | `1` | `mtah/v2/tracking.py:155` | Converts 1-based ATD frame_id to 0-based video frame index | official document (DATA.md:245) |
| Pitch dimensions scaling | `(105.3, 68.0)` | `mtah/v2/tracking.py:147, 162, 165` | Multipliers scaling 0-1 normalized coordinates to physical meters | official document (DATA.md:94-95) |
| Pre-kickoff tag parsing rule | `code == "set pieces"` and `Type == "kick off (start)"` | `mtah/v2/tracking.py:59-67` | Filter Smart Tagging XML events to locate start-of-half kickoffs | official document (DATA.md:135-137) |
| Pre-kickoff start frame calculation | `round(start_s * fps)` | `mtah/v2/tracking.py:85` | Convert Smart Tagging timestamp seconds to tactical video frame index | official document (DATA.md:164-166) |
| Midpoint backdating rule | `mid = (a + z) // 2` | `mtah/v2/touch.py:70` | Places unseen hand-over at temporal midpoint of ball-tracking occlusion gap | physical limit |
| Pre-kickoff warmup touch suppression rule | `t_h[lo:kick_frame] = False`, `t_a[lo:kick_frame] = False` | `mtah/v2/touch.py:117-118` | Zeros out touch events before kickoff to prevent warmup touches from biasing play | official document (FAQ.md:38-41) |
| Kickoff geometry rule | `has & (frac_own >= OWN_HALF_FRAC) & (~vis \| centre)` | `mtah/v2/deadball.py:115` | Cues dead ball when teams are separated by halfway line | football law (Law 8) |
| Stay tie-break rule | `(score, 1, state)` staying vs `(score, 0, state)` switching | `mtah/v2/decode.py:40-41` | Breaks path score ties towards staying in current state | football law (CHALLENGE.md:62-64) |
| Kickoff frame lookup | `(trk.get("kickoffs") or {}).get(half)` | `mtah/v2/decode.py:74-76` | Reads kickoff frame for each half from tracking dict | official document (DATA.md:164-166) |
| Pre-kickoff dead ball rule | `labels[:start - lo] = "D"` | `mtah/v2/decode.py:96-97` | Forces all frames before kickoff in graded halves to be labeled D | official document (FAQ.md:38-41) |
| Emission clamping floor | `-1e9` | `mtah/v2/fallback.py:308` | Clamps emission log-likelihoods to >= -1e9 to guard prefix sums against underflow | heuristic, no support found |
| Refinement half-window | `2` (`CELL // 2`) | `mtah/v2/fallback.py:325-327` | Boundary search radius around coarse DP cut for exact frame refinement | design choice, not validated against labels |
| Refinement tie-break rule | `cand[ties] - cut` distance minimizer | `mtah/v2/fallback.py:290-302` | Breaks emission gain ties by preferring frame closest to coarse DP cut | heuristic, no support found |
| Pre-kickoff warmup touch suppression in fallback emissions | `t_h[lo:kick_frame] = False`, `t_a[lo:kick_frame] = False`, `kick[lo:kick_frame] = False` | `mtah/v2/fallback.py:121-125` | Zeros out touches and kicks before kickoff in graded blocks | official document (FAQ.md:38-41) |
| `TEAM_RE` | `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` | `mtah/v2/run.py:27` | Regex validating submission team name | official document (SUBMISSION.md:85) |
| `WINDOWS_RESERVED` | `{"CON", "PRN", "AUX", "NUL", ...}` | `mtah/v2/run.py:28` | Set of forbidden Windows reserved device names | heuristic, no support found |
| Graded half frame ranges | `[(1, 11185, 90651), (2, 110808, 190197)]` | `mtah/v2/run.py:100` | Defines graded frame blocks where predictions must be H/A/D | official document (DATA.md:257-268) |
| Short possession duration threshold | `3.0` s | `mtah/v2/run.py:62` | Defines short possession duration cutoff for shape diagnostic output | official document (README.md:158) |

---

## 3. Real-Tracking Sensitivity Analysis (Shipped Variant)

The sensitivity table below lists the top 10 constants whose perturbation most changes the output predictions for the shipped semi-Markov variant (`fallback.decode(trk, "semi_markov")`). Sensitivity was measured in-memory on the real match tracking data (158,857 graded frames), perturbing each constant one at a time by $\times 0.5$ and $\times 2.0$, and measuring the exact share of graded frames whose label changed (`pred != base`).

| Rank | Constant / Parameter | Default Value | File Location | Change at $\times 0.5$ | Change at $\times 2.0$ | Max Impact |
|:---:|---|:---:|---|:---:|:---:|:---:|
| 1 | `fallback.SLOW_SPEED` | `1.5 m/s` | `mtah/v2/fallback.py:39` | 22.05% | 45.18% | 45.18% |
| 2 | `fallback.DEAD_BIAS` | `-2.5 nats` | `mtah/v2/fallback.py:41` | 26.10% | 25.95% | 26.10% |
| 3 | `fallback.DEAD_SLOW_BONUS` | `3.0 nats` | `mtah/v2/fallback.py:43` | 21.77% | 24.42% | 24.42% |
| 4 | `fallback.SLOW_WIDTH` | `0.35` | `mtah/v2/fallback.py:40` | 9.82% | 18.40% | 18.40% |
| 5 | `touch.HOLD` | `12 frames` | `mtah/v2/touch.py:22` | 13.75% | 13.47% | 13.75% |
| 6 | `touch.DV_KICK` | `5.0 m/s` | `mtah/v2/touch.py:18` | 9.79% | 6.39% | 9.79% |
| 7 | `touch.K` | `6 frames` | `mtah/v2/touch.py:16` | 6.39% | 9.50% | 9.50% |
| 8 | `touch.R_CTRL` | `1.5 m` | `mtah/v2/touch.py:20` | 9.47% | 6.57% | 9.47% |
| 9 | `fallback.STILL_R` | `1.0 m` | `mtah/v2/fallback.py:33` | 4.08% | 6.73% | 6.73% |
| 10 | `touch.R_KICK` | `2.0 m` | `mtah/v2/touch.py:19` | 4.50% | 5.78% | 5.78% |

The output is sensitive to the dead-ball constants (perturbing the dead-ball speed threshold changes up to 45.18% of graded frames), and no constant was tuned on any score.

---

## 4. Unlabeled Statistics for Key Constants

The constants that matter most to pipeline output can be grounded and measured directly on the match tracking data without using ground truth or labels:

1. `fallback.SLOW_SPEED`: Bimodal valley between walking and running modes in the mean player speed distribution (`analysis/unlabeled_stats.py:62-79, 106-112`).
2. `fallback.DEAD_BIAS`: Log-odds calculated from the overall match dead-ball frame fraction `log(p_dead / (1 - p_dead))` derived from unlabeled stationary ball and out-of-bounds duration (`analysis/real_checks.py:144`).
3. `fallback.DEAD_SLOW_BONUS`: Empirical log-odds ratio of stoppage given mean player speed below 1.5 m/s versus normal speed.
4. `fallback.SLOW_WIDTH`: Inverse slope of the transition region between walking and running modes in the per-frame player speed distribution.
5. `touch.HOLD`: Distribution of continuous co-movement run lengths where the ball remains within 1.5 m of a single player.
6. `touch.DV_KICK`: Valley between the noise mode and action mode in the distribution of ball velocity vector changes (`dv`) when the ball is within 2 m of any player.
7. `touch.K`: Autocorrelation decay length of raw single-frame ball position differences, or the corner frequency of the ball trajectory power spectral density.
8. `touch.R_CTRL`: Inflection point in the nearest-player distance distribution conditioned on frames where the ball velocity vector matches the player velocity vector (`analysis/unlabeled_stats.py:116`).
9. `fallback.STILL_R`: 1-second displacement distribution of stationary balls during set pieces and stoppages.
10. `touch.R_KICK`: Mode of nearest-player distances to the ball conditioned on frames with confirmed sharp ball velocity changes (`dv > 5.0 m/s`).
