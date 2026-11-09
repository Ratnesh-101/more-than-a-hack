# 08 Multi-cue dead ball (`cand_08_dead_ball.py`)
**Claim.** A dead-ball logit built from robust player-walking statistics, persistent ball cues and restart geometry, decoded with an exact "every stoppage lasts >= 3 s" Viterbi, finds every true stoppage on dev seeds with few false ones (97 predicted vs 92 true D runs, 5 false; v1 predicts 206 for the same 92). That alone takes spellF1 from 0.496 to 0.530 realistic and 0.677 to 0.758 easy. H/A is v1's cue, unchanged.

**Football rationale.** In a stoppage almost everyone walks, but one restarter or a ghost slot can drag a MEAN speed up, so I use the share of known-team slots below 2 m/s plus a count of high-speed runners (>5.5 m/s means live play). Ball cues only count when they PERSIST for 2 s: a lofted ball projects beyond the lines, and a shielded ball pauses briefly in live play. The kickoff law (all players in their own half) still works when the ball is missing. Real stoppages last >= ~3 s, so a D run shorter than that is an artefact and would split a possession in two (semi-Markov D chain, checked against brute force on 300 random cases).

| constant | value | why |
|---|---|---|
| WALK_MPS | 2.0 m/s | top of walking pace (~7 km/h) |
| SLOW_FRAC / WIDTH | 0.9 / 0.03 | stoppage = at most ~2 of 20 outfielders above walking; ~1 player of softness |
| HSR_MPS, W_HSR | 5.5 m/s, -2 per runner (cap 2) | standard high-speed-running band; sprinting means live |
| SMOOTH_S | 1 s | a stoppage is a state, not a frame |
| BALL_STILL / PASS | 1.5 / 8 m/s | slower than walking / slowest ground pass |
| PERSIST_S / FRAC | 2 s / 75 % | longer than a lofted ball's spell beyond the line; tolerates glitches |
| SPOT_R, LINE, GOAL_AREA | 2 m, 1 m, 6x10 m | corner arc / centre spot, touchline, goal area (5.5x18.32 m) plus tracking error |
| OWN_HALF_FRAC | 0.95 | kickoff law, allowing one mis-tracked slot |
| MIN_DEAD_S | 3 s | fastest real restarts (quick throw-in / free kick) |
| W_SLOW, W_OUT, bias, OUT_MARGIN | 3, 5, -2.5, 0.5 m | v1's values, kept (not re-tuned) |
| W_STILL, W_SPOT, W_KICKOFF, W_PASS | 1.5, 2, 3, -2 | each is a moderate log-odds cue that cannot outvote the player cue alone |
| switch/dead cost | 30 / 12 | v1 defaults, untouched |

**Dev results (seeds 1-5, 20 min, mean +/- sd over seeds).** Realistic spellF1 **0.5301 +/- 0.057** vs incumbent 0.4955 +/- 0.044 (paired +0.035, ahead on 4/5 seeds, tie on seed 2). Realistic macroF1 0.836 vs 0.732. Easy spellF1 **0.7575** vs 0.6765 (ahead on 5/5), easy macroF1 0.922 vs 0.811. With the true D given to the decoder (diagnostic only, not in the candidate): 0.5485 / 0.7757. My D comes within ~0.02 of a perfect D, so the rest of the gap is H/A. Runtime on a 199,319-frame x 46-slot tiled synth: emissions 0.4 s, decode 0.1-4 s, peak ~200 MB. Deterministic, trk not mutated.

**Known weaknesses (honest).**
- On synth, almost all of the gain comes from the walking-share cue (ablated: 0.398) and the min-duration decoder. With persistence on, the four ball cues together add only +0.003 realistic and cost -0.007 easy, which is noise. Without persistence they HURT (0.503): the synthetic live ball sits up to ~1 m beyond the lines. I keep them because they move the D start earlier and hedge real stoppages where players do NOT walk (goal celebrations, players jogging into position for corners).
- The kickoff cue never fires on synth (its kickoffs ignore the own-half law), so it is completely unvalidated. It may also misfire in real live play if both teams' lines sit exactly on halfway.
- Sensitive to the walking threshold: WALK_MPS 1.5 gives 0.456 and 2.5 gives 0.521. On real ATD, speed noise or a different speed definition could shift the walking share. SLOW_FRAC 0.85/0.95 gives 0.512/0.517.
- Persistence delays the detected D start: median 0.56 s late (end 0.12 s), which costs IoU on short possessions.
- MIN_DEAD_S = 5 scores higher (0.537), but I kept 3 s on football grounds: quick restarts would otherwise be swallowed. A real D shorter than 3 s is always missed.
- Realistic possession count is low (48 vs 66 true; mean 20 s vs 14 s). v1's H/A with switch_cost 30 merges hand-overs, so this candidate should be paired with a better H/A champion.
- Real players jogging during stoppages, more than 2 at a time, will make the slow cue miss them. It is untested on real data (none available); with --real only the possession-shape stats can be checked.
- The first decoded frame may be D with no minimum length (a stoppage can span the block start).
