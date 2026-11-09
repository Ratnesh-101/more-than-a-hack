# 01 incumbent_plus — v1, tightened where it misreads the tracking

**Claim.** Same model as v1 (nearest-player softmax + slowness + ball_out, v1 Viterbi with the 25-frame dead chain, v1 costs 30/12 untouched). Seven mechanism fixes, every constant a physical/football bound, none fitted. Small on spellF1, large on macroF1 (fewer fake stoppages), lowest-risk promotion: it can only differ from v1 where v1 is physically wrong.

**Rationale / fixes.** (1) Ball glitches: sample > 8 m off the 15-sample running median = detector jump, dropped. (2) Gaps <= 12 frames bridged if the implied speed is < 30 m/s (= features.fill_ball_gaps rule). (3) No player speed -> no slowness evidence (v1 used a 50 % "slow"). (4) Out margin 0.5 -> 1.0 m; a ball last seen out stays out <= 5 s while unseen. (5) Ball faster than 9 m/s (top sprint) is in flight: the player it flies past is not controlling it, team evidence fades to flat. (6) Slow players mean stoppage only if the ball is not covering > 3 m/s over 1 s (slow build-up with passes is live). (7) X before the first informative frame (no state-0 = H start bias); a hand-over Viterbi puts at the end of a ball gap (stay-tie rule) is moved to the gap middle.

| constant | value | justification |
|---|---|---|
| GLITCH_M / GLITCH_WIN | 8 m / +-7 samples | median follows straight flight exactly; window > 2-6 frame glitch |
| INTERP_GAP, MAX_BALL_STEP_M | 12 fr, 1.2 m/fr | < a pass, so it cannot bridge a hand-over; 30 m/s ball limit |
| OUT_MARGIN_M | 1.0 m | ball radius + ~0.5 m tracking error, x2 |
| HOLD_OUT_S | 5 s | quickest throw-in restart; after that an unseen ball may be live |
| FLIGHT_MPS / width / +-2 fr | 9 m/s / 1 / 0.16 s | top sprint speed; width = speed noise of the 4-frame difference |
| MOVE_MPS / width / +-12 fr | 3 m/s / 0.5 / ~1 s | jog pace; a ball at rest or walked to a spot never covers it |
| all other constants, costs | v1 | unchanged by design |

**Dev seeds 1-5 (court, mean +/- sd over seeds).**

| | spellF1 realistic | macroF1 real | spellF1 easy | macroF1 easy | poss pred/true (real) |
|---|---|---|---|---|---|
| incumbent | 0.4955 +/- 0.044 | 0.7323 | 0.6765 | 0.8105 | 56/66 |
| incumbent_plus | 0.4991 +/- 0.050 | 0.7688 | 0.6874 | 0.8623 | 55/66 |

Ablations (realistic/easy spellF1, measured before the 5 s hold cap): no (5)+(6) 0.5016/0.6780; (5) only 0.5040/0.7072; (6) costs ~0.005/0.02 spellF1 but gives +0.03/+0.04 macroF1 — kept because spurious mid-play D is the dominant v1 error (11k live frames called D in 3 seeds) and is worse on real slow build-ups. (1)-(4),(7) are each < 0.006 on synthetic (synthetic gaps are rarely <= 12 fr; real ones are). Runtime: 0.3 s emissions + 0.1 s decode on 199k frames.

**Known weaknesses (honest).**
- spellF1 gain on realistic (+0.004) is inside noise; paired per-seed it is better on only 2/5 (seeds 2, 5: +0.036, +0.011) and worse on 3/5 (seeds 1, 3, 4: -0.003, -0.018, -0.007). Easy: better 4/5 (seed 4 -0.093).
- Undercounts possessions on realistic (55/66): many missed possessions have the ball hidden for their whole length; ball-only evidence is flat there by construction. I did not fix "flat evidence on ball-missing frames" beyond bridging <= 0.5 s gaps.
- Hand-overs via D: with the 25-frame chain a fake stoppage costs 24 + 25 frames of dead emission, so direct switches DO occur (flicker 0.71/min vs truth 2.4) — but still far too few. Equalising costs (switch = dead = 20, a football transition-frequency argument) raised easy spellF1 to 0.751 and left realistic flat (0.496); NOT adopted because I would be choosing the magnitude from synthetic scores.
- v1 slowness constants (1.5 m/s) kept: in synthetic 24 % of live frames have mean speed < 1.5 m/s; real live play is probably similar. Gate (6) only helps when the ball is visible.
- Hold-out (4) can extend D up to 5 s into play after a very quick unseen restart; centring (7) moves switches by up to half a gap, which can be wrong if the real change was at a gap edge.
- No real-data diagnostics run (no data/cache in this checkout).
