# 03 pass_chain: control-confirmation machine

**Claim.** Possession changes only when the opponent *confirms* control: two receptions in a row by its players (one completed pass) or one touch held for 0.5 s. A lone opponent touch, deflection or duel never flips it. Offline decoding lets each confirmed take-over be back-dated to the first reception of the chain (the middle of the hidden gap if the ball was invisible just before), so a late confirmation doesn't cost a late boundary.

**Football rationale.** This is the CHALLENGE.md proxy for "controlled" possession ("keeps the ball for a few seconds" / "consecutive passes"), with "contested or loose stays with the last team" built in. The current team keeps possession through ball-missing gaps and loose balls. Any own-team touch cancels a pending take-over. Touches are collective (nearest player per team), so tracklet churn and missing slot identities don't matter. D comes unchanged from the incumbent's dead-ball model. Default Viterbi, default costs.

| constant | value | justification |
|---|---|---|
| TOUCH_M | 1.5 m | ball ~0.7 m ahead of a carrier's feet + ~0.5 m tracking noise |
| MARGIN_M | 0.5 m | the nearest team must be clearly nearer; otherwise it's a duel, not control |
| MIN_TOUCH_S | 0.12 s (3 fr) | shorter proximity is a ball flying past |
| PASS_MIN_M | 5 m | shortest deliberate pass; less movement is the same player dribbling |
| CHAIN | 2 | reception + reception by a team-mate = a completed pass |
| HOLD_S | 0.5 s | first touch + settle = a controlled reception |
| TEAM_LL | 1 nat/frame | with switch_cost 30, a machine run needs ~1.2 s to survive Viterbi |
| PARAMS | {} | incumbent switch/dead costs; overriding them gave nothing |

**Dev results (seeds 1-5, 20 min; `court.run --seeds 1,2,3,4,5`).**

| | spellF1 realistic | macroF1 realistic | spellF1 easy | macroF1 easy | poss pred/true |
|---|---|---|---|---|---|
| 03_pass_chain | **0.5332 +/- 0.068** | 0.8123 | **0.7746** | 0.8872 | 60/66 |
| incumbent | 0.4955 +/- 0.044 | 0.7323 | 0.6765 | 0.8105 | 56/66 |

Paired realistic delta per seed: +0.041, +0.080, **-0.066** (seed 3), +0.044, +0.089, mean +0.038 (inside the ~0.04 noise band). Easy: better on 5/5 seeds, +0.098. Full match (199k frames, tiled synthetic): 0.4 s, peak ~117 MB.

**Ablations (final code, realistic / easy spellF1).** Shipped CHAIN=2: 0.5332 / 0.7746. **CHAIN=1 (every opponent touch of 0.12 s or more flips): 0.5322 / 0.7514. Hold-only (no chain rule): 0.5374 / 0.7809.** So on realistic data the confirmation rule is NOT what beats the incumbent. The default Viterbi switch_cost 30 already drops short flips. The gain comes from the touch-state machine itself: the last confirmed toucher carried through gaps, plus back-dating. Confirmation helps only on easy (+0.023 vs CHAIN=1), and the chain is within noise of hold-only. (With switch_cost 6, CHAIN=1 collapsed to 0.41, so the machine needs either a confirmation rule or a decoder cost.) I kept CHAIN=2 because real one-touch passing produces receptions shorter than 0.5 s, and the chain alone confirms those. That is unproven on synthetic data. Back-dating: gap midpoint 0.5332, reception frame 0.5297, gap start 0.4955. PASS_MIN_M 3 or 8 m: within 0.005. HOLD_S 1.0: 0.523. A stoppage-reset rule gained nothing and was removed.

**Known weaknesses.**
1. Ball-invisible possessions are unrecoverable. Synth has possessions (up to 40 s) with 0% ball visibility. The machine carries the old team straight through them. Most of the realistic loss is here (oracle: true D + my team = 0.45 to 0.77; my D + true team = 0.85 to 0.94).
2. Team logic is the bottleneck, and D is the incumbent's. Over-segmentation on easy (79 pred vs 66 true) comes partly from D fragments that this candidate doesn't touch.
3. Wrong-team slots: a mis-teamed player carrying the ball for 0.5 s flips possession.
4. A pressing defender within 0.5 m of the ball suppresses touches (MARGIN), so take-overs can be detected late or missed.
5. Back-dating takes over at the first opponent reception, so a tackler's first poke can start the possession up to ~1 s before the annotated control.
6. Same-team events that merge across a hidden gap (<5 m apart) count the hidden time as "held".
7. High seed variance (sd 0.068), one dev seed below incumbent. The gain is not significant on 5 seeds.
8. The headline mechanism (the pass chain) is not the source of the realistic gain (see ablations). The synthetic carries are long (seconds), so the hold rule dominates.
9. Not validated on real data (unavailable); `--real` shape stats are still pending.
