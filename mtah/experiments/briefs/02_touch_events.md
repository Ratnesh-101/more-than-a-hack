# 02 Touch events: possession = team of the last credible touch
**Claim.** Possession changes when the other team touches the ball, not when one of its players happens to be nearest, so the evidence comes from touches only: a kick (sharp change in ball velocity with a player in reach) or a carry (ball within 1.5 m of one team, clearly nearer that team, for 1 s). The H/A log-odds is the clipped difference "time since last A touch minus time since last H touch". It carries through gaps and loose balls, stays flat in alternating scrambles and grows once one team keeps touching.

**Football rationale.** In a scramble, an aerial duel or a press, the nearest-player identity changes many times a second while possession does not (CHALLENGE.md: a touch without control, a deflection or a won aerial is not a change). A pass that flies past a defender is not a touch: the ball's velocity does not change. A one-off contested touch gives a lead of a fraction of a second, which is far below the switch cost. About 1.5 s of continued touching by the new team is needed to flip possession. Through a ball-missing gap the touch-time difference stays constant, so the last toucher keeps the ball. When the ball reappears at the other team's feet, the unseen hand-over is put at the gap midpoint (the minimum-expected-error guess). Dead ball: the incumbent's evidence (ball out, everyone slow), plus two cues from the ball track: a ball resting for 1 s sits on the spot (dead), and a fresh kick means play is live.

| const | value | justification |
|---|---|---|
| switch_cost / dead_cost | 30 / 20 | ~1.5 s of new-team touching flips; a deflection costs 60; 20 > 30/2 stops hand-overs from being routed through a fake D |
| K | 6 fr | 0.24 s line-fit per side averages tracker jitter but keeps touches apart |
| V_MAX | 40 m/s | above the hardest shot (~35 m/s); a faster step is a tracking glitch, excluded from velocity |
| DV_KICK | 5 m/s | a deliberate pass/trap/shot changes ball velocity by more than 18 km/h; jitter and brushes do not |
| R_KICK | 2.0 m | leg reach (~1 m) from a body-centre track plus ~1 m position error |
| R_CTRL / HOLD | 1.5 m / 1 s | close control in a dribble; "keeps the ball", not "runs past it" |
| MARGIN | 0.5 m | both teams this close = contested touch (credited to both, no control) |
| TAU / CLIP | 1 s / 3 nats | 1 s of touch lead = 1 nat; one stale touch cannot outweigh 3 s of fresh evidence |
| STILL_R / STILL_BONUS | 1 m in 1 s / +3 | a ball resting on the spot; not certain, since a keeper may hold it |
| KICK_LIVE / LIVE_S | -3 / 2 s | a kick near a player proves play for about one pass flight |

**Dev results (seeds 1-5, 20 min, court output).**
| | spellF1 realistic | macroF1 r | spellF1 easy | macroF1 e | poss pred/true | flicker/min |
|---|---|---|---|---|---|---|
| **02_touch_events** | **0.5968 +/- 0.054** | 0.8348 | **0.8445** | 0.9408 | 55/66 | 1.69 |
| incumbent | 0.4955 +/- 0.044 | 0.7323 | 0.6765 | 0.8105 | 56/66 | 0.78 |

The candidate beats the incumbent on all 10 seed/difficulty pairs. Ablations (realistic spellF1): touch attribution with the incumbent's dead model unchanged scores 0.540. Removing the still cue gives 0.560, removing the kick-live cue 0.558, and removing the gap backdating 0.567. Carry-only (no kicks) gives 0.502. Kick-only (no carries, early version) gives 0.477. Plateau checks: switch_cost 20/45 gives 0.598/0.593, dead_cost 12/30 gives 0.592/0.595, TAU 2 s gives 0.598. Runtime: 0.4 s for emissions and 4.7 s with Viterbi on a 199,319-frame tiled synthetic Trk, with peak memory around 143 MB.

**Known weaknesses (honest).**
1. **Carries are close-range proximity.** The carry rule is still proximity, made exclusive and held for 1 s. Without it the score collapses to 0.477. Kicks alone cannot hold possession through a dribble.
2. **About half the gain is the dead model.** The +0.10 splits into about +0.044 from touch attribution (near the real-match noise band of 0.04) and about +0.057 from the two ball-track dead cues.
3. **Undercount.** It predicts 55 possessions against 66 true, mean 18.0 s against 14.2 s. Possessions under 4 s are rarely recovered (about 17% hit at IoU 0.5). When the ball is mostly hidden at a possession's start (59% of synthetic starts), the hit rate is 0.32 against 0.73 when it is visible.
4. **The still cue may not transfer.** It relies on a visible, motionless ball at restarts. Real tracking often loses the ball in stoppages, where the cue is silent and the incumbent's evidence takes over. A keeper holding the ball, or a very slow carry, can be falsely marked D.
5. **Kick detection assumes small jitter.** It assumes ball jitter of a few tens of cm, which is unverified on real data. Bounces and headers near a player read as kicks, because ball height is not observed. The synthetic pass reception snaps the ball onto the receiver, which may make kicks easier to see than in reality.
6. **Mis-teamed tracklets.** A wrong-team slot or a duplicate tracklet that carries the ball counts fully as the other team's control.
7. **Midpoint backdating can misfire.** If the first touch after a gap is wrongly attributed, half the gap is flipped. It is a one-frame virtual touch with no confidence weighting.
8. **Not checked on real data.** The real Trk is not available, so `--real` has not been run.
