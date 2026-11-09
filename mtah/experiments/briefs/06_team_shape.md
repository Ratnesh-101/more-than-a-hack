# 06 Team shape (advocate 06)

**Claim.** When the ball is missing or loose, the two teams' collective movement says who has it: the joint block drifts toward the defending team's goal, and the attacking team is longer front-to-back. A small, ball-gated H/A log-odds term built from this mostly *moves* hand-overs hidden in ball gaps to the right frame. It invents almost no new possessions (56/66 predicted, same as the incumbent).

**Football rationale.** The team in possession progresses the ball and pushes its lines up. The team out of possession retreats toward its own goal and compacts. So 20 outfield players move together toward the goal of the team that does not have the ball. With flat evidence in a gap the Viterbi path switches at the gap's end (ties stay). A sign-changing shape cue switches where the block turned round instead.

**Formula** (per half; no labels). Home's attack direction is the sign of the median of (away centroid x minus home centroid x). The advance cue `adv` is the mean per-slot x velocity of all team-known players, using only slots tracked on consecutive frames so churn cannot fake motion. It is clipped at 10 m/s, put through a centred 4 s mean and signed by home's direction. The stretch cue `stretch` is the 4 s mean of home's x-std minus away's x-std. The score is `(2 tanh(adv/1) + 1 tanh(stretch/3)) / 3 * 3 nats / 100 frames`, weighted by (1 - ball control confidence) and added as +/- half to the incumbent's H/A rows. Everything else (dead-ball model, decoder, costs) is the incumbent's.

| constant | value | justification |
|---|---|---|
| WINDOW_S | 4 s | a block needs ~3-6 s to turn round after a turnover |
| V_MAX | 10 m/s | above elite sprint speed, so a bigger step is an ID switch |
| V_REF | 1 m/s | ~10 m of progression per 10 s possession counts as a clear advance |
| DEPTH_REF | 3 m | team length in possession vs out of it differs by a few metres |
| W_ADV : W_DEPTH | 2 : 1 | advance is the primary, better-established cue |
| NATS_PER_WINDOW | 3 | a confident cue is about a 95/5 call per *independent* 4 s window, spread over its 100 frames so correlated frames are not counted 100 times |

**Dev results (seeds 1-5, 20 min, court output).**

| | spellF1 realistic | macroF1 realistic | spellF1 easy | macroF1 easy | poss | s/run |
|---|---|---|---|---|---|---|
| team_shape | 0.5129 +/- 0.057 | 0.7479 | 0.6856 | 0.8119 | 56/66 | 0.2 |
| incumbent | 0.4955 +/- 0.044 | 0.7323 | 0.6765 | 0.8105 | 56/66 | 0.1 |

Per seed (realistic): +0.054, +0.025, 0.000, 0.000, +0.007. It never loses, but the mean gain (+0.017) is under the 0.04 noise band. On a 199k-frame tiled synthetic Trk, emissions take 0.7 s.

**Sensitivity (disclosed; not used to pick values).** Realistic spellF1 by NATS_PER_WINDOW: 0 gives 0.4955, 1 gives 0.510, 3 gives 0.513, 10 gives 0.534, 30 gives 0.552. I did NOT take 30. The synthetic formations are pure ball-followers (`base + pull*(ball-centre)`), so block advance is far cleaner there than in real football, and a weight that strong would let shape overrule the switch cost on its own. Other results: W_DEPTH=0 gives 0.522, W_ADV=0 gives 0.512, WINDOW 3 s / 6 s give 0.519 / 0.510, V_REF 0.5 / 2 give 0.521 / 0.518.

**Known weaknesses.**
1. The synthetic world builds in the advance cue (formations follow the ball), so dev gains overstate real value. The depth/stretch cue is barely modelled in synth (only 2 forward runners), and removing it scores slightly *better* on synth. I keep it on football grounds only.
2. Real football breaks the advance cue: recycling the ball backward, build-up from the keeper, a low-block team keeping the ball deep, and counter-pressing all move the block against the possession sign.
3. The centred window uses up to 2 s of future frames (offline, so legal) and lags/smears the true turn-round by about 1-2 s. That is enough to cost IoU at 0.6-0.7 thresholds.
4. Attack direction comes from a per-half median of centroids. It fails if a half is dominated by one team camped in the other's half, or if team labels in slots are badly wrong. Mis-team slots (about 1%) and duplicates bias both cues slightly.
5. It cannot detect a hand-over by itself (its weight is below switch_cost by design). On seeds 3 and 4 it adds nothing; it only helps where ball evidence on both sides of a gap already disagrees.
6. The ball-gating confidence uses the incumbent's 3 m control radius, so a ball glitch near a player suppresses the shape term on those frames.
7. Not validated on real data (none available); the real-match sign of the depth cue is an assumption.
