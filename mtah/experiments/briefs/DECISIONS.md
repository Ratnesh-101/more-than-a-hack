# DECISIONS (decider) - constants marked `UNJUSTIFIED - ask lead`, and the half-head behaviour
Rules: no synthetic score, public GT or hand label is used as evidence below. Each decision rests on a law of the game, a physical-performance norm or a mechanism argument, and names the unlabeled statistic that confirms or kills it, plus the action registered before any real run. Numbers marked (approx) come from general football-physics knowledge, not from a source I can cite exactly. **All six decisions are KEEP: no code change, golden tests unaffected.**

Notation: 25 fps, so MIN_DEAD_S 3 s = 75 frames. Per-frame D-vs-live advantage = dead logit + |log sigmoid(+-r)|, which is logit + 0.05 when the team evidence is confident (|r| = 3) and logit + 0.69 when r = 0.

## The cue lattice (shared by items 2-4)
The nats are not calibrated probabilities. Frames are strongly correlated, so a sum over 75 frames overcounts. What matters is the sign and ordering of the combinations, and each of these is a football statement:
| situation (smoothed cues) | logit | p | football statement |
|---|---|---|---|
| no cue | -2.5 | 0.08 | live play is the default; a 3 s D with no cue costs 136-184 nats, so it never happens |
| >=90 % walking only | +0.5 | 0.62 | everyone walking leans dead, but weakly |
| walking + 1 sustained sprinter, or + ball at pass speed | -1.5 | 0.18 | someone running above 19.8 km/h, or a ball moving >8 m/s, means play is on |
| ball out (persisted 2 s) only | +2.5 | 0.92 | Law 9: a ball wholly over a line is out of play; this cue decides on its own |
| ball out + 2 sustained sprinters | -1.5 | 0.18 | just-crossed ball with chasers still at full speed: wait until they slow down |
| ball out + walking / kickoff + walking | +5.5 / +3.5 | >0.97 | unambiguous stoppage |
Constraints the values satisfy: W_OUT > |bias| (out decides alone); 0 < bias + W_SLOW < |W_HSR| (walking leans dead, and one runner vetoes it); bias + W_OUT + 2 W_HSR < 0. Any rescaling that keeps these signs gives the same decisions on clear frames. The exact magnitudes only matter on borderline frames, and I cannot support them beyond that.

## 1. deadball.SPEED_LAG = 2 frames: KEEP
- Why: it is a 0.16 s central difference, close to the time resolution of the tracker speed it replaces. It is only used where the ATD speed is NaN, and DATA.md documents speed per slot in m/s, so fallback use should be rare. Noise cost: per-axis speed error is about 1.41 x sigma_pos / 0.16 s. For white position noise of 0.15 m that is about 1.3 m/s, enough to push a standing player over WALK_MPS about 30 % of the time. Real tracker noise is usually temporally correlated (approx), which makes the error much smaller.
- Confirm: no row exists yet. The lead should add one: the share of (slot, frame) with ps NaN and px finite, and on frames where both exist, the p90 of |position-speed - ps| for ps < 0.5 m/s. Related row: real_checks "frames with walking share >= 0.9" (it uses the same SPEED_LAG).
- Pre-registered: if the fallback share is < 1 %, ignore. If it is >= 1 % and that p90 is > 1.0 m/s, the lead approves SPEED_LAG = 12 (1 s, the same as SMOOTH_S, so the walking cue loses no time resolution). That code change breaks golden equality with the experiment, so the golden test must be re-baselined explicitly.

## 2. deadball.DEAD_BIAS = -2.5: KEEP (relabel it: it is not a "prior")
- Why: it is not a base rate. The ball is out of play for roughly a third or more of match time (approx: effective playing time is about 55-60 of ~95 min). EVALUATION.md also calls D "a large share of all frames". -2.5 is the log-odds when every cue is silent. It makes D something the cues must earn, which the 3 s floor needs: a no-cue fake D costs over 130 nats. Inheriting it from v1 is not a reason, because v1 used it with a mean-speed cue. The lattice above is the reason.
- Confirm: real_checks "D share" (INFO; football expectation about 0.3-0.45, approx) and "D share - compare D share" against v1 (gate +-0.1).
- Pre-registered: if D share < 0.20 or > 0.55 in either half, do NOT retune the bias. Read the walking-share rows (item 3). If those fail, ship `--variant touch_only` (VERDICT section 4). Otherwise the lead decides, with no public-GT input.

## 3. deadball.W_SLOW = 3.0 / W_OUT = 5.0: KEEP
- Why: see the lattice. W_SLOW: in a stoppage nearly everyone walks (<7 km/h, the standard walking band); in live play a full 3 s with >=90 % walking, no sustained sprinter and no pass-speed ball is uncommon. W_OUT: Law 9; a ball beyond the line for 2 s is the most direct physical evidence of a stoppage, so it alone must give p > 0.9.
- Confirm: real_checks "frames with walking share >= 0.9", "walking-share bimodal", "walking share >= 0.9 labelled D"; unlabeled_stats a) and d) (share of visible-ball frames more than 0.5 m outside the lines).
- Pre-registered: walking share >= 0.9 on < 0.10 of graded frames (well below the approx one-third stoppage time), or no valley -> walking premise failed -> ship touch_only (VERDICT 4). unlabeled_stats d) "out > 0.5" above 10 % of visible frames -> check pitch scaling or lofted-ball projection before trusting W_OUT. That is a lead decision.

## 4. deadball.W_HSR = -2.0 per runner, cap 2: KEEP
- Why: 19.8 km/h (5.5 m/s) is the standard high-speed-running band (approx, widely used in physical reports). In a stoppage nobody needs to run that fast for a whole second. n_hsr is smoothed over 1 s, so a 2-frame speed spike from an ID swap counts only about 0.08 of a runner. -2 is the smallest round weight that makes one sustained runner veto walking-only (+0.5 -> -1.5). The cap stops duplicate or ghost slots from burying a real ball-out (it floors out-only at -1.5).
- Known cost: goal celebrations and sprints to fetch the ball for a quick restart are dead but fast, so the D start comes late.
- Confirm: real_checks "walking share >= 0.9 labelled D". It should be high, because walking-dominated frames are mostly stoppages.
- Pre-registered: if it is < 0.5, the lead inspects n_hsr on those frames (unlabeled). A change to a 1-runner cap or to W_HSR is a code change that needs lead approval. It is never chosen by score.

## 5. decode.DEAD_COST = 12: KEEP; the 24 < 30 route is intended
- Why: a stoppage costs 24 for the round trip. A genuine stoppage of 3 s or more pays that easily (walking-only: 75 x 0.55 = 41 > 24). Through a stoppage, H->D->A costs 24 against 30 for a direct switch, and that is football-correct. Throw-ins, goal kicks and corners go to the opponents of the last toucher (Laws 15-17), and free kicks to the offended team (Law 13), so team changes cluster at stoppages.
- Brief 02's reason for 20 (">30/2 stops routing hand-overs through a fake D") applied to a decoder without the 3 s floor. With the floor, a D needs 75 frames whose average logit is above about -0.77 at r about 0, i.e. the dead model must already say dead-ish. A no-cue fake D costs over 130 nats.
- Residual risk: a hand-over during a slow, walking-heavy live phase can be routed through a short false D, which shaves about 1.5 s off each neighbouring possession. It creates no extra possession.
- Confirm: real_checks "flicker H<->A /min", "D share", "share < 3 s". The lead should add a D-run count per half: football expects roughly 50-65 restarts per half (approx, from ~100-130 per match), and a count far above that means fake D.
- Pre-registered: D runs per half > 1.5 x 65 -> lead review of W_SLOW/DEAD_COST as an approved code change. Never via the public GT.

## 6. Behaviour: H on ties at the head of a half: KEEP H; do not write X
- When ties actually happen: H and A path scores are equal only until the half's first one-team touch. After that r equals the gap between the two teams' last touches, and the path history breaks later emission ties, even after a contested touch. If the first touch arrives before any D, Viterbi already labels the head with that team's letter (no switch cost), the same result X would give. H-by-default bites only (a) on live frames before the first touch that are followed by a D, and (b) on a half with no touch at all.
- X is worse, not neutral: the scorer fills head X with the first non-X label of the half. In case (a) that label is D, so the head possession is a guaranteed miss. A team guess that is right with probability q adds a match in expectation; it beats omission when q > F1/2 (from F1 = 2TP/(G+P)), i.e. q > 0.3 at F1 0.6. A coin flip qualifies. The effect is about 1 possession per half out of roughly 100+ (approx), below the 0.04 noise band.
- Football context: half 1 is graded from the kickoff (block 11185; games.py anchor strike at 11186), so the kickoff team owns the first possession. The kickoff + walking cue may give a short D at the head (the first frame is exempt from the floor), which costs about 1 s of IoU on that possession. Half 2: FAQ.md says frames 110,808-113,741 (about 117 s) are graded and all D before the kickoff. The pipeline must find that D through kickoff geometry and walking. If no players are tracked there, `has` is False, slow is 0 and the logit is -2.5, so those frames go live/H. That would merge with the first possession.
- **Gate bug:** real_checks "D run at half start s" FAILs above 60 s. A correct half-2 output has about 117 s, so read that FAIL as expected for half 2. The lead should amend the gate. Do not hard-code D from FAQ.md: that would use organiser label information as an input. Use it as a check only.
- Better fix (recommend, not implement): Law 8 says opponents stay outside the centre circle at kickoff, so the team with a player at the centre spot is the kicking team, and the other team kicks off half 2. Seed the head with that team, and check that the two halves alternate.
- Risk if touch detection fails broadly: whole halves become H/D, and spellF1 in those halves falls toward 0 whether we write H or X. Confirm: real_checks "visible frames in runs >= 13" (FAIL < 0.3) and "larger team share of live frames" (WARN > 0.8). Pre-registered: both bad in a half -> the touch evidence is unusable -> the lead considers the v1 incumbent (mtah/decode.py). I have not checked that it still runs end-to-end.

## Other justifications weaker than TECHNICAL.md claims
1. SWITCH_COST/TAU "a deflection costs 60": the evidence for a deflecting team is 25 x (gap between the two teams' last touches, s) x (loose time, s) nats, not a fixed amount. A header 1.5 s after a long H pass, followed by a 2 s loose ball, gives about 75 > 60, i.e. a false A spell. Check: "share < 3 s".
2. touch.DV_KICK/K: for white ball noise sigma, dv noise is about 6.7 sigma per axis, giving P(dv > 5) of about exp(-0.28/sigma^2): about 0.1 % at 0.2 m, 5 % at 0.3 m, 33 % at 0.5 m. The 5 m/s threshold holds only if sigma <= ~0.3 m. Check: real_checks "ball step m median / p95".
3. touch.R_CTRL "ball rarely > 1.5 m from a dribbler": unsupported. At speed, players knock the ball several metres ahead (approx), and the only evidence is the synthetic 0.7 m. Check: unlabeled_stats b).
4. Section 2, restart spots "are where restarts are taken": free kicks (Law 13, anywhere) and penalties (11 m spot) are not covered.
5. Section 2 and the deadball docstring "one sprinter cannot flip it": one SUSTAINED sprinter does flip walking-only (lattice). Only spikes are absorbed.
6. Section 2, kickoff geometry "happens only at a kickoff": a deep block with both lines near halfway and the ball missing also fires it (brief 08 admits this).
7. SLOW_FRAC "2 of 20 outfielders": the share is over all known-team slots, including keepers and duplicates (mean 20.5 tracked, DATA.md).
8. MIN_DEAD_S "fastest restarts take about 3 s": no measured source; quick free kicks can be faster (approx), and the GT's D-onset convention is unknown.

## What I could not support
The exact effective-playing-time share, walking time-share, restart counts and shot speeds (all approx); whether real tracker noise is white or correlated; the GT's D convention (line crossing vs whistle); absolute nat magnitudes (only their ordering is argued); the Law 8 kickoff-team seed (a mechanism, untested); whether v1 still runs end-to-end.

## First real-data check
`python -m analysis.real_checks --csv submissions/nirakshak.csv`. Read "frames with walking share >= 0.9", "walking-share bimodal" and "D share" together; decisions 2-5 all rest on them. Read the half-2 "D run at half start" FAIL as expected (about 117 s).
