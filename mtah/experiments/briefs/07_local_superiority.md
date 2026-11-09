# 07 Local superiority and pressure (pressure-damped nearest player)

**Claim.** The nearest-player cue is only as trustworthy as the player is free: dividing it by (1 + weighted count of opponents pressing the ball) keeps a lone player's cue at full strength and fades a contested ball (duel, scramble, tackle in progress) toward a floor, so the HMM keeps the last team through contests. On dev seeds 1-5 this ties the incumbent (+0.007 realistic, -0.008 easy spellF1, both well inside noise); the more useful result is a measured negative: raw local numerical superiority points the WRONG way.

**Football rationale.** A ball at the feet of an unmarked player is controlled; one with an opponent within tackle reach is being contested, and CHALLENGE.md says a contested ball stays with the last possessor. But the defending team closes down the ball, so near the ball the team WITHOUT it usually has more bodies. "More bodies near the ball means possession" is false. Pressure tells you how far to trust the nearest player. It does not tell you who has the ball.

| constant | value | justification |
|---|---|---|
| PRESS_RINGS inner | 0-1.5 m, w 1.0 | tackle reach: one stride plus a leg, so the challenge is live |
| PRESS_RINGS outer | 1.5-3 m, w 0.5 | closing down: arrives in about 0.5 s at sprint speed, half a threat |
| ISO_FLOOR | 0.3 | a pressed player with the ball at his feet still usually keeps it; never zero |
| team_scale_m, control_radius_m, dead model, costs | incumbent's | unchanged, so the only difference is the pressure term |
| CHUNK | 20000 | memory bound only, no effect on output |

**Dev results** (court, seeds 1-5, 20 min; mean +/- sd over seeds)

| | spellF1 realistic | macroF1 realistic | spellF1 easy | macroF1 easy | poss pred/true |
|---|---|---|---|---|---|
| 07 | 0.5020 +/- 0.051 | 0.7333 | 0.6684 | 0.8068 | 56/66 |
| incumbent | 0.4955 +/- 0.044 | 0.7323 | 0.6765 | 0.8105 | 56/66 |

Per seed, realistic: +0.031 on seed 1, +0.007 on seed 5, -0.007 on seed 3, 0 on seeds 2 and 4. Easy: -0.036 on seed 4, the rest within 0.003. **This is a tie, not a win.**

**Tried and rejected (same seeds, measured):**
- Signed own-minus-opponent density at 3/6/10 m as added evidence: 0.462-0.487 realistic.
- Kernel density over all players instead of the nearest one: 0.469.
- Boosting the cue of an isolated player (gain 1.5-2): 0.469-0.489.
- A 10-25 m "defensive block" ring: 0.466-0.478.
- Closing speed toward the ball: per-frame signal too weak (P(H) moves 0.40 to 0.65), not shipped.
- ISO_FLOOR 0.1/0.5 and a 1.5 m ring only: 0.497-0.502, all flat.

Per-frame diagnostics on dev seeds explain the outcome. Under no pressure the nearest player is right 96% of the time, but only 53% with 3 opponents within 3 m. That shows pressure carries real information. The incumbent's softmax gap, though, already captures most of it.

**Known weaknesses**
- No significant gain on synth. The whole landscape (every variant above, switch_cost 20-60) sits in 0.47-0.50, so the bottleneck is temporal (live H<->A transitions: only about 18% recovered within 4 s, and those come about 1 s early), not per-frame weighting. This candidate does not address it.
- Easy spellF1 is 0.008 below the incumbent. Damping also slows the detection of a real change when the winner is still crowded.
- The pressure count depends on team labels per slot. Wrong-team or duplicated tracklets near the ball inflate pressure. Unknown-team slots (-1) are ignored.
- The ring radii are football priors, not calibrated on real tracking. Real pressing distances and tracking noise (about 0.5 m or more in crowds) may shift them.
- The synth's pressers always chase the carrier, so the "superiority points the wrong way" finding may be exaggerated there. It still matches the football prior.
- It computes ball-to-player distances twice (once more inside incumbent.emissions for the dead model). This is wasteful but cheap: 0.34 s on 199k frames.
