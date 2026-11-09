# VERDICT of the Court (judge) - nirakshak, Phase 1
Sources: `judge/audit.py` (tiled 199,319-frame Trk + DEV-seed probes), `judge/trial.py` (HIDDEN seeds 101-110, 20 min, realistic + easy, one process, decision rule pre-registered in its docstring, nothing tuned on them). Full output: `judge/out/trial.md`.

## 1. Rules audit (all 12 + 4 judge stacks)
| check | result |
|---|---|
| `--rules-lint`, GT/evaluation imports, data files, absolute paths | all OK; imports are numpy, `mtah.decode.PARAMS`/`viterbi`, `court.incumbent` only |
| determinism (emissions bit-exact x2, labels x2), NaN / +inf, trk mutation | all pass, 0 NaN, 0 +inf, trk hash unchanged |
| runtime, 199k frames | emissions <= 0.9 s for all; decode <= 0.8 s except 12 / stack_1 (5.6-5.8 s) |
| scope | 12 cand + 12 briefs only; court/*.py hashes unchanged during judging, mtimes predate all candidates. Note: `mtah/decode.py` mtime (18:44) is after the court (18:39), the likely cause of the drifting incumbent row |
| STRUCK | **none**. A constant gets struck only if it is generator-derived AND the score depends on it (DEV probes, seeds 1-3) |
| FLAG 01 | GLITCH_WIN sized to synth's 2-6 frame glitches; turning the filter off costs 0.003, so it is not an exploit |
| FLAG 08 | HSR_MPS 5.5 = synth chaser speed (5.0/6.0 give 0.535/0.532 vs 0.537, flat). WALK_MPS 2.0 sits just above synth's 1.6 m/s dead walk: 1.8 gives -0.020, 1.5 gives -0.07 (brief). A cliff, see risks |
| FLAG 12 | LIVE_MEDIAN_S 10 = synth's stated median; 6/15 give -0.021/-0.004. Mild |
| FLAG 02 | STILL_R peaks sharply (0.5/1/2 m: 0.563/0.601/0.527), fitted to synth's 0.2 m ball noise + perfectly still dead ball |
| FLAG 03, 05, 10 | 03 cites synth's 0.7 m carry offset (TOUCH_M flat 1.2-2 m); 05 calibrated v/tau on dev gap errors; 10 NATS_PER_S picked on dev |
| DISCLOSE 09 | ridge-logistic weights fitted per half on pseudo-labels from the match's own nearest-player frames. Legal (algorithmic, no human, no GT); risk = pseudo-label transfer to scrambles |

## 2. Hidden seeds 101-110 (realistic; d = paired vs incumbent, 95% bootstrap CI of the mean)
| entry | spellF1 | macroF1 | poss/true | <3s | d spell [CI] | won |
|---|---|---|---|---|---|---|
| **stack_2 02 team + 08 dead/decoder** | **0.648+/-.063** | **0.867** | 52.9/66.8 | .063 | +0.184 [+.158,+.209] | 10/10 |
| stack_1 02 + 12 decoder | 0.640+/-.066 | 0.850 | 54.2/66.8 | .052 | +0.176 [+.143,+.210] | 10/10 |
| stack_3 02 + 10 delta | 0.638+/-.063 | 0.852 | 56.0/66.8 | .069 | +0.174 [+.139,+.206] | 10/10 |
| stack_4 02 + 05 delta | 0.631+/-.058 | 0.848 | 55.2/66.8 | .069 | +0.167 [+.134,+.198] | 10/10 |
| 02 touch events | 0.625+/-.057 | 0.847 | 55.3/66.8 | .067 | +0.161 [+.128,+.195] | 10/10 |
| 03 pass chain / 10 movement / 09 territory | 0.588 / 0.551 / 0.537 | .818/.757/.770 | 60/63/56 | | +.124 / +.087 / +.073 (all CI>0) | 10/10 |
| 06 / 04 / 05 | 0.511 / 0.509 / 0.499 | | | | +.047 / +.045 / +.035 (CI>0) | 10,8,8 |
| 08 / 11 / 12 / 01 (alone) | 0.484 / 0.479 / 0.479 / 0.473 | .837/.752/.741/.773 | 44.5/65/53/51 | | CI includes 0 | 6,6,7,7 |
| incumbent / 07 | 0.464 / 0.460 | .740/.737 | 53.5/53.0 | .104 | 07: -0.005 (CI<0) | 1/10 |
Stacks vs 02 (realistic): stack_2 +0.024 [+.011,+.036] 9/10; stack_1 +0.016 [+.008,+.024] 10/10; stack_3 +0.014 [+.003,+.023]; stack_4 +0.007 [-.002,+.015] (not supported). Easy: stack_2 0.901 (+0.052 vs 02), 68.7/66.8 possessions.
Over-claims: 08's dev +0.035 is not significant on hidden seeds and it undercounts badly (44.5/66.8); 12 alone is not significant either. Both only pay off on top of 02's team evidence.

## 3. Ruling
- **CHAMPION: stack_2 (STACK)** = 02 team log-odds + 08 dead_logit + 08 min-3-s-stoppage Viterbi. It is best on realistic and easy, and it is the only stack with a CI above 0 against its base on both difficulties. It also DROPS 02's fragile still-ball/kick dead cues.
- **RUNNER-UP: stack_1 (STACK)** = 02 emissions + 12 semi-Markov decoder (costs 30/20). It improves on a different axis (decoder, not dead model). If the 5.7 s decode or the extra code is unwanted, fall back to **02 single**, which is supported on its own.

## 4. Cross-examination: how each top contender fails on REAL tracking
- **stack_2**: real stoppage players jog (corners, celebrations) or the ATD speed definition differs, so the walking share never reaches 0.9 and D is missed. WALK_MPS has a cliff (-0.02 at 1.8). The undercount (53/67) gets worse. Check `unlabeled_stats` a): the mean_speed valley/modes and the per-frame share of slots < 2 m/s must show a clear high mode. Fallback: 02's own dead model.
- **stack_1**: kick detection (DV_KICK 5 m/s over 0.24 s) depends on ball jitter. Real 0.5-1 m jitter or bounces create spurious kicks, and the 2-s live minimum then forbids the real short turnovers. Check share < 3 s and flicker on `--real`, and ball step-noise from b)/g). Fallback: 02 + v1 Viterbi.
- **stack_3**: the closing-speed sign inverts when real defenders retreat into a low block or both teams sprint on a counter (synth chasers always converge). Check whether flicker rises and possession count jumps vs 02 on real data. The sign agreement of movement_score with confident proximity frames (tracking-only AUC, no GT) must be > 0.6. Otherwise drop the delta.
- **02 single**: the still-ball cue (STILL_R peaked at 1 m) needs a visible motionless ball at restarts. Real tracking often drops the ball in stoppages (`unlabeled_stats` e) missing runs; g) stationary runs). Few stationary runs means a silent cue. Real jitter > 1 m/s means it never fires. The synthetic snap of the ball onto the receiver flatters the kick detection.

## 5. PROMOTION SPEC for Kiro (port into mtah/; do not change constants)
1. Team evidence: port `cand_02_touch_events._touches`, `_backdate`, `_since` and the r = clip((since_A - since_H)/(TAU*fps), +-CLIP) part of `emissions` (K 6, V_MAX 40, DV_KICK 5, R_KICK 2.0, R_CTRL 1.5, MARGIN 0.5, HOLD 12, TAU 1, CLIP 3; physical bounds per brief 02). Team log-probs = log sigmoid(+-r). Do NOT port STILL_BONUS/KICK_LIVE.
2. Dead evidence: port `cand_08_dead_ball.dead_logit` and `_speeds` verbatim (WALK 2.0, HSR 5.5, SLOW_FRAC 0.9/0.03, persistence 2 s/75%, spot geometry from the Laws, W_* as filed, v1 bias -2.5). Emissions = [team_h + log(1-p), team_a + log(1-p), log p], with p clipped to [1e-6, 1-1e-6] (exactly `stack_2_touch_deadball.emissions`).
3. Decoder: `cand_08_dead_ball._viterbi_min_dead`, MIN_DEAD_S 3, costs 30/12 (v1).
4. Real-data acceptance, UNLABELED only (court `--real` stats / analysis/unlabeled_stats.py; never GT, never the public 5 min for choosing). Possessions per half are within 0.7-1.3x of 02-alone's count and not below the incumbent's. Mean duration 8-25 s. Share < 3 s <= 0.10. Flicker 0.5-3/min. D share is close to the v1 D share (+-0.1). No half starts > 60 s in D. If the walking-share check in section 4 fails, ship 02 single (runner-up fallback) and re-run the same tests.
5. Tests: golden-file equality of the ported emissions/labels against `stack_2_touch_deadball` on synth seeds 1-3, plus determinism x2, no NaN, < 60 s on the 199k Trk.
