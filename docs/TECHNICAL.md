# Technical document: team nirakshak, More Than a Hack 2027, Phase 1 (ball possession)

Status: DRAFT for the champion design in `mtah/v2/`. No real-data result exists yet; every missing number is `TODO(result)`. Every number labelled "synthetic" comes from our own synthetic match generator, not from the provided game. Code references are `file:function` (no match data is in the repository).

## 1. Approach

For each of the 199,319 tactical frames the pipeline writes `H`, `A`, `D` or `X`:

`raw tactical ATD` -> `Trk` arrays (`mtah/v2/tracking.py:from_atd`) -> touches and team log-odds (`touch.py:team_logodds`) -> dead-ball logit (`deadball.py:dead_logit`) -> 3-state emissions (`decode.py:emissions`) -> Viterbi with a hard 3 s minimum stoppage, switch cost 30, dead cost 12 (`decode.py:_viterbi_min_dead`) -> `X` outside the two graded halves (`decode.py:decode`; blocks 11185-90651 and 110808-190197 from `games.py:half_video_blocks`).

- **Tracking** (`tracking.py:_stream`): player x, y (scaled to metres by the 105.3 x 68.0 m pitch), player speed, team of each of the 46 slots (`starter/baseline.py:team_of_each_track`, first listed team = home) and the ball.
- **Team evidence** (`touch.py`): possession is the team of the last credible touch. A touch is a kick (ball velocity changes by more than 5 m/s at a player) or a carry (ball within 1.5 m of one team and clearly nearer it, for 1 s). The H-vs-A log-odds `r` is the clipped difference "time since the last A touch" minus "time since the last H touch" in seconds, so it is constant through ball-missing gaps and loose balls and near zero when both teams touch in turn.
- **Dead ball** (`deadball.py`): a sum of football cues (most players walking, ball beyond the lines, ball still on a restart spot, kickoff geometry, minus cues for sprinting and passing). Emissions are `[log P(H) + log(1-p), log P(A) + log(1-p), log p]` with `p` the dead probability, so D never carries a team.
- **Decoder** (`decode.py`): exact Viterbi over H, A, D per half in which every dead run lasts at least 3 s.

Why: `spellF1` scores whole possessions matched by IoU (EVALUATION.md). Per-frame nearest-player labels flicker through scrambles and deflections and invent short possessions. Touch evidence plus a switch cost makes a team change only after sustained touching.

## 2. Possession rules implemented

| Rule | Code | Football justification |
|---|---|---|
| Last-touch team keeps possession through loose balls and ball-missing gaps | `touch.py:team_logodds` (`r` depends only on time since each team's last touch, so it does not change without a touch); `decode.py:_viterbi_min_dead` (SWITCH_COST) | CHALLENGE.md: contested or loose ball stays with the last team. The v1 nearest-player rule changed team whenever another player became nearest. |
| A team takes over only on a kick or a carry | `touch.py:_touches` (`kick`: `dv > DV_KICK` at a player within `R_KICK`; `held_h`/`held_a`: within `R_CTRL` for `HOLD`) | A pass flying past a defender is not a touch (ball velocity unchanged); a deflection or won aerial is not control. |
| Contested touch is credited to both teams | `touch.py:_touches` (`reach_h`/`reach_a` use `MARGIN`; `held_*` require a clear `MARGIN` advantage) | A duel gives neither team control; both timers reset, `r` stays near zero. |
| Hand-over inside a hidden-ball gap is placed at the gap midpoint | `touch.py:_backdate` | If the ball reappears at the other team's feet, it changed hands somewhere in the gap; with no evidence of when, the midpoint minimises expected boundary error. |
| Dead ball from the share of players walking | `deadball.py:dead_logit` (`frac_walk`, `slow`, `W_SLOW`, `W_HSR`) | In a stoppage almost everyone walks; a share is robust to one restarter or a ghost slot, where a mean speed is not. |
| Dead ball: ball outside the lines | `deadball.py:dead_logit` (`out`, `W_OUT`, persisted 2 s) | Ball over a line is out of play; persistence because a lofted ball projects beyond the lines. |
| Dead ball: still ball on a restart spot | `deadball.py:dead_logit` (`still`, `spot`, `W_STILL`, `W_SPOT`) | Corner arc, touchline, goal area and centre spot are where restarts are taken. |
| Dead ball: kickoff geometry | `deadball.py:dead_logit` (`kickoff`, `W_KICKOFF`) | Both teams entirely in their own halves happens only at a kickoff; works with the ball missing. |
| Minimum stoppage 3 s | `decode.py:_viterbi_min_dead` (`k = MIN_DEAD_S * fps`; the first frame of a half is exempt) | Fastest real restarts take about 3 s; a shorter D is an artefact that would split a possession. |
| No ownership inferred for dead ball | `decode.py:emissions` (D emission is `log p`, no team term) | D is its own label; the team log-odds `r` is not reset by D, so the last toucher is the default after a stoppage. There is no restart-team rule. |
| No `X` inside graded halves | `decode.py:decode` fills `X` everywhere, then overwrites every frame of each block from `_viterbi_min_dead`, which returns states 0, 1, 2 only | Verified by reading the code; confirm on the real CSV (Claims to verify). |

## 3. Parameters

Every constant in `touch.py`, `deadball.py`, `decode.py`. "Reason" is what the briefs or code comments say. Nothing was fitted. No constant is learned from the provided game. Constants marked **dev** were checked against our synthetic dev seeds 1-5, which are not independent of the generator.

| Constant | Value | Meaning | Reason as written |
|---|---|---|---|
| `touch.K` | 6 frames | velocity line-fit half-window (0.24 s) | averages tracker jitter but keeps touches apart |
| `touch.V_MAX` | 40 m/s | steps faster are glitches | above hardest shot (~35 m/s) |
| `touch.DV_KICK` | 5 m/s | velocity change that counts as a kick | deliberate pass/trap/shot exceeds 18 km/h; jitter and brushes do not. Assumes small real ball jitter, unverified |
| `touch.R_KICK` | 2.0 m | reach for a kick | leg ~1 m plus ~1 m position error |
| `touch.R_CTRL` | 1.5 m | carry distance | ball rarely farther than 1.5 m from a dribbler |
| `touch.MARGIN` | 0.5 m | team-distance margin for contested touch | both this close = duel, not control. Brief 03 cites synthetic carry offset 0.7 m (flat 1.2-2 m) for a related radius |
| `touch.HOLD` | 12 frames each side (1 s) | carry must persist | "keeping the ball, not passing by it". No sensitivity run recorded; whether it was set after viewing dev seeds is not recorded - ask lead |
| `touch.TAU` | 1.0 s | seconds of touch lead per nat | 1 s lead = 1 nat. Synthetic: TAU 2 s gives 0.598 vs 0.597 (cand 02 alone) |
| `touch.CLIP` | 3.0 nats | cap on `r` | one old touch cannot outweigh 3 s of fresh evidence. Whether set after viewing dev seeds is not recorded - ask lead |
| `touch.BIG` | 1e9 | "never touched" sentinel | bookkeeping, not a tuning parameter |
| `deadball.OUT_MARGIN_M` | 0.5 m | ball-out margin | ball-tracking error near a line; same as v1 |
| `deadball.WALK_MPS` | 2.0 m/s | walking pace ceiling | top of walking (~7 km/h). **dev**: sits just above the synthetic dead-walk speed 1.6 m/s; synthetic 1.8 gives -0.020 spellF1, 1.5 gives -0.07, 2.5 gives -0.009. A cliff |
| `deadball.HSR_MPS` | 5.5 m/s | high-speed-running threshold | standard in football physical reports (19.8 km/h). **dev**: equals the synthetic chaser speed; 5.0 and 6.0 are flat |
| `deadball.SLOW_FRAC` / `SLOW_WIDTH` | 0.9 / 0.03 | walking share that signals a stoppage; softness | at most ~2 of 20 outfielders above walking pace; ~1 player of softness. Synthetic 0.85/0.95 gives -0.018/-0.013 |
| `deadball.SMOOTH_S` | 1.0 s | player cues averaged | a stoppage is a state, not a frame |
| `deadball.SPEED_LAG` | 2 frames | half-window of position-derived speed | UNJUSTIFIED - ask lead |
| `deadball.BALL_WIN_S` | 1.0 s | ball displacement window | long enough to ignore jitter |
| `deadball.BALL_STILL_MPS` / `BALL_PASS_MPS` | 1.5 / 8.0 m/s | ball "still" / "passing" | slower than walking / slowest typical ground pass |
| `deadball.PERSIST_S` / `PERSIST_FRAC` | 2.0 s / 0.75 | ball cues must hold this long, on this share of sightings | longer than a lofted ball's spell beyond the line or a shielding pause; tolerates a glitch or two |
| `deadball.SPOT_R_M` | 2.0 m | restart-spot radius | corner arc (1 m) / centre spot plus tracking error |
| `deadball.LINE_M` | 1.0 m | throw-in band | taken on or behind the touchline |
| `deadball.GOAL_AREA` | (6.0, 10.0) m | goal-area depth, half-width | goal area 5.5 m deep, 18.32 m wide, plus error |
| `deadball.OWN_HALF_FRAC` | 0.95 | kickoff share in own half | allows one mis-tracked slot |
| `deadball.DEAD_BIAS` | -2.5 | prior dead log-odds | inherited from v1, "kept, not re-tuned"; v1 gave no per-value reason - UNJUSTIFIED - ask lead |
| `deadball.W_SLOW` / `W_OUT` | 3.0 / 5.0 | cue weights (nats) | inherited from v1 unchanged; no per-value reason - UNJUSTIFIED - ask lead |
| `deadball.W_HSR` | -2.0 per runner, cap 2 | sprinting means live | standard high-speed-running band; the weight itself has no stated reason - UNJUSTIFIED - ask lead |
| `deadball.W_STILL`, `W_SPOT`, `W_KICKOFF`, `W_PASS` | 1.5, 2.0, 3.0, -2.0 | other cue weights | "moderate cues that cannot outvote the player cue alone". Synthetic: the four ball cues together add +0.003 spellF1, i.e. noise; the kickoff cue never fires on synthetic data |
| `decode.SWITCH_COST` | 30 | H<->A penalty (nats) | ~1.5 s of new-team touching flips; a deflection (flip and back) costs 60. Synthetic plateau: 20/45 gives 0.598/0.593 (cand 02 alone) |
| `decode.DEAD_COST` | 12 | live<->dead penalty | v1's value, untouched - UNJUSTIFIED - ask lead. Synthetic plateau: 12/30 gives 0.592/0.595 (cand 02 alone) |
| `decode.MIN_DEAD_S` | 3.0 s | minimum stoppage | fastest real restarts (quick throw-in, free kick). Synthetic: 5 s scored higher (0.537 vs 0.530); 3 s kept on football grounds |
| `decode.EPS` | 1e-6 | probability clip | keeps log-probabilities finite |

## 4. Assets and training data

**Used:** tactical ATD (player positions, player speeds, ball position) and its metadata (team of each slot); pitch size. **Not used:** tactical or broadcast video, homography, Smart Tagging, broadcast feed (no code in `mtah/v2/` opens them; `tracking.py` requests only `atd_meta` and `atd_raw`). Why they were not used is not recorded beyond scope. The public 5-minute ground truth is used only to check format and alignment (`evaluation/validate_submission.py`, `analysis/diagnostics.py --check-alignment`), never to tune or choose anything.

**Training data: none.** No hand labels anywhere, no data from the provided game, no learned or fitted parameters, no external or event data. All constants are fixed in source (section 3).

**Synthetic data.** Development used a synthetic match generator written by the team (`mtah/experiments/court/synth.py`). Its ground truth is the generator's own. It cannot tell us how real annotators label possession. It rewards ball-proximity logic by construction (the ball snaps onto the receiver, carries sit ~0.7 m from a player), and its ball noise (0.2 m), ball-missing gaps (41%) and dead-ball physics (players walk at 1.6 m/s, ball perfectly still) are our guesses. The briefs flag several constants as fitted to it (section 3, **dev**).

## 5. Development process

- Twelve candidate designs were built independently against one shared synthetic harness (`mtah/experiments/court/`, briefs in `mtah/experiments/briefs/`) and compared on synthetic hidden seeds 101-110 (20 min each, decision rule fixed beforehand, `VERDICT.md`). The experiments are kept as frozen evidence.
- Champion: touch-event team evidence (02) + multi-cue dead ball and minimum-stoppage decoder (08). Synthetic spellF1 0.648 against 0.464 for the v1 nearest-player baseline: paired mean +0.184 (bootstrap 95% CI +0.158 to +0.209), ahead on 10/10 seeds. Synthetic only.
- Dropped, one line each (synthetic, `VERDICT.md`): pass-chain confirmation (03: 0.588, below touch events); local superiority/density (07: wrong-signed, near ties v1, since defenders outnumber the attacker near the ball); pitch-control intercept time (11: no spellF1 gain, CI includes 0); team shape (06: 0.511, weak); territory (09: 0.537, weights fitted on pseudo-labels from the match's own nearest-player frames); ball reconstruction (05: stacked on the champion +0.007, CI includes 0); movement (10) and dwell time (04): below touch events; semi-Markov decoder (12) as a layer: stack 0.640 vs 0.648, so the min-stoppage decoder won; incumbent-plus (01): CI includes 0.
- Dead-ball (08) alone and decoder (12) alone are not significant on hidden seeds; both pay off only on top of touch-event team evidence.
- **Real-data performance is unmeasured until the real run.** Section 8 is empty until then.

## 6. Limitations

- The walking-share dead-ball cue may fail if real stoppage players jog, or if the ATD speed definition differs from m/s: the share never reaches 0.9, D is missed, and WALK_MPS has a cliff (synthetic -0.02 at 1.8 m/s).
- Kick detection depends on real ball jitter. Real jitter of 0.5-1 m or bounces create spurious kicks. Ball height is not observed, so bounces and headers near a player read as kicks.
- The still-ball cue is silent if the ball is untracked at stoppages. A keeper holding the ball can be marked D.
- Possessions are undercounted on synthetic data (53 predicted vs 67 true). Turnovers shorter than 2-3 s are structurally missed (minimum evidence, 3 s stoppage floor).
- A hand-over hidden in a long ball gap is placed at the gap midpoint, which can be wrong by half the gap.
- No restart logic: after a stoppage the last toucher keeps possession until the other team touches the ball.
- A wrong-team or duplicate slot that carries the ball counts as the other team's control.
- Home/away comes from team order in the ATD metadata (first = home; DATA.md says home is Barcelona).
- One match only; the constants have not been tested elsewhere.

With more time: measure ball jitter and the walking-speed distribution on the real data (label-free) and replace the guessed ones; add an explicit restart-team rule.

## 7. Reproduction

Python 3.8 to 3.11. The data is gated and is not in the repository.

```bash
git clone <repository-url> && cd <repository>
python -m venv .venv                    # activate it (see README.md, Reproduce)
pip install -r requirements.txt
# unpack the data bundle into ./data, or set MTAH_DATA_DIR
python -m mtah.audit                    # data-arrival audit (checks our parse against DATA.md)
python -m mtah.run --team nirakshak     # writes submissions/nirakshak.csv
python evaluation/validate_submission.py submissions/nirakshak.csv
python -m analysis.real_checks --csv submissions/nirakshak.csv   # unlabeled acceptance gate
python -m analysis.preflight --team nirakshak                    # final checklist, run last
```

- The first run streams the raw ATD once (about 2 minutes, quoted from the `tracking.py` docstring, not measured) and caches the tracking under `data/cache/`. The cache carries a version and code-hash stamp (`tracking.py:cache_stamp`) and is recomputed when it does not match; `--refresh` forces a rebuild.
- Fallbacks: `mtah.run` delegates to `mtah.v2.run`, which has `--variant {champion,touch_only,semi_markov}` (default champion). `touch_only` is candidate 02 alone and `semi_markov` is the runner-up stack_1 (`mtah/v2/fallback.py`). They are pre-built and switched with `python -m mtah.run --team nirakshak --variant touch_only`. They are used only if the real-data acceptance gate (`analysis.real_checks`) fails the walking-share premise, a decision for the lead, never made from public-GT score.
- Deterministic: no randomness in `mtah/v2/`, tie-breaks are fixed (`_viterbi_min_dead` keeps the current state), the file is written with `\n` newlines, and `run.py` prints a SHA-256 of the output. Golden test: `python -m mtah.v2.tests.test_golden` checks the port against the experiment on synthetic seeds 1-3 (emissions to 1e-9), determinism and a runtime limit.
- Tests (all synthetic, no match data): `python -m mtah.v2.tests.test_golden`, `python -m mtah.v2.tests.test_fallback` (fallback variants equal their experiments), `python -m mtah.v2.tests.test_e2e` (synthetic bundle through the entry point and the official validator) and `python -m pytest tests`. `python -m analysis.preflight --team nirakshak` is the last step before upload.
- `python -m mtah.guard` must pass before any push (blocks match data and derived exports).
- `analysis.real_checks` implements the acceptance gates of `VERDICT.md` item 4 from unlabeled statistics only (possessions per half, mean duration 8-25 s, share under 3 s at most 0.10, flicker 0.5-3 per minute, D share, no half starting over 60 s in D). It never reads ground truth. Those gates were set by us before any real run.

## 8. Results (to be filled after the real run)

Nothing below has been measured on the provided game. Do not read the synthetic figures in section 5 as results.

| Item | Value |
|---|---|
| Possessions per half and total | TODO(result) |
| Mean possession duration | TODO(result) |
| Share of possessions under 3 s | TODO(result) |
| Direct H<->A flicker per minute | TODO(result) |
| Label counts H / A / D / X | TODO(result) |
| `validate_submission.py` output | TODO(result): paste literal output |
| `analysis.real_checks` output | TODO(result): paste literal output |
| Private-set `spellF1` / `macroF1` | not available to the team |

## Claims to verify

1. All "synthetic" numbers (0.648, +0.184, 53 vs 67, plateaus, cliff values) are copied from `VERDICT.md` and the briefs, not re-run for this document.
2. "About 2 minutes" is quoted from the `tracking.py` docstring.
3. "No `X` inside graded halves" is from reading `decode.py`; the real CSV has not been inspected.
4. Home = first ATD team is an assumption from `starter/baseline.py` and DATA.md.
5. Whether `HOLD` and `CLIP` were set after viewing synthetic dev seeds is not recorded in the briefs; ask lead.
6. The WALK_MPS premise (real stoppage players walk below 2 m/s in the ATD speed unit) is unchecked until `analysis.real_checks` runs on the real data.
7. Section 3 lists `UNJUSTIFIED - ask lead` where neither briefs nor code comments give a reason; the lead may hold reasons that are not written down.
8. Python 3.8 to 3.11 support is claimed from the pinned requirements; `sliding_window_view` needs numpy 1.20 or newer (pinned 1.24.4).
