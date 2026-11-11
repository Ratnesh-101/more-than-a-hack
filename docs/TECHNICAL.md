# Technical Document: Team nirakshak, More Than a Hack 2027, Phase 1

Ball possession estimation system by team **nirakshak** for More Than a Hack 2027 (Phase 1). Code repository: https://github.com/Ratnesh-101/more-than-a-hack. Licensed under the MIT License (see `LICENSE`). The match data is not included in the repository, as guarded by `mtah.guard`.

## 1. Approach and Summary

For each of the 199,319 tactical frames (158,857 graded across two halves, 40,462 ungraded outside them), the pipeline assigns `H` (FC Barcelona), `A` (Athletic Club), `D` (dead ball), or `X` (ungraded). Data facts (`python -m mtah.audit`): 199,319 records parsed, 158,857 graded frames, ball visible on 59.3% of graded frames (94,214 frames), mean 20.5 tracked objects, tracker speed in m/s (median 1.41 m/s, p99 6.4 m/s).

Our shipped default pipeline is a semi-Markov segment-level duration decoder (`mtah/v2/fallback.py:_decode_semi_markov`), executed via `python -m mtah.run --team nirakshak`. Its output SHA-256 digest is `55fd9ef525637467ee6ef405c96bd9821b41d478afc8a5d488e1831a2f464b5f` across repeated runs. Documented alternatives: `--variant champion` (`7fd3bc08fb6157854437a25d5cb124ead1e5346f54e9b7c74231d3981326413c`) and `--variant touch_only` (`704d7cd316a428c9899bce39df9d579b3fc94e8e3a58842bd35642e089cdbd27`).

The decoder choice between minimum-stoppage Viterbi and semi-Markov was made entirely from unlabeled output structure on the real match: minimum-stoppage Viterbi produced 0.160 of possessions under 3 s (and 33 live islands between dead runs with no ball sighting), whereas semi-Markov produced 0.070. The acceptance band (at most 0.10) was written down before data arrived; no ground-truth label was used. It remains a choice made on this match.

Why: `spellF1` evaluates whole possessions matched by IoU (EVALUATION.md). Per-frame nearest-player rules flicker through scrambles and deflections, creating short artifactual possessions. Combining touch-event evidence with explicit segment duration modeling and switch penalties ensures a possession change occurs only after sustained opposing team contact.

Pipeline execution proceeds through six stages:
1. **Tracking ingestion** (`mtah/v2/tracking.py:from_atd`): Player and ball coordinates scaled to pitch dimensions (105.3 m x 68.0 m).
2. **Touch evidence** (`mtah/v2/touch.py:team_logodds`): Establishes team log-odds evidence `r` from kicks and sustained carries.
3. **Dead-ball emissions** (`mtah/v2/fallback.py:emissions`, `mtah/v2/fallback.py:_v1_log_dead`): Stoppage logits from walking shares and boundary cues.
4. **Duration decoding** (`mtah/v2/fallback.py:_decode_semi_markov`): Resolves sequence labels using segment dynamic programming with lognormal priors.
5. **Kickoff suppression** (`mtah/v2/decode.py:decode_half`): Enforces dead ball prior to start-of-half kickoff tags via Smart Tagging.
6. **Live island suppression** (`mtah/v2/decode.py:_label_evidence_free_islands_dead`): Converts evidence-free live runs into `D`. Ungraded frames receive `X` (`games.py:half_video_blocks`).

![End-to-end pipeline of the shipped semi-Markov possession decoder.](out/figures/pipeline.png)

| Metric | Half 1 | Half 2 | Full Match |
|---|---|---|---|
| Possessions count | 130 | 138 | 268 |
| Mean possession duration | 17.2 s | 15.2 s | 16.1 s |
| Share of possessions under 3 s | 0.046 | 0.094 | 0.071 |
| Direct H<->A flicker per minute | 1.15 /min | 1.23 /min | 1.19 /min |
| Dead ball (D) frame share | 0.298 | 0.341 | 0.319 |
| Longest possession run | 98.3 s | 83.8 s | 98.3 s |
| Longest dead ball (D) run | 146.4 s | 143.2 s | 146.4 s |
| Leading D run at half start | 0.0 s | 116.2 s | 116.2 s |
| Kickoff team (first live label) | A | H | A, H (alternates per Law 8) |
| Post-goal restart side | 2 of 2 | 2 of 2 | conceding side in 4 of 4 cases |
| Frame counts (H / A / D / X) | 33,264 / 22,544 / 23,659 / 0 | 30,370 / 21,941 / 27,079 / 0 | 63,634 / 44,485 / 50,738 / 40,462 |

## 2. Possession Rules Implemented

The pipeline implements seven explicit possession and stoppage rules grounded in football structure and tracking mechanics:

1. **Loose balls**: Possession remains with the team of the last credible touch throughout loose balls, aerial trajectories, and unobserved tracking gaps (`mtah/v2/touch.py:team_logodds`). The team log-odds `r` reflects the difference between time elapsed since the opponent's last touch and time since the team's last touch. In the absence of new touches, `r` remains constant, and a transition cost `SWITCH_COST = 30.0` nats prevents spurious possession alternations.
2. **Contested touches**: Duels and contested touches are credited simultaneously to both teams (`mtah/v2/touch.py:_touches`). Using contest margin `MARGIN = 0.5` m, when players from opposing teams are equidistant to the ball, neither team establishes control. Both timers reset, holding `r` near zero and preserving the prevailing possession state.
3. **Restarts**: Dead ball frames carry no team ownership (`mtah/v2/fallback.py:emissions`). Stoppage frames receive the label `D`. Because dead ball does not reset the team log-odds timer, the team with last touch prior to the stoppage retains possession upon restart until the opposing team executes a touch.
4. **Hidden-ball gaps**: When ball possession changes across an unobserved tracking occlusion, the transition point is placed at the temporal midpoint of the gap (`mtah/v2/touch.py:_backdate`). If the ball disappears under Team H and reappears with Team A, assigning the handover to the midpoint minimises expected boundary displacement error. When the same team retains the ball across the occlusion, possession continuity is maintained throughout.
5. **Dead ball cues**: Play is classified as dead (`mtah/v2/fallback.py:_v1_log_dead`) when a high fraction of players are walking (`SLOW_SPEED = 1.5` m/s, `SLOW_WIDTH = 0.35`, `DEAD_SLOW_BONUS = 3.0` nats), when the ball crosses pitch boundaries (`DEAD_OUT_BONUS = 5.0` nats, `OUT_MARGIN_M = 0.5` m per Law 9), or when the ball rests motionless (`STILL_R = 1.0` m over 1.0 s, `STILL_BONUS = 3.0` nats), offset by recent kick events (`KICK_LIVE = 3.0` nats for `LIVE_S = 2.0` s). Baseline stoppage log-odds default to `DEAD_BIAS = -2.5` nats when all cues are quiescent.
6. **Smart Tagging kickoff rule**: Frames preceding each half's kickoff tag start are marked `D` (`mtah/v2/decode.py:decode_half`, `mtah/v2/tracking.py:load_kickoffs`). The Smart Tagging file is consumed as automatic, uncorrected input, reading only the "Kick Off (Start)" timestamps. In Half 1, the tag begins at frame 11097, prior to the first graded frame 11185, resulting in a 0.0 s dead run at the start of Half 1. In Half 2, the tag begins at frame 113660, yielding an unbroken leading dead run of 2,905 frames (116.2 s) from frame 110808 to 113659.
7. **Evidence-free live island rule**: A live possession run (`H` or `A`) bounded by dead runs (`D`) on both sides with zero ball sightings is converted to `D` (`mtah/v2/decode.py:_label_evidence_free_islands_dead`). On the real match, this rule relabels exactly 4 runs totaling 868 frames, all in Half 2, adjusting total match possessions from 272 down to 268 on the default decoder.

## 3. Assets Used and Dropped, Training Data

**Assets used:**
- **Tactical ATD tracking file** (`data/tactical/BAR-ATH_tactical_atd.txt`): 199,319 frames parsed, providing player positions, tracker speeds, and ball coordinates.
- **Tactical metadata XML** (`data/tactical/BAR-ATH_tactical_atd_meta.xml`): Slot-to-team mapping across 46 slots (Barcelona as home, Athletic Club as away).
- **Tactical Smart Tagging XML** (`data/tactical/BAR-ATH_tactical_tagging.xml`): Start timestamps of the two "Kick Off (Start)" tags, parsed automatically without human modification.
- **Pitch dimensions**: 105.3 m x 68.0 m from DATA.md.

**Assets dropped:**
- **Video recordings**: Tactical and broadcast MP4 files were unread.
- **Broadcast feed**: Broadcast tracking stream was unread.
- **Homography**: Homography calibration files were unread; ATD coordinates directly scale to metric pitch positions.
- **Tactical tags**: All event tags other than start-of-half kickoffs (passes, shots, fouls, offsides, substitutions, cards) were unread.

**Training data: none.** No hand labels were created or used; no data from the provided match was used as labels; no learned or fitted parameters exist in the pipeline; no external or event datasets (Opta, StatsBomb, Wyscout, etc.) were used; and no pre-trained model and no LLM is in the pipeline.

The public 5-minute ground truth window was scored once, early in development on an earlier decoder version, solely as a sanity check of format and coordinate alignment as permitted by competition rules. No subsequent change or parameter selection used it, and we have no full-match ground truth score.

**Parameter justification and sensitivity:** Every fixed constant in the pipeline is cataloged in `docs/PARAMETERS.md` along with its design basis (official document, measured on the match without labels, football law, physical limit, design choice not validated against labels, heuristic no support found). A one-at-a-time perturbation (0.5x and 2.0x) of pipeline constants on the real tracking data shows that output is most sensitive to dead-ball thresholds, where altering `fallback.SLOW_SPEED` changes up to 45.18% of graded frames, followed by `fallback.DEAD_BIAS` (26.10%) and `fallback.DEAD_SLOW_BONUS` (24.42%). Across the top-10 most sensitive parameters, four govern stoppage detection and six govern touch attribution (`touch.HOLD` at 13.75%, `touch.DV_KICK` at 9.79%, `touch.K` at 9.50%, `touch.R_CTRL` at 9.47%, `fallback.STILL_R` at 6.73%, and `touch.R_KICK` at 5.78%). None of these constants was tuned against labels or scores, and several have no measured empirical support beyond fixed design choices (see `docs/PARAMETERS.md`).

## 4. Results from the Real Match

The pipeline was executed on the match data via `python -m mtah.run --team nirakshak`.

**Official submission validation:** Validation via `python evaluation/validate_submission.py submissions/nirakshak.csv` confirmed format compliance:

```
OK    nirakshak.csv
      199319 rows, frame 0..199318

      label counts
        H    63634   31.9%
        A    44485   22.3%
        D    50738   25.5%
        X    40462   20.3%

      reports 268 possessions of mean 16.1 s
      (counted with each X as a gap; the scorer bridges abstentions inside a real
       possession, so its count can be lower than this one)

Valid. Note this checks the format only -- it says nothing about whether the labels are right.
```

**Label-free diagnostic checks:** Diagnostic evaluation via `python -m analysis.real_checks --csv submissions/nirakshak.csv` returned 0 FAIL, 2 WARN:

```
WARN  tracking  walking-share bimodal  0  valley found
WARN  ball      ball missing in D frames  0.601  <= 0.5
```

The diagnostic output reported the following verified properties of the tracking and prediction:
- **Ball roughness**: `INFO ball roughness x m median / p99: 0.004 / 0.040`, `INFO ball roughness y m median / p99: 0.005 / 0.048`.
- **Kick enrichment across thresholds**: `INFO kick player within 2 m (all usable frames): 0.515`; `INFO kick DV_KICK 3 m/s (/min, near share vs all): 20.72 /min, 0.587 vs 0.515`; `INFO kick DV_KICK 5 m/s (/min, near share vs all): 7.58 /min, 0.437 vs 0.515`; `INFO kick DV_KICK 8 m/s (/min, near share vs all): 1.47 /min, 0.211 vs 0.515`.
- **Kickoff structure**: `INFO kickoff half 1/2 first live label alternates: A, H (alternates)`; `INFO kickoff post-goal kickoff side opposite to scorer: 4/4 holds`.

![Possession density across visible-ball frames (59.3% visible; 490 out-of-pitch frames excluded).](out/figures/possession_heatmap.png)

![Match timeline of decoded possession and stoppage sequences across both halves.](out/figures/possession_timeline.png)

## 5. Limitations

**1. Unobserved ball tracking:** The ball is untracked on 40.7% of graded frames (64,643 of 158,857 frames), with longest continuous occlusion gaps of 138.6 s in Half 1 and 221.5 s in Half 2. Across these intervals, possession tracking relies entirely on the team of the last recorded touch and gap-midpoint handover logic.

**2. Kick detector selectivity:** The kick detector barely isolates deliberate kicks on this dataset. As measured in the kick enrichment table, at `DV_KICK = 5.0` m/s, the share of kick events with a player within 2.0 m is 0.437, which is lower than the baseline player-near-ball proportion of 0.515 across all usable frames. At 8.0 m/s, it falls to 0.211. Spurious ball velocity spikes from bounces or tracking noise trigger kick detections without a player nearby.

**3. Ball tracking smoothness and coordinate noise sensitivity:** The real ball trajectory is smooth to approximately 4 mm (roughness median: 0.004 m in x, 0.005 m in y), so raw coordinate noise is small. However, an in-memory perturbation stress test shows sharp sensitivity to ball coordinate jitter: adding Gaussian noise with sigma = 0.1 m alters 13,956 graded frames (8.7853%, seed 1100), while sigma = 0.3 m alters 33,749 graded frames (21.2449%, seed 1300), increasing direct flicker from 1.15-1.23 to 2.06-2.11 /min (measured on the decoder prior to its last two post-processing corrections).

**4. Visibility and speed dropouts:** Blanking the ball on an extra 10% and 20% of visible frames in 2.0 s bursts changes 4.32% to 6.23% (10%) and 9.89% to 9.98% (20%) of graded frames. In contrast, dropping player speed on 10% of slots changes under 1.5% of graded frames (0.75% to 1.46% for entire slot columns; 0.20% for random entry dropout), as mean player speed averages across multiple players.

**5. Dead-ball parameter sensitivity:** Predictions are highly sensitive to dead-ball thresholds. Perturbing `SLOW_SPEED` alters up to 45.18% of graded frames, `DEAD_BIAS` alters 26.10%, and `DEAD_SLOW_BONUS` alters 24.42%. None of these parameters was tuned against ground truth, and several lack direct empirical measurement. The top rows of the sensitivity table from `docs/PARAMETERS.md` are:

| Rank | Constant / Parameter | Default Value | File Location | Change at 0.5x | Change at 2.0x | Max Impact |
|---|---|---|---|---|---|---|
| 1 | `fallback.SLOW_SPEED` | 1.5 m/s | `mtah/v2/fallback.py:39` | 22.05% | 45.18% | 45.18% |
| 2 | `fallback.DEAD_BIAS` | -2.5 nats | `mtah/v2/fallback.py:41` | 26.10% | 25.95% | 26.10% |
| 3 | `fallback.DEAD_SLOW_BONUS` | 3.0 nats | `mtah/v2/fallback.py:43` | 21.77% | 24.42% | 24.42% |
| 4 | `fallback.SLOW_WIDTH` | 0.35 | `mtah/v2/fallback.py:40` | 9.82% | 18.40% | 18.40% |
| 5 | `touch.HOLD` | 12 frames | `mtah/v2/touch.py:22` | 13.75% | 13.47% | 13.75% |
| 6 | `touch.DV_KICK` | 5.0 m/s | `mtah/v2/touch.py:18` | 9.79% | 6.39% | 9.79% |

**6. Walking-speed distribution:** The dead-ball cue assumes a clear division between walking and running, but `analysis.real_checks` reports `WARN tracking walking-share bimodal 0 valley found`. Without a clear valley in the walking-share distribution, setting `SLOW_SPEED = 1.5` m/s is an uncalibrated design choice.

**7. Smart Tagging kickoff offset:** Kickoff tags precede the published kickoff frames in DATA.md and FAQ.md by 88 frames (3.5 s) in Half 1 and 82 frames (3.3 s) in Half 2. Consequently, Half 2 turns live approximately 3.3 s before the published kickoff frame. We state this plainly and do not correct it with an invented constant.

**8. Kickoff initial possession evidence:** The first possession following a kickoff has weak touch evidence because the initial touch is brief.

**9. Single match evaluation:** Evaluated on a single match (FC Barcelona vs Athletic Club). Constants have not been evaluated on other matches or under different tracking systems.

**Future work:** With more time, we would model ball elevation or 3D trajectories to distinguish bounces and aerial passes from ground kicks, implement explicit restart-team attribution rules reflecting referee decisions and set-piece laws (Laws 13, 15, 17), and calibrate player stoppage speed thresholds adaptively from match-wide motion clustering without labels.

## 6. Reproduction and Compliance

**Reproduction:** Reproduction requires Python 3.8 to 3.11. The match data is licensed and not included in the repository.

Single-command reproduction from repository root: `python -m mtah.reproduce --team nirakshak` (accepts optional `--data-dir PATH`; runs the audit, executes the decoder twice to verify determinism, validates the submission, evaluates real-data diagnostics, and checks the final SHA-256 digest `55fd9ef525637467ee6ef405c96bd9821b41d478afc8a5d488e1831a2f464b5f`).

Step-by-step reproduction and verification commands:

```bash
python -m mtah.audit
python -m mtah.run --team nirakshak
python evaluation/validate_submission.py submissions/nirakshak.csv
python -m analysis.real_checks --csv submissions/nirakshak.csv
python -m mtah.run --team nirakshak --variant touch_only
python -m mtah.guard
python -m analysis.preflight --team nirakshak --sha256 55fd9ef525637467ee6ef405c96bd9821b41d478afc8a5d488e1831a2f464b5f --repo-url https://github.com/Ratnesh-101/more-than-a-hack
```

**Compliance statement:**
- **Algorithmically generated labels**: Every label was generated entirely by deterministic code without human intervention or manual inspection.
- **No human-produced match data**: No event data (Opta, StatsBomb, Wyscout, etc.) or manual tracking corrections were accessed or used.
- **No full-match ground truth**: Full-match ground truth was never accessed or referenced.
- **Public window check only**: The public 5-minute ground truth was evaluated once early in development solely to check format compliance and alignment; no subsequent code changes or parameter selections used it.
- **Automatic Smart Tagging**: Smart Tagging was consumed strictly as an uncorrected automated stream for kickoff timestamps.
- **No learned parameters**: The pipeline contains no learned parameters, no pre-trained weights, and no LLM in the pipeline.
- **Data protection**: Competition match data is excluded from the repository via `.gitignore` and enforced by `mtah.guard`, preventing redistribution or raw coordinate export.
- **Licensing**: The codebase and documentation are published under the MIT License (see `LICENSE`).
