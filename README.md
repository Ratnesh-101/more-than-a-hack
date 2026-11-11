# More than a Hack 2027 — Phase 1: Ball Possession

```bash
git clone https://github.com/Ratnesh-101/more-than-a-hack && cd more-than-a-hack
pip install -r requirements.txt
python -m mtah.reproduce --team nirakshak
```
Continuous integration (.github/workflows/tests.yml) runs the tests on Python 3.8 and 3.11 on Ubuntu and Windows.

**Organised by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com)
and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**, at the Talent Arena in Barcelona.

A football data challenge on real professional match data, open to analysts, data scientists and
engineers. You are given one complete LaLiga match — two camera feeds, automatic tracking of every
player and the ball, automatic tactical tagging, and the camera-to-pitch geometry — and one
question to answer with it.

---

## The task

**For every frame of the match, say who has the ball.**

```
frame,label
11185,A        away in possession, ball alive
11186,A
...
18240,D        dead ball
...
18684,H        home in possession, ball alive
```

That is the whole task. One CSV, one row per video frame, 199,319 rows. It is scored automatically
against a private frame-by-frame ground truth, so there is a number at the end and no argument
about it — no jury, and no presentation to argue from.

The tracking is automatic and uncorrected: it has gaps and identity
switches, and the ball itself is invisible on 41% of the frames you are scored on.

---

## The challenge in one screen

| |                                                                                                                           |
|---|---------------------------------------------------------------------------------------------------------------------------|
| **Input** | one full match: tactical video, broadcast video, automatic tracking **with the ball** for both, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X`                                                             |
| **Ranked on** | `spellF1` — how many of the match's real possessions you recovered as whole possessions                                   |
| **Also reported** | `macroF1` — the per-frame view, and the first tie-break                                                                   |
| **Baseline to beat** | `spellF1` **0.2433**, in this repository                                                                                  |
| **Ground truth you get** | the first **5 minutes** of play — to check your pipeline, not to train on                                              |
| **Ground truth you are scored on** | the full match, **private**                                                                                               |

It is deliberately **not** a tracking challenge. The tracking, tagging and homography are given to
you so that nobody has to rebuild detection, re-identification and camera calibration before
getting to the actual question. See [CHALLENGE.md](CHALLENGE.md).

---

## Try it right now

The scorer, a 5-minute ground truth and four working baselines are in this repository. No
enrolment, no download, no data:

```bash
pip install -r requirements.txt
python evaluation/run_evaluation.py
```

Scored on the public 5 minutes, not on the full match -- which is why the order below is not the
order of the floor quoted above:

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s  <- compare the poss and dur columns

rank  submission                macroF1  spellF1   poss     dur  
----  ------------------------  -------  -------  -----  ------  
   1  baseline_smooth125         0.5812   0.4242     16    18.5  
   2  baseline_smooth50          0.6207   0.3256     26    10.8  
   3  baseline_smooth25          0.6200   0.2868     36     7.6  
   4  baseline_raw               0.5972   0.0378    131     2.0  
```

Those four are **the same algorithm four times** — nearest tracked player to the ball, then a
majority vote over a window. Only the window changes. `macroF1` barely moves; `spellF1` moves by a
factor of eleven. That disagreement is the whole point of the metric, and
[EVALUATION.md](EVALUATION.md) explains it.

Once you have the data, regenerate them yourself and go further:

```bash
python starter/baseline.py                              # rebuild the four baselines
python evaluation/validate_submission.py <your>.csv     # before every submission
```

**Validate every file before you submit it.** `validate_submission.py` runs exactly the checks the
official scorer runs, and a file that fails them is dropped from the board with a one-line reason.
It takes two seconds and it is the single most common way a good entry scores nothing.

---

## Getting the data

The match assets are released to enrolled participants who have accepted the terms of use.

1. Enrol: **[More than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)**
2. Read and sign the terms and conditions in the registration form
3. You receive a download link.
4. Unpack it into `./data`, or point `MTAH_DATA_DIR` at wherever you put it.

Full description of every asset and how the frames line up: **[DATA.md](DATA.md)**.

---

## Reproduce (team nirakshak)

From a clean clone to `submissions/nirakshak.csv`. Python 3.8 to 3.11. The method is described in
[docs/TECHNICAL.md](docs/TECHNICAL.md); the pipeline is `mtah/v2/`.

One-command reproduction:

```bash
python -m mtah.reproduce --team nirakshak
```

This runs `mtah.audit`, `mtah.run` (twice, confirming byte-identical determinism),
`evaluation/validate_submission.py`, `analysis.real_checks`, and reports the final SHA-256 hash.

**PowerShell**

```powershell
git clone <repository-url>; cd <repository>
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# unpack the data bundle into .\data  (so that data\tactical\BAR-ATH_tactical_atd.txt exists),
# or point to it:  $env:MTAH_DATA_DIR = "D:\mtah2027"
python -m mtah.audit
python -m mtah.run --team nirakshak
python evaluation/validate_submission.py submissions/nirakshak.csv
python -m analysis.real_checks --csv submissions/nirakshak.csv
python -m mtah.guard
python -m analysis.preflight --team nirakshak
```

**bash**

```bash
git clone <repository-url> && cd <repository>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# unpack the data bundle into ./data  (so that data/tactical/BAR-ATH_tactical_atd.txt exists),
# or point to it:  export MTAH_DATA_DIR=/path/to/mtah2027
python -m mtah.audit
python -m mtah.run --team nirakshak
python evaluation/validate_submission.py submissions/nirakshak.csv
python -m analysis.real_checks --csv submissions/nirakshak.csv
python -m mtah.guard
python -m analysis.preflight --team nirakshak
```

- `mtah.audit` is the data-arrival audit: it checks our parse of the bundle against the facts
  published in DATA.md (frame counts, slots, team order) and exits 1 on any FAIL. Run it first.
- `mtah.run` writes the possession file; it is a thin delegate to `mtah.v2.run`. The first run
  streams the raw tracking once (about 2 minutes, per the `mtah/v2/tracking.py` docstring) and
  caches it under `data/cache/`, which git ignores. The cache carries a version and code-hash stamp
  (`tracking.py:cache_stamp`) and is recomputed on mismatch; add `--refresh` to force a rebuild.
- Variants: `mtah.v2.run` has `--variant {semi_markov,champion,touch_only}` (default `semi_markov`).
  The shipped decoder is now `semi_markov`; the former champion, the minimum-stoppage Viterbi, is
  the alternative (`--variant champion`). The choice was made on unlabeled structure evidence: 28
  of 44 champion possessions under 3 s were live islands between dead runs and 11 were flip-and-flip-back
  deflections, while semi_markov has share under 3 s 0.07 vs 0.16, and never on the public scoreboard.
  The fallback command is `python -m mtah.run --team nirakshak --variant touch_only` (candidate 02
  alone) and `--variant champion` is documented as the alternative. They are used only if the
  real-data acceptance gate (`analysis.real_checks`) fails the walking-share premise, decided by
  the lead, never from public-GT score.
- `analysis.real_checks` is the acceptance gate from unlabeled statistics only (possession count,
  durations, share under 3 s, flicker, dead-ball share). It never reads ground truth.
- `mtah.guard` refuses a repository that contains match data, derived exports or secrets. Run it
  before any push.
- The pipeline has no randomness: the same data gives a byte-identical CSV (`mtah.v2.run` prints
  its SHA-256). Golden test: `python -m mtah.v2.tests.test_golden` (synthetic data, no download).
- `analysis.preflight` is the final pre-upload checklist (CSV, SHA-256, PDF, markers, repo hygiene).
  Tests, with no match data: `python -m mtah.v2.tests.test_golden`,
  `python -m mtah.v2.tests.test_fallback`, `python -m mtah.v2.tests.test_e2e`,
  `python -m mtah.v2.tests.test_robust` (about 1 min, 32 edge-case ATD cases),
  `python -m mtah.v2.tests.test_speed_guard` and `python -m pytest tests`.
  `tracking.check_speed_unit` aborts every variant if the speed field does not look like m/s;
  `real_checks` gates are the lead's guesses, to be re-read against real numbers.
- `requirements-analysis.txt` / `requirements-docs.txt` are needed for diagnostics / PDF build
  (`pip install -r requirements-analysis.txt` for `python -m analysis.diagnostics --check-alignment`;
  `pip install -r requirements-docs.txt` for `python docs/build_pdf.py`).

---

## Read these, in this order

| | |
|---|---|
| **[CHALLENGE.md](CHALLENGE.md)** | the problem, the four labels, what counts as possession |
| **[DATA.md](DATA.md)** | every provided asset, its format, and the frame alignment |
| **[EVALUATION.md](EVALUATION.md)** | how you are scored, and why — read before optimising anything |
| **[SUBMISSION.md](SUBMISSION.md)** | the file contract, the three deliverables, how to submit |
| **[RULES.md](RULES.md)** | what you may and may not do |
| **[FAQ.md](FAQ.md)** | |
| **[OVERVIEW.md](OVERVIEW.md)** | the competition as a whole — context, why the 2027 format changed, the calendar |


---

## What is in this repository

```
games.py                    per-match constants, asset paths and the frame alignment
evaluation/
  possession_metrics.py     the metric library — every number on the board is defined here
  run_evaluation.py         the scorer. This exact file produces the official result
  validate_submission.py    check a file before you send it
  gt/                       the public 5-minute ground truth
  results/                  the four baselines; put your submissions here
starter/
  baseline.py               the reference baseline, numpy only
  load_atd.py               read the tracking and line it up with video frames
  load_smart_tagging.py     read the tactical tagging as video-frame intervals
  load_homography.py        read the homography; project pitch <-> pixels
  utils.py                  pitch plotting for the notebooks
  01_smart_tagging.ipynb    guided tour of the tactical tagging
  02_tracking_atd.ipynb     guided tour of the tracking data, with plots
  03_first_submission.ipynb data in, scored submission out — start here
mtah/                       run.py (entry point, delegates to v2), audit.py (data-arrival audit), guard.py (leak guard), selfcheck.py, plus the retired v1 files features.py and decode.py
  v2/                       the submitted pipeline: tracking.py, touch.py, deadball.py, decode.py, run.py, tests/
  experiments/              frozen evidence of the 12-candidate comparison: briefs/, cand_*.py, stack_*.py, court/ (synthetic harness), judge/
analysis/                   diagnostics.py (plots), unlabeled_stats.py (label-free statistics), real_checks.py (unlabeled acceptance gate)
docs/                       the technical document
data/                       unpack the downloaded match assets here. Empty in a fresh clone
```

The scorer you run locally is the scorer that decides the competition. If it scores on your
machine, it scores the same way on ours.

---

## Requirements

**Python 3.8 or newer.** The evaluation needs `numpy` and `pandas` and nothing else; it is tested
under 3.8 and 3.11 and produces identical output under both. `starter/baseline.py` needs nothing
more either — you can reach a scored submission without installing anything else.

`requirements-analysis.txt` / `requirements-docs.txt` are needed for diagnostics / PDF build.
The notebooks additionally want `kloppy` and `matplotlib` (`requirements-starter.txt`), and
**Python 3.9+ is recommended for those**: on Windows, installing Jupyter under 3.8 hits a
`pywinpty` build failure that has nothing to do with this repository.

---

## Timeline

| |                |
|---|----------------|
| registration opens | **October 1, 2026** |
| data released | **October 20, 2026** |
| submission deadline | **November 17, 2026** |
| results announced | **Before November 26, 2026**      |

---

## Questions

Open an issue for anything about the data, the metrics or the documentation — corrections are
genuinely useful, and a fix that helps you helps everyone equally. For enrolment and the terms of
use: **hackathon.talentarena@mobileworldcapital.com**.

## Licence and citation

The code and documentation in this repository are released under the MIT Licence — see
[LICENSE](LICENSE). **The match data is not covered by it**: it is separately licensed for this
competition only, under the terms you accept at enrolment.

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}
```
