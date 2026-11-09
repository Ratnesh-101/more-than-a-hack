# Court: shared harness for feature candidates

Run from the repo root with the project venv:

    .venv\Scripts\python.exe -m mtah.experiments.court.run                 # incumbent only, 3 seeds
    .venv\Scripts\python.exe -m mtah.experiments.court.run --cand 03_foo --seeds 1,2,3,4,5
    .venv\Scripts\python.exe -m mtah.experiments.court.run --real          # + unlabeled real stats
    .venv\Scripts\python.exe -m mtah.experiments.court.run --rules-lint    # static rules check only

## Add a candidate
1. Create `mtah/experiments/cand_<nn>_<slug>.py` (auto-discovered; `--cand` takes name substrings).
2. Expose `NAME`, `PARAMS` (only `switch_cost`, `dead_cost` honoured) and
   `emissions(trk) -> (n, 3) float64` log-likelihoods for H, A, D. Optional `decode(trk, e)`.
3. The input `trk` is documented in `court/__init__.py` and `court/trk.py`. Read nothing else.
4. Run the court. A candidate that raises, returns bad shapes/NaN or breaks the lint is reported
   `DISQUALIFIED(...)`; the run continues.

## What you get
- Ranking by mean realistic-synthetic spellF1 (also macroF1, easy-synthetic scores, possession count
  vs truth, mean duration, share under 3 s, H<->A flicker per minute, runtime). Per-seed table too.
- Written to `mtah/experiments/court/out/` (git-ignored): `court_synth.md/.json`, `court_real.md/.json`.
- `--real`: possession count, mean duration, share <3 s / <5 s, longest, H/A/D/X shares, flicker per
  half. No label is read anywhere. Needs the gated data (or its cache `data/cache/*_court_trk.npz`);
  otherwise it says so and skips.

## Reading the numbers
- Differences under ~0.04 spellF1 are noise on a real match; synthetic seeds are cheaper, so use
  5+ seeds and compare against the incumbent row.
- Synthetic truth is ours: it tests robustness to missing ball, glitches, tracklet churn and
  look-alike scrambles, NOT whether a feature matches how real football is annotated. Use the
  `--real` stats (no 2-s possessions, plausible count and durations) as the reality check.
