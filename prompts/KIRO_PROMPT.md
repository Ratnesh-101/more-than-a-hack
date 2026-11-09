# Prompt 1 — paste into Kiro now (setup + smoke test)

```
Read .kiro/steering/project.md, README.md, CHALLENGE.md, EVALUATION.md, RULES.md and DATA.md first.

Setup:
1. Create a Python 3.11 venv in .venv and install requirements-starter.txt.
2. Run: python evaluation/run_evaluation.py   (scores the 4 official baselines on the public window)
3. Run: python -m mtah.selfcheck   (synthetic smoke test of our pipeline in mtah/)
4. If selfcheck fails, fix bugs in mtah/ ONLY. Do not touch starter/, evaluation/ or games.py.
5. Init a git repo locally (do not push, do not commit until I say COMMIT).

Report back: exact command outputs (baseline scoreboard + selfcheck lines), any fixes you made and why.
```

# Prompt 2 — paste into Kiro AFTER the data email arrives and it's unzipped into data/

```
The match data is now in data/tactical/. Team name is: nirakshak (exact, lowercase)

1. Run: python -m mtah.run --team nirakshak
   (first run parses the ATD and caches features; takes a few minutes)
2. Run: python evaluation/validate_submission.py submissions/nirakshak.csv
3. Run: python evaluation/run_evaluation.py   (sanity check only — public window is NOT a tuning set)
4. Also run: python starter/baseline.py, then re-run the evaluation so baseline_smooth50 is on the board next to ours.

Report: label counts, possession count + mean duration over the full match, the scoreboard rows
for ours vs baseline_smooth50, and anything that looks wrong (alignment, all-D halves, etc.).
Do NOT change PARAMS in mtah/decode.py based on the public-window score.
```

# Prompt 3 — v2 improvements (one at a time, I'll give the go-ahead per item)

```
Implement improvement <N> from the list below in mtah/, behind a flag so v1 stays reproducible.
Justify every constant from football knowledge or from the data's own statistics (NOT from the public GT).
After each: regenerate, validate, report possession count/mean duration and public-window scores side by side.

1. Ball gap filling: interpolate ball position across short gaps (<= 0.5 s) when both ends are close in space.
2. Smart Tagging priors: set-piece intervals (Type = Throw-in/Corner/Goal Kick/Free Kick/Kick Off) →
   boost D just before the restart frame, and boost the tagged Team just after. Team "N/A" = no team boost.
3. Control proxy: a team "gains control" only if the same team stays nearest for >= ~0.5 s; use this as stronger emission evidence than a single frame.
4. Dead-ball detector: ball out of bounds, ball static + players slow, long ball-missing stretch with low mean speed.
5. Abstain with X inside long ball-missing stretches when the decoder's evidence is flat (free on spellF1, see EVALUATION.md).
```
