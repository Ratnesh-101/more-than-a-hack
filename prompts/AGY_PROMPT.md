# Prompt 1 — paste into Antigravity (analysis + docs; Kiro owns mtah/)

```
Read .kiro/steering/project.md and all the *.md files at the repo root. Kiro owns the pipeline
code in mtah/ — do NOT edit mtah/, starter/, evaluation/ or games.py. Never commit/push unless I say COMMIT.

Your jobs:

A) docs/TECHNICAL.md — draft the technical document required by SUBMISSION.md, from the code in mtah/:
   approach (features → 3-state HMM/Viterbi, why it targets spellF1), assets used and dropped,
   training data (none — no labels used; public GT only for pipeline correctness),
   possession rules implemented (loose ball keeps last team, dead ball, restarts), limitations,
   reproduction steps. Keep it under ~4 pages, factual, no hype. Leave TODO markers where results go.

B) analysis/diagnostics.py — once data/ exists: plots saved to analysis/out/ (gitignored) showing,
   for a chosen frame range: ball visibility, d_home/d_away, mean_speed, our predicted labels.
   Read features via mtah.features.load(). Also print: % ball visible per half, distribution of
   our possession durations, longest ball-missing stretches.
   NEVER use evaluation/gt for anything other than overlaying it on the first 5 minutes to check alignment.

C) README section "Reproduce": exact commands from clean clone to submissions/nirakshak.csv.

Report what you created, and anything in mtah/ that looks like a bug (describe it — don't fix it).
```
