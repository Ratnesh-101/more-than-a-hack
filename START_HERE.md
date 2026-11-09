# START HERE

Official repo (metrica-sports/more-than-a-hack-2027) + our v1 pipeline in `mtah/`.

## Deadlines (IST)
- **Nov 10** — all teammates register with the SAME team name
- Data link email — within 72 h of Nov 8
- Submission form — by Nov 14
- **Submit by Nov 18, 04:29 IST** (Nov 17 23:59 CET)

## Who does what
- **Kiro** → `prompts/KIRO_PROMPT.md` (setup, run pipeline, model improvements). Steering is in `.kiro/steering/`.
- **Antigravity** → `prompts/AGY_PROMPT.md` (technical doc, diagnostics, README).

## Commands
```
pip install -r requirements-starter.txt
pip install -r requirements-lock.txt for exact reproduction (pinned direct deps; full Windows env: requirements-freeze-windows.txt)
python -m mtah.selfcheck                      # works without data
python -m mtah.audit                          # FIRST after data arrives: parse vs DATA.md figures
python -m mtah.run --team nirakshak              # needs data/tactical/
python evaluation/validate_submission.py submissions/nirakshak.csv
python evaluation/run_evaluation.py
```

## Submit (3 things, one Google form)
1. `nirakshak.csv`  2. public GitHub repo URL (NO data/)  3. `nirakshak.pdf` technical doc
