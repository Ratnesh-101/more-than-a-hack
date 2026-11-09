# Security

## What this project is, and what can actually go wrong

`mtah/` is an **offline batch pipeline**: it reads local files from `data/`, writes one CSV, and
exits. It runs no server, opens no network connection, has no accounts and takes no input from
anyone but the person running it. There is no remote attack surface to defend.

The real risks for this repository are the ones that would get the team **disqualified or
embarrassed**:

| risk | impact | controls |
|---|---|---|
| Gated match data (or a renamed / derived copy) pushed to the public repo | licence breach, disqualification ([RULES.md](RULES.md)) | `.gitignore`, `mtah/guard.py` (path + content scan), git hooks |
| A credential (API key, token, private key) committed | account takeover; public repos are scraped within minutes | `.gitignore`, guard secret patterns, git hooks |
| Writing outside the project via a crafted CLI argument | overwrite arbitrary files | `run.safe_output()` whitelists `--team` and re-checks the resolved path |
| Loading a tampered or corrupt feature cache | code execution via pickle; silently wrong labels | `allow_pickle=False`, strict key/dtype/shape validation, atomic cache writes |
| Dependency drift or substitution | non-reproducible file (rule 3), malicious package | exact pins in `requirements-lock.txt` (see "Supply chain" below) |

## One-time setup (every clone, every teammate)

```
git config core.hooksPath .githooks
```

That enables two hooks, both running `mtah/guard.py`:

- **pre-commit**: scans what is *staged*, reading the index content, not the work tree.
- **pre-push**: scans every tracked file before anything leaves the machine.

Never use `--no-verify`. If the guard blocks something, unstage it (`git rm --cached <path>`).
If a finding is a false positive, fix the rule in `mtah/guard.py` and add a test.

Run it by hand any time:

```
python -m mtah.guard            # everything `git add -A` would commit
python -m mtah.guard --staged   # what is staged now
python -m mtah.guard --tracked  # everything already tracked
```

## What the guard blocks

- Anything under `data/` except `data/README.md`.
- Match-asset names (`BAR-ATH_tactical*`, `BAR-ATH_broadcast*`) anywhere in the tree.
- Private ground-truth names (`*_possession_gt.csv`, `*_gt_reference.csv`). The public 5-minute
  file is allowed.
- Video, caches and dumps: `.mp4 .npz .npy .pkl .parquet .feather .h5 .zip .gz …`
- Key material and credential files: `.pem .key .p12 .env .netrc .pypirc credentials.json …`
- Any file over 5 MB.
- **By content, so renaming a file does not get it through:**
  - 50 or more ATD tracking lines
  - 50 or more homography rows
  - 20 or more Smart Tagging instances
  - 20 or more ATD `<Player teamId=…>` entries
  - private keys, AWS/GitHub/Google/Slack/`sk-` tokens, and hard-coded `password=`/`token=`
    strings

Volume thresholds keep documentation that quotes one example row (DATA.md) and the synthetic
fixtures in `mtah/selfcheck.py` from being flagged.

## Hardening in the code

- `mtah/run.py`: `--team` must match `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`, must not be a Windows
  device name (`CON`, `NUL`, `COM1`, …), and the resolved output path must sit directly inside
  `submissions/` (or `evaluation/results/`). It is checked before the slow feature pass.
- `mtah/features.py`:
  - The cache is opened with `allow_pickle=False`, because unpickling can run code.
  - A cache is accepted only if it has exactly the expected keys, numeric dtypes and the right
    lengths and shapes. Anything else triggers a rebuild from the raw data.
  - Writes go to a `.partial.npz` file, then `os.replace` swaps it in.
- `mtah/selfcheck.py`:
  - It runs the official validator with `subprocess` and an argument list, so no shell is
    involved.
  - `--set` converts values with the existing param's type and never calls `eval`.
- No module makes network calls.

All of this is covered by `tests/test_security.py`.

## Supply chain

`requirements-lock.txt` pins our five direct dependencies exactly.
`requirements-freeze-windows.txt` is the full resolved environment.

**Not yet done, recommended before the final submission:** hash-pinning
(`pip install --require-hashes`) and a vulnerability scan (`pip-audit`). Both need network access
and a full resolve, so they were not run in this environment. `pip-audit` sends package names and
versions, never project code, to the PyPI/OSV advisory database.

## If something leaks anyway

1. **Do not just delete the file in a new commit.** It stays in history, and anyone who cloned
   or forked the repo still has it.
2. Make the repo private immediately, if it is already public.
3. Remove the file from history (`git filter-repo --path <file> --invert-paths`), then
   force-push. **This rewrites shared history, so coordinate with the whole team first.**
4. For a credential: **revoke and rotate it at the provider first**. Assume it was copied the
   moment it was pushed.
5. For match data: tell the organisers. Reporting it early is far better than having it found.

## Reporting a vulnerability

Open a private security advisory on the repository (GitHub → Security → Report a vulnerability)
or contact the team directly. Please do not open a public issue for anything involving data
or credentials.
