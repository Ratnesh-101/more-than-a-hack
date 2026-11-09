"""Submission entry point: a thin delegate to the champion pipeline in mtah.v2.run.

    python -m mtah.run --team TEAMNAME            # stream/cached tracking, write submission
    python -m mtah.run --team TEAMNAME --refresh  # rebuild the tracking cache
    python -m mtah.run --team TEAMNAME --data-dir D --out OTHER_FOLDER
    python evaluation/validate_submission.py submissions/TEAMNAME.csv

Writes exactly one file, submissions/<TEAM>.csv (or <out>/<TEAM>.csv). All logic lives in
mtah.v2.run; the names below are re-exported for callers of the old module.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.v2.run import main, safe_output, shape, write  # noqa: E402,F401

if __name__ == "__main__":
    main()
