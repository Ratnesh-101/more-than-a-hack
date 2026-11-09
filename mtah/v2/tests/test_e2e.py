"""End-to-end: synthetic ATD bundle -> python -m mtah.v2.run -> official validator.

    python -m mtah.v2.tests.test_e2e

Writes only inside a temp directory (--data-dir and --out both point there).
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

from mtah.selfcheck import make_bundle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        data, out = os.path.join(tmp, "data"), os.path.join(tmp, "sub")
        os.makedirs(data)
        make_bundle(data, np.random.default_rng(0))
        run = [sys.executable, "-m", "mtah.v2.run", "--team", "test", "--data-dir", data,
               "--out", out]
        subprocess.check_call(run, cwd=ROOT)
        first = open(os.path.join(out, "test.csv"), "rb").read()
        subprocess.check_call(run + ["--refresh"], cwd=ROOT)
        assert first == open(os.path.join(out, "test.csv"), "rb").read(), "file not byte-identical"
        rc = subprocess.call([sys.executable, os.path.join(ROOT, "evaluation",
                                                           "validate_submission.py"),
                              os.path.join(out, "test.csv")], cwd=ROOT)
        assert rc == 0, "validate_submission failed"
        print("E2E OK")


if __name__ == "__main__":
    main()
