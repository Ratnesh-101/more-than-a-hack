"""Tests for mtah.reproduce."""

import os
import subprocess
import sys
import tempfile
import numpy as np

from mtah.selfcheck import make_bundle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_reproduce_synthetic():
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "data")
        out = os.path.join(tmp, "out")
        os.makedirs(data)
        os.makedirs(out)
        make_bundle(data, np.random.default_rng(0))

        cmd = [sys.executable, "-m", "mtah.reproduce", "--team", "test",
               "--data-dir", data, "--out", out, "--skip-audit"]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert p.returncode == 0, "reproduce failed: {}\n{}".format(p.stdout, p.stderr)
        assert "PASS  mtah.audit (skipped)" in p.stdout
        assert "PASS  mtah.run (run 1)" in p.stdout
        assert "PASS  mtah.run (run 2)" in p.stdout
        assert "PASS  evaluation/validate_submission.py" in p.stdout
        assert ("PASS  analysis.real_checks" in p.stdout or
                "FAIL  analysis.real_checks" in p.stdout)
        assert "PASS  determinism (run 1 == run 2)" in p.stdout
        assert "sha256 36665403938ea340cb8f81fd9c650f7c35d9f577987bed4b718e4e1505c3001a" in p.stdout
        assert os.path.isfile(os.path.join(out, "test.csv"))


def test_reproduce_audit_failure_on_missing_data():
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "empty_data")
        out = os.path.join(tmp, "out")
        os.makedirs(data)
        os.makedirs(out)

        # Without --skip-audit on empty/invalid data, mtah.audit must fail and exit 1
        cmd = [sys.executable, "-m", "mtah.reproduce", "--team", "test",
               "--data-dir", data, "--out", out]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert p.returncode == 1
        assert "FAIL  mtah.audit" in p.stdout
