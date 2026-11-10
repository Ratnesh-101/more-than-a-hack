"""One-command reproduction of the submission:

    python -m mtah.reproduce --team nirakshak [--data-dir D] [--out DIR]

Runs in order:
1. mtah.audit
2. mtah.run (twice, into a temp dir or --out, never overwriting submissions/ unless --out points there)
3. evaluation/validate_submission.py
4. analysis.real_checks
5. SHA-256 comparison between runs (and against submissions/<team>.csv.sha256 if present)

Prints PASS or FAIL per step and a final line with the SHA-256 hash.
Exits 1 on any hard failure; a real_checks FAIL gate is reported but does not fail reproduction.
Never reads evaluation/gt.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-command reproduction of the submission.")
    parser.add_argument("--team", required=True, help="exact team name")
    parser.add_argument("--data-dir", default=None, help="path to match data bundle")
    parser.add_argument("--out", default=None, help="folder for <team>.csv (default: temp directory)")
    parser.add_argument("--skip-audit", action="store_true", help="skip mtah.audit (for synthetic test bundles)")
    parser.add_argument("--synthetic", nargs="?", const=0, type=int, default=None,
                        help="run on synthetic bundle with SEED (default 0)")
    parser.add_argument("--verbose", action="store_true", help="stream step outputs")
    args = parser.parse_args(argv)

    synth_dir = None
    env = os.environ.copy()
    if args.synthetic is not None:
        import numpy as np
        from mtah.selfcheck import make_bundle
        synth_dir = tempfile.mkdtemp(prefix="mtah_synth_bundle_")
        synth_data = os.path.join(synth_dir, "data")
        os.makedirs(synth_data, exist_ok=True)
        make_bundle(synth_data, np.random.default_rng(args.synthetic))
        args.data_dir = synth_data
        args.skip_audit = True

    if args.data_dir:
        env["MTAH_DATA_DIR"] = os.path.abspath(args.data_dir)
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    try:
        # Step 1: mtah.audit
        if args.skip_audit:
            print("PASS  mtah.audit (skipped)")
        else:
            cmd_audit = [sys.executable, "-m", "mtah.audit"]
            if args.data_dir:
                cmd_audit.extend(["--data-dir", args.data_dir])
            p_audit = subprocess.run(cmd_audit, cwd=ROOT, env=env,
                                     capture_output=not args.verbose, text=True)
            if p_audit.returncode != 0:
                print("FAIL  mtah.audit")
                if not args.verbose and p_audit.stdout:
                    print(p_audit.stdout.strip())
                if not args.verbose and p_audit.stderr:
                    print(p_audit.stderr.strip(), file=sys.stderr)
                sys.exit(1)
            print("PASS  mtah.audit")

        tmp_dir = tempfile.mkdtemp(prefix="mtah_repro_")
        try:
            if args.out:
                run1_dir = os.path.abspath(args.out)
                os.makedirs(run1_dir, exist_ok=True)
            else:
                run1_dir = os.path.join(tmp_dir, "run1")
                os.makedirs(run1_dir, exist_ok=True)

            run2_dir = os.path.join(tmp_dir, "run2")
            os.makedirs(run2_dir, exist_ok=True)

            target_csv = os.path.join(run1_dir, "{}.csv".format(args.team))

            # Step 2: mtah.run (run 1)
            cmd_run1 = [sys.executable, "-m", "mtah.run", "--team", args.team, "--out", run1_dir]
            if args.data_dir:
                cmd_run1.extend(["--data-dir", args.data_dir])
            p_run1 = subprocess.run(cmd_run1, cwd=ROOT, env=env,
                                    capture_output=not args.verbose, text=True)
            if p_run1.returncode != 0:
                print("FAIL  mtah.run (run 1)")
                if not args.verbose and p_run1.stderr:
                    print(p_run1.stderr.strip(), file=sys.stderr)
                sys.exit(1)
            print("PASS  mtah.run (run 1)")
            sha1 = sha256_of(target_csv)

            # Step 2 (continued): mtah.run (run 2)
            cmd_run2 = [sys.executable, "-m", "mtah.run", "--team", args.team, "--out", run2_dir]
            if args.data_dir:
                cmd_run2.extend(["--data-dir", args.data_dir])
            p_run2 = subprocess.run(cmd_run2, cwd=ROOT, env=env,
                                    capture_output=not args.verbose, text=True)
            if p_run2.returncode != 0:
                print("FAIL  mtah.run (run 2)")
                if not args.verbose and p_run2.stderr:
                    print(p_run2.stderr.strip(), file=sys.stderr)
                sys.exit(1)
            print("PASS  mtah.run (run 2)")
            csv2 = os.path.join(run2_dir, "{}.csv".format(args.team))
            sha2 = sha256_of(csv2)

            # Step 3: evaluation/validate_submission.py
            cmd_val = [sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), target_csv]
            p_val = subprocess.run(cmd_val, cwd=ROOT, env=env,
                                   capture_output=not args.verbose, text=True)
            if p_val.returncode != 0:
                print("FAIL  evaluation/validate_submission.py")
                if not args.verbose and p_val.stdout:
                    print(p_val.stdout.strip())
                sys.exit(1)
            print("PASS  evaluation/validate_submission.py")

            # Step 4: analysis.real_checks
            cmd_rc = [sys.executable, "-m", "analysis.real_checks", "--csv", target_csv]
            p_rc = subprocess.run(cmd_rc, cwd=ROOT, env=env,
                                  capture_output=not args.verbose, text=True)
            if p_rc.returncode != 0:
                print("FAIL  analysis.real_checks (gate thresholds not met)")
            else:
                print("PASS  analysis.real_checks")

            # Step 5: compare SHA-256 determinism
            if sha1 != sha2:
                print("FAIL  determinism mismatch: run 1 ({}) != run 2 ({})".format(sha1, sha2))
                sys.exit(1)
            print("PASS  determinism (run 1 == run 2)")

            # Sidecar check if submissions/<team>.csv.sha256 exists
            sidecar_path = os.path.join(ROOT, "submissions", "{}.csv.sha256".format(args.team))
            if os.path.isfile(sidecar_path):
                with open(sidecar_path, "r", encoding="utf-8") as fh:
                    sidecar_text = fh.read()
                m = re.search(r"\b[0-9a-fA-F]{64}\b", sidecar_text)
                if m:
                    expected_sha = m.group(0).lower()
                    if sha1.lower() != expected_sha:
                        print("FAIL  submissions/{}.csv.sha256 mismatch: got {} want {}".format(
                            args.team, sha1.lower(), expected_sha))
                        sys.exit(1)
                    print("PASS  submissions/{}.csv.sha256 match".format(args.team))

            # Final line with hash
            print("sha256 {}".format(sha1))

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    finally:
        if synth_dir and os.path.isdir(synth_dir):
            shutil.rmtree(synth_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
