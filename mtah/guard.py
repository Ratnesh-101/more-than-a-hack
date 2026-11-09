"""Leak guard: refuse to let gated match data, derived exports or secrets reach the public repo.

    python -m mtah.guard            # scan everything `git add -A` would commit (tracked + untracked,
                                    # minus .gitignore) -- run before every push
    python -m mtah.guard --staged   # scan only what is staged, reading the INDEX content
                                    # (what the pre-commit hook in .githooks/ runs)
    python -m mtah.guard --tracked  # scan every tracked file (what the pre-push hook runs)

Why: the repository must be public (SUBMISSION.md) and the match data must never be in it
(RULES.md, DATA.md terms). .gitignore stops the obvious paths; this also catches renamed copies,
derived exports (feature caches, frame-level tracking dumps) and credentials, by content.
Exit code 1 on any finding. No network, no writes.
"""

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_BYTES = 5_000_000          # the largest legit files (baseline/GT CSVs) are ~2 MB
SCAN_BYTES = 2_000_000         # content-scan this much of each file

ALLOWED_DATA = {"data/README.md"}

# match-data / derived-data file types that have no business in this repo
BLOCKED_EXT = {
    ".mp4", ".mov", ".avi", ".mkv",                       # video
    ".npz", ".npy", ".pkl", ".pickle", ".joblib",         # feature caches / pickles
    ".h5", ".hdf5", ".parquet", ".feather", ".arrow",     # dataframe dumps
    ".gzip", ".gz", ".zip", ".7z", ".rar", ".tar",        # bundles
    ".pl3container",
    ".pem", ".key", ".p12", ".pfx", ".keystore", ".jks",  # key material
}
BLOCKED_NAMES = {".env", ".netrc", ".pypirc", "id_rsa", "id_ed25519", "credentials.json",
                 "service-account.json", "possession_by_frame.pkl"}
BLOCKED_PATH_RE = [
    (re.compile(r"(^|/)data/(?!README\.md$)"), "file under data/ (gated match data)"),
    (re.compile(r"BAR-ATH_(tactical|broadcast)"), "match-asset file name"),
    (re.compile(r"_possession_gt(_reference|_meta)?\.(csv|json)$"),
     "private full-match ground-truth file name"),
    (re.compile(r"(^|/)\.env(\.|$)"), "dotenv file"),
]
PUBLIC_GT = re.compile(r"^evaluation/gt/BAR-ATH_possession_gt_public(_meta)?\.(csv|json)$")

SECRET_RE = [
    (re.compile(rb"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"), "private key"),
    (re.compile(rb"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "GitHub token"),
    (re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{50,}\b"), "GitHub fine-grained token"),
    (re.compile(rb"\bAIza[0-9A-Za-z_\-]{35}\b"), "Google API key"),
    (re.compile(rb"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "Slack token"),
    (re.compile(rb"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{32,}\b"), "API secret key (sk-...)"),
    (re.compile(rb"(?i)\b(?:password|passwd|secret|api_?key|token)\s*[:=]\s*['\"][^'\"\s]{12,}"),
     "hard-coded credential"),
]

# Content signatures of the licensed assets, so a renamed copy is still caught. Each needs VOLUME
# (a real leak is thousands of rows; docs and synthetic fixtures quote one or two), which keeps
# DATA.md's example row and selfcheck's synthetic template from tripping it.
ATD_LINE = re.compile(rb"^\d+:[-0-9.,;NaN]+:[-0-9.,NaN]*\r?$", re.M)
HOMOGRAPHY_ROW = re.compile(rb"^\d+(?:,[-0-9.eE+NaN]+){9},[-0-9.eE+NaN]+,[01]\r?$", re.M)
DATA_SIGNATURES = [   # (pattern, minimum matches, what it is)
    (ATD_LINE, 50, "ATD tracking lines (frame:players:ball)"),
    (HOMOGRAPHY_ROW, 50, "homography export rows"),
    (re.compile(rb"<instance>", re.I), 20, "Smart Tagging / SportsCode XML"),
    (re.compile(rb"<Player[^>]+teamId=", re.I), 20, "ATD metadata XML"),
]


def _norm(path):
    return path.replace("\\", "/")


def check_path(path):
    """Findings from the path alone."""
    p = _norm(path)
    if p in ALLOWED_DATA or PUBLIC_GT.match(p):
        return []
    out = []
    name = p.rsplit("/", 1)[-1]
    ext = os.path.splitext(name)[1].lower()
    if ext in BLOCKED_EXT:
        out.append("blocked file type {}".format(ext))
    if name.lower() in BLOCKED_NAMES:
        out.append("blocked file name {}".format(name))
    out += [why for rx, why in BLOCKED_PATH_RE if rx.search(p)]
    return out


def check_content(path, blob, size):
    """Findings from size and content (blob = first SCAN_BYTES bytes)."""
    p = _norm(path)
    out = []
    if size > MAX_BYTES:
        out.append("large file ({:.1f} MB > {:.0f} MB)".format(size / 1e6, MAX_BYTES / 1e6))
    if p == "mtah/guard.py":            # this file defines the patterns
        return out
    out += ["possible secret: " + why for rx, why in SECRET_RE if rx.search(blob)]
    if p in ALLOWED_DATA or PUBLIC_GT.match(p):
        return out
    out += ["content looks like " + why for rx, k, why in DATA_SIGNATURES
            if len(rx.findall(blob)) >= k]
    return out


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def candidate_files(staged, tracked=False):
    if staged:
        raw = _git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    elif tracked:
        raw = _git("ls-files", "-z", "--cached")
    else:
        raw = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    return [p for p in raw.decode("utf-8", "replace").split("\0") if p]


def read(path, staged):
    """(first SCAN_BYTES bytes, full size). Staged mode reads the index, not the work tree."""
    if staged:
        size = int(_git("cat-file", "-s", ":" + path).strip() or 0)
        return _git("cat-file", "blob", ":" + path)[:SCAN_BYTES], size
    full = os.path.join(ROOT, path)
    size = os.path.getsize(full)
    with open(full, "rb") as fh:
        return fh.read(SCAN_BYTES), size


def scan(paths, staged=False):
    findings = []
    for path in paths:
        hits = check_path(path)
        try:
            blob, size = read(path, staged)
            hits += check_content(path, blob, size)
        except (OSError, subprocess.CalledProcessError) as err:
            if not os.path.exists(os.path.join(ROOT, path)) and not staged:
                continue                 # deleted in work tree
            hits.append("unreadable: {}".format(err))
        findings += [(path, h) for h in hits]
    return findings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--staged", action="store_true", help="scan the index only (pre-commit)")
    ap.add_argument("--tracked", action="store_true",
                    help="scan every tracked file (pre-push)")
    args = ap.parse_args(argv)
    try:
        paths = candidate_files(args.staged, args.tracked)
    except (OSError, subprocess.CalledProcessError) as err:
        print("GUARD ERROR: cannot list files via git ({})".format(err))
        return 2
    findings = scan(paths, args.staged)
    for path, why in findings:
        print("  BLOCK  {:<60} {}".format(path, why))
    print("GUARD {}  ({} files scanned, {} findings)".format(
        "OK" if not findings else "FAILED", len(paths), len(findings)))
    if findings:
        print("Remove the file(s) from the commit (git rm --cached <path>) -- do not just\n"
              "rename them. If a finding is a false positive, fix the rule in mtah/guard.py.")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
