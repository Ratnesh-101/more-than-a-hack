"""Final pre-submission checklist. Prints PASS / FAIL / INFO per item; exit 1 on any FAIL.

    python -m analysis.preflight --team nirakshak [--repo-url https://github.com/OWNER/REPO]
                                 [--sha256 HEX] [--max-pages 6]

Run it on the exact files you are about to upload. It never reads evaluation/gt (the only GT
related thing it does is look at file NAMES in the tree to make sure no full-match GT is present),
never touches the network and writes nothing.

Determinism evidence: mtah.v2.run prints "sha256 <hex>" but writes no sidecar. Either pass that
value as --sha256, or save it to submissions/<team>.csv.sha256 (first hex token in the file).

Items: 1 CSV validates | 2 SHA-256 | 3 X only outside graded halves | 4 PDF + document markers |
5 repo hygiene | 6 repo readiness | 7 reproduction commands agree | 8 repo URL | 9 reminder.
"""

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SKIP_DIRS = {".venv", "venv", "env", "__pycache__", ".git", ".pytest_cache", "node_modules"}
RESULTS = []   # (status, item, message)


def rec(status, item, msg):
    RESULTS.append((status, item, msg))
    print("[{}] {}: {}".format(status, item, msg))


def walk(skip_extra=()):
    """Yield (relative posix path, absolute path) for every file under ROOT, skipping venvs."""
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT).replace(os.sep, "/")
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and not (rel_dir + "/" + d).lstrip("./") in skip_extra]
        for name in filenames:
            ab = os.path.join(dirpath, name)
            yield os.path.relpath(ab, ROOT).replace(os.sep, "/"), ab


def read_text(path):
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", "replace")


def run(cmd):
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.communicate()[0].decode("utf-8", "replace")
    return p.returncode, out


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- 1 --------------------------------------------------------------------------------------
def check_csv(team, csv_path):
    item = "1 submission CSV"
    want = team + ".csv"
    d = os.path.join(ROOT, "submissions")
    names = sorted(os.listdir(d)) if os.path.isdir(d) else []
    if not os.path.isfile(csv_path):
        others = [n for n in names if n.lower().endswith(".csv")]
        rec("FAIL", item, "submissions/{} not found (csv files present: {})".format(
            want, others or "none"))
        return False
    if want not in names:   # case-insensitive filesystems
        rec("FAIL", item, "file name is not exactly {} (found {})".format(
            want, [n for n in names if n.lower() == want.lower()]))
        return False
    code, out = run([sys.executable, os.path.join("evaluation", "validate_submission.py"),
                     os.path.join("submissions", want)])
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    summary = "; ".join(ln.strip() for ln in lines[:12])
    if code != 0 or any(ln.lstrip().startswith("FAIL") for ln in lines):
        rec("FAIL", item, "validate_submission.py exit {}: {}".format(code, summary))
        return False
    rec("PASS", item, "validate_submission.py OK")
    for ln in lines:
        print("        " + ln.rstrip())
    return True


# ---- 2 --------------------------------------------------------------------------------------
def check_sha(csv_path, sha_arg):
    item = "2 determinism SHA-256"
    if not os.path.isfile(csv_path):
        rec("FAIL", item, "no CSV to hash")
        return
    actual = sha256_of(csv_path)
    expected, src = None, None
    if sha_arg:
        expected, src = sha_arg, "--sha256"
    else:
        side = csv_path + ".sha256"
        if os.path.isfile(side):
            m = re.search(r"\b[0-9a-fA-F]{64}\b", read_text(side))
            if m:
                expected, src = m.group(0), os.path.relpath(side, ROOT)
    if not expected:
        rec("FAIL", item, "no recorded SHA-256 (actual {}). mtah.v2.run prints 'sha256 <hex>' but "
            "writes no sidecar: rerun it on a clean checkout and pass --sha256 HEX".format(actual))
        return
    if expected.strip().lower() == actual:
        rec("PASS", item, "CSV sha256 matches {} ({})".format(src, actual))
    else:
        rec("FAIL", item, "mismatch: recorded {} ({}) vs actual {}".format(expected, src, actual))


# ---- 3 --------------------------------------------------------------------------------------
def check_x(csv_path):
    item = "3 X only outside graded halves"
    if not os.path.isfile(csv_path):
        rec("FAIL", item, "no CSV")
        return
    try:
        from games import GAME, half_video_blocks
        blocks = [(h, lo, hi) for h, lo, hi in half_video_blocks(GAME)]
        labels = []
        with open(csv_path, newline="") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for row in rd:
                labels.append(row[1].strip().upper() if len(row) > 1 else "")
    except Exception as e:   # unreadable or games.py failure
        rec("FAIL", item, "could not read blocks/CSV: {!r}".format(e))
        return
    n = len(labels)
    inside = [False] * n
    msgs, bad = [], False
    for h, lo, hi in blocks:
        hi_c = min(hi, n - 1)
        xs = sum(1 for i in range(lo, hi_c + 1) if labels[i] == "X")
        for i in range(lo, hi_c + 1):
            inside[i] = True
        msgs.append("half {} [{}..{}]: {} X of {} frames".format(h, lo, hi, xs, hi_c - lo + 1))
        if xs:
            bad = True
    outside_nonx = sum(1 for i in range(n) if not inside[i] and labels[i] != "X")
    msgs.append("{} non-X frames outside the blocks".format(outside_nonx))
    if outside_nonx:
        bad = True
    rec("FAIL" if bad else "PASS", item, "; ".join(msgs))


# ---- 4 --------------------------------------------------------------------------------------
def pdf_pages(path):
    with open(path, "rb") as fh:
        data = fh.read()
    n = len(re.findall(rb"/Type\s*/Page(?![A-Za-z])", data))
    if n:
        return n
    counts = [int(c) for c in re.findall(rb"/Count\s+(\d+)", data)]
    return max(counts) if counts else 0


def marker_lines(text, pattern, ignore_code_spans=False):
    """Line numbers matching pattern; optionally ignore text inside `inline code` spans."""
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        if ignore_code_spans:
            ln = re.sub(r"`[^`]*`", "", ln)
        if re.search(pattern, ln):
            out.append(i)
    return out


def check_pdf_and_doc(team, max_pages):
    item = "4 PDF + document"
    pdf = os.path.join(ROOT, "docs", "out", team + ".pdf")
    md = os.path.join(ROOT, "docs", "TECHNICAL.md")
    if not os.path.isfile(pdf):
        rec("FAIL", item, "docs/out/{}.pdf missing (python docs/build_pdf.py --team {})".format(
            team, team))
    else:
        problems = []
        if os.path.isfile(md) and os.path.getmtime(pdf) < os.path.getmtime(md):
            problems.append("PDF is OLDER than docs/TECHNICAL.md (rebuild it)")
        pages = pdf_pages(pdf)
        if pages < 1 or pages > max_pages:
            problems.append("page count {} not in 1..{}".format(pages, max_pages))
        if problems:
            rec("FAIL", item, "; ".join(problems) + " (pages={})".format(pages))
        else:
            rec("PASS", item, "docs/out/{}.pdf exists, newer than TECHNICAL.md, {} page(s)".format(
                team, pages))
    bad = False
    for fname in ("docs/TECHNICAL.md", "README.md"):
        path = os.path.join(ROOT, fname)
        if not os.path.isfile(path):
            rec("FAIL", "4 markers in " + fname, "file missing")
            bad = True
            continue
        text = read_text(path)
        for label, pat in (("TODO(result)", r"TODO\(result\)"), ("UNJUSTIFIED - ask lead", r"UNJUSTIFIED\s*-\s*ask lead"),
                           ("TODO(lead)", r"TODO\(lead\)")):
            lines = marker_lines(text, pat, ignore_code_spans=(label != "TODO(result)" and label != "TODO(lead)"))
            if lines:
                bad = True
                rec("FAIL", "4 markers in " + fname, "{} x{} at line(s) {}".format(
                    label, len(lines), ", ".join(map(str, lines))))
    if not bad:
        rec("PASS", "4 markers", "no TODO(result), UNJUSTIFIED or TODO(lead) in TECHNICAL.md/README.md")


# ---- 5 --------------------------------------------------------------------------------------
def check_hygiene():
    item = "5 repo hygiene"
    code, out = run([sys.executable, "-m", "mtah.guard"])
    tail = " | ".join(ln.strip() for ln in out.strip().splitlines()[-4:])
    rec("PASS" if code == 0 else "FAIL", item + " (mtah.guard)", "exit {}: {}".format(code, tail))

    files = list(walk())
    bad_data = [r for r, _ in files if r.startswith("data/") and r != "data/README.md"
                and not r.startswith("data/cache/")]
    cache = [r for r, _ in files if r.startswith("data/cache/")]
    if bad_data:
        rec("FAIL", item + " (data/)", "files other than README.md: {}".format(bad_data[:10]))
    else:
        rec("PASS", item + " (data/)", "only data/README.md" +
            (" (+{} git-ignored file(s) in data/cache/)".format(len(cache)) if cache else ""))
    mp4 = [r for r, _ in files if r.lower().endswith(".mp4")]
    rec("FAIL" if mp4 else "PASS", item + " (*.mp4)", str(mp4[:10]) if mp4 else "none")
    pdfs = [r for r, _ in files if r.lower().endswith(".pdf")]
    rec("INFO", item + " (*.pdf)", "PDFs in tree {} are git-ignored; the PDF is uploaded to the "
        "form, NOT the repo".format(pdfs))

    gi_path = os.path.join(ROOT, ".gitignore")
    gi = read_text(gi_path).splitlines() if os.path.isfile(gi_path) else []
    gi = [ln.strip() for ln in gi]
    missing = [p for p in ("data/*", "*.pdf", "evaluation/results/*.csv") if p not in gi]
    rec("FAIL" if missing else "PASS", item + " (.gitignore)",
        "missing patterns {}".format(missing) if missing else "data/*, *.pdf, "
        "evaluation/results/*.csv present")

    pub = [r for r, _ in files if r.startswith("evaluation/gt/") and r.endswith("_public.csv")]
    rec("PASS" if pub else "FAIL", item + " (public GT)",
        "present: {}".format(pub) if pub else "evaluation/gt/*_public.csv missing")
    gt_re = re.compile(r"(_possession_gt(\.csv|_meta\.json|_reference\.csv)$)|^possession_by_frame\.pkl$")
    full = [r for r, _ in files if gt_re.search(os.path.basename(r)) and not r.endswith("_public.csv")
            and "_public" not in os.path.basename(r)]
    rec("FAIL" if full else "PASS", item + " (no full-match GT)",
        "FOUND {}".format(full) if full else "none in tree")


# ---- 6 --------------------------------------------------------------------------------------
def check_readiness():
    item = "6 repo readiness"
    readme = os.path.join(ROOT, "README.md")
    txt = read_text(readme) if os.path.isfile(readme) else ""
    has = bool(re.search(r"^#+\s*Reproduce", txt, re.M))
    rec("PASS" if has else "FAIL", item + " (README Reproduce)",
        "section found" if has else "no '## Reproduce' heading in README.md")

    req = os.path.join(ROOT, "requirements.txt")
    if not os.path.isfile(req):
        rec("FAIL", item + " (requirements.txt)", "missing")
    else:
        unpinned = [(i, ln.strip()) for i, ln in enumerate(read_text(req).splitlines(), 1)
                    if ln.strip() and not ln.strip().startswith("#") and "==" not in ln]
        rec("FAIL" if unpinned else "PASS", item + " (requirements.txt)",
            "unpinned: {}".format(unpinned) if unpinned else "every requirement uses ==")

    # patterns built from pieces so this file does not flag itself
    pats = [("C:" + "\\\\Users", re.compile(r"[A-Za-z]:\\{1,2}Users", re.I)),
            ("/" + "Users/", re.compile(r"/" + "Users/")),
            ("/" + "home/", re.compile(r"/" + "home/"))]
    lint_def = re.compile(r"""^\s*\(?\s*[rR]["']|re\.(compile|search|match|findall|sub)\(""")
    hits = []
    for rel, ab in walk():
        if rel.startswith(("data/", "docs/out/", "analysis/out/")):
            continue
        if not rel.endswith((".py", ".md", ".txt")) or rel == "analysis/preflight.py":
            continue
        for i, ln in enumerate(read_text(ab).splitlines(), 1):
            if lint_def.search(ln):
                continue   # a regex literal that defines a path linter, not a real path
            for name, rx in pats:
                if rx.search(ln):
                    hits.append("{}:{} ({})".format(rel, i, name))
                    break
    rec("FAIL" if hits else "PASS", item + " (absolute user paths)",
        "{} hit(s): {}{}".format(len(hits), hits[:15], " ..." if len(hits) > 15 else "")
        if hits else "none in .py/.md/.txt")

    words = re.compile(r"\b(" + "TO" + "DO|FIX" + "ME|X" + "XX)\b")
    hits = []
    for rel, ab in walk():
        if (rel.startswith("mtah/v2/") or rel.startswith("analysis/")) and rel.endswith(".py") \
                and rel != "analysis/preflight.py":
            for i, ln in enumerate(read_text(ab).splitlines(), 1):
                if words.search(ln):
                    hits.append("{}:{}".format(rel, i))
    rec("FAIL" if hits else "PASS", item + " (open markers in mtah/v2, analysis)",
        "{}".format(hits[:20]) if hits else "none")


# ---- 7 --------------------------------------------------------------------------------------
def repro_set(path):
    if not os.path.isfile(path):
        return None
    cmds = set()
    for m in re.finditer(r"python3?\s+-m\s+mtah\.run\s+--team\b[^\n`#]*", read_text(path)):
        cmds.add(re.sub(r"\s+", " ", m.group(0)).strip().replace("python3 ", "python "))
    return cmds


def check_repro(team):
    item = "7 reproduction commands"
    a = repro_set(os.path.join(ROOT, "README.md"))
    b = repro_set(os.path.join(ROOT, "docs", "TECHNICAL.md"))
    want = "python -m mtah.run --team " + team
    if a is None or b is None:
        rec("FAIL", item, "README.md or docs/TECHNICAL.md missing")
        return
    probs = []
    if a != b:
        probs.append("README {} != TECHNICAL {}".format(sorted(a), sorted(b)))
    for name, s in (("README", a), ("TECHNICAL", b)):
        if want not in s:
            probs.append("{} lacks '{}'".format(name, want))
    rec("FAIL" if probs else "PASS", item, "; ".join(probs) if probs
        else "same set in README and TECHNICAL: {}".format(sorted(a)))


# ---- 8 --------------------------------------------------------------------------------------
def check_url(url):
    item = "8 repo URL"
    if not url:
        rec("INFO", item, "not given (pass --repo-url https://github.com/OWNER/REPO)")
        return
    ok = re.match(r"^https://github\.com/[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]+/?$", url)
    ph = re.search(r"[<>\[\]{}]|repository-url|owner|your|example|repo$|username|OWNER", url, re.I)
    if ok and not ph and not url.rstrip("/").endswith(".git"):
        rec("PASS", item, "{} looks like a public GitHub repo URL (reachability NOT checked: "
            "open it in a private window)".format(url))
    else:
        rec("FAIL", item, "{!r} is not a plausible https://github.com/<owner>/<repo> URL "
            "(or is a placeholder / ends in .git)".format(url))


# ---- main -----------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Final pre-submission checklist (exit 1 on any FAIL).")
    ap.add_argument("--team", required=True, help="exact team name (file names derive from it)")
    ap.add_argument("--repo-url", default=None, help="public GitHub URL to sanity-check")
    ap.add_argument("--sha256", default=None,
                    help="SHA-256 printed by mtah.v2.run, compared with the CSV's")
    ap.add_argument("--max-pages", type=int, default=6, help="PDF page limit (default 6)")
    args = ap.parse_args(argv)
    team = args.team
    if not re.match(r"^[A-Za-z0-9._-]+$", team):
        ap.error("team name must be a plain file-name stem")
    csv_path = os.path.join(ROOT, "submissions", team + ".csv")

    print("Preflight for team {!r}".format(team))
    ok = check_csv(team, csv_path)
    check_sha(csv_path, args.sha256)
    if ok:
        check_x(csv_path)
    else:
        rec("FAIL", "3 X only outside graded halves", "skipped: CSV invalid or missing")
    check_pdf_and_doc(team, args.max_pages)
    check_hygiene()
    check_readiness()
    check_repro(team)
    check_url(args.repo_url)

    nfail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    print("\nSummary: {} PASS, {} FAIL, {} INFO".format(
        sum(1 for s, _, _ in RESULTS if s == "PASS"), nfail,
        sum(1 for s, _, _ in RESULTS if s == "INFO")))
    print("\n[9] UPLOAD REMINDER (one Google Form, one response per team; last one counts)")
    print("    1. possession file : submissions/{}.csv   (validate THIS exact file)".format(team))
    print("    2. public repo URL : {}".format(args.repo_url or "https://github.com/<owner>/<repo>"))
    print("    3. technical doc   : docs/out/{0}.pdf  (upload as {0}.pdf; not in the repo)".format(team))
    print("    DEADLINE: 17 Nov 2026 23:59 CET  =  18 Nov 2026 04:29 IST")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
