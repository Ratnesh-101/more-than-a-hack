"""analysis/preflight.py checks only what would be published (git ls-files --cached --others
--exclude-standard). Each test builds a throwaway git repo in a temp dir; the real repo is never
touched, and nothing is committed (a staged file is already "tracked" for `--cached`), so no git
identity or config is needed.

    python -m pytest tests -q -p no:cacheprovider
"""

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import preflight  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

# Built from pieces so this file itself holds no absolute user path for preflight to find.
USER_PATH = "C:" + "\\" + "Users" + "\\" + "someone" + "\\" + "data"


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("data/*\n!data/README.md\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "README.md").write_text("put the bundle here\n")
    (tmp_path / "data" / "tactical").mkdir()
    (tmp_path / "data" / "tactical" / "match_atd.txt").write_text("1:0.5,0.5:NaN\n")
    git(tmp_path, "add", ".gitignore", "data/README.md")
    return tmp_path


def status_of(item_suffix):
    rows = [s for s, item, _ in preflight.RESULTS if item.endswith(item_suffix)]
    assert len(rows) == 1, preflight.RESULTS
    return rows[0]


def data_status(repo):
    preflight.RESULTS.clear()
    preflight.hygiene_file_checks(preflight.published_files(str(repo)))
    return status_of("(data/)")


def published(repo):
    return [rel for rel, _ in preflight.published_files(str(repo))]


# (a) an ignored data file is not published -> PASS
def test_ignored_data_file_passes(repo):
    assert "data/tactical/match_atd.txt" not in published(repo)
    assert data_status(repo) == "PASS"


# (b) the same file force-added is published -> FAIL
def test_force_added_data_file_fails(repo):
    git(repo, "add", "-f", "data/tactical/match_atd.txt")
    assert "data/tactical/match_atd.txt" in published(repo)
    assert data_status(repo) == "FAIL"


# (c) a data file that no rule ignores is published -> FAIL
def test_non_ignored_data_file_fails(repo):
    (repo / ".gitignore").write_text("data/cache/\n")      # data/* no longer ignored
    (repo / "data" / "x.txt").write_text("leak\n")
    assert "data/x.txt" in published(repo)
    assert data_status(repo) == "FAIL"


# (d) a file excluded via .git/info/exclude holding an absolute user path -> PASS
def test_excluded_file_with_user_path_passes(repo):
    (repo / "HANDOFF.md").write_text("notes in {}\n".format(USER_PATH))
    with open(repo / ".git" / "info" / "exclude", "a") as fh:
        fh.write("HANDOFF.md\n")
    assert "HANDOFF.md" not in published(repo)
    assert preflight.absolute_path_hits(preflight.published_files(str(repo))) == []


# (e) a tracked file holding an absolute user path -> FAIL
def test_tracked_file_with_user_path_fails(repo):
    (repo / "NOTES.md").write_text("first line\nsee {}\n".format(USER_PATH))
    git(repo, "add", "NOTES.md")
    hits = preflight.absolute_path_hits(preflight.published_files(str(repo)))
    assert hits and hits[0].startswith("NOTES.md:2")


def test_untracked_but_not_ignored_file_with_user_path_also_fails(repo):
    (repo / "draft.md").write_text(USER_PATH + "\n")        # never added, not ignored
    assert "draft.md" in published(repo)
    assert preflight.absolute_path_hits(preflight.published_files(str(repo)))


def test_tracked_pdf_and_mp4_fail_ignored_ones_pass(repo):
    (repo / ".gitignore").write_text("data/*\n!data/README.md\n*.pdf\n")
    (repo / "doc.pdf").write_bytes(b"%PDF-1.4")
    (repo / "clip.mp4").write_bytes(b"\x00")
    preflight.RESULTS.clear()
    preflight.hygiene_file_checks(preflight.published_files(str(repo)))
    assert status_of("(*.pdf)") == "PASS"      # ignored
    assert status_of("(*.mp4)") == "FAIL"      # not ignored
    git(repo, "add", "-f", "doc.pdf")
    preflight.RESULTS.clear()
    preflight.hygiene_file_checks(preflight.published_files(str(repo)))
    assert status_of("(*.pdf)") == "FAIL"      # force-added


def test_published_full_match_gt_fails(repo):
    (repo / "evaluation" / "gt").mkdir(parents=True)
    (repo / "evaluation" / "gt" / "BAR-ATH_possession_gt.csv").write_text("frame,label\n")
    preflight.RESULTS.clear()
    preflight.hygiene_file_checks(preflight.published_files(str(repo)))
    assert status_of("(no full-match GT)") == "FAIL"


def test_not_a_git_repo_raises(tmp_path):
    with pytest.raises(RuntimeError):
        preflight.published_files(str(tmp_path))
