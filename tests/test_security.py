"""Security tests: leak guard rules, output-path safety, cache integrity. No real data.

    python -m pytest tests -q -p no:cacheprovider
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import source_frames  # noqa: E402
from mtah import features, guard, run  # noqa: E402


# --- guard: paths ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "data/tactical/BAR-ATH_tactical_atd.txt",
    "data/cache/BAR-ATH_features.npz",
    "data\\tactical\\anything.csv",                       # Windows separators
    "notes/BAR-ATH_tactical_homography.csv",              # asset name outside data/
    "backup/video.mp4",
    "scratch/feats.npz",
    "x/model.pkl",
    "evaluation/gt/BAR-ATH_possession_gt.csv",            # private GT name
    "evaluation/gt/BAR-ATH_possession_gt_reference.csv",
    ".env", "config/.env.local", "deploy/server.pem", "id_rsa",
])
def test_guard_blocks_paths(path):
    assert guard.check_path(path)


@pytest.mark.parametrize("path", [
    "data/README.md",
    "evaluation/gt/BAR-ATH_possession_gt_public.csv",
    "evaluation/gt/BAR-ATH_possession_gt_public_meta.json",
    "mtah/decode.py", "tests/test_decode.py", "README.md",
    "evaluation/results/baseline_smooth50.csv",
])
def test_guard_allows_paths(path):
    assert guard.check_path(path) == []


# --- guard: content -------------------------------------------------------------------------

def fake(*parts):
    """Assemble a fake secret at runtime so this file never contains one literally."""
    return "".join(parts).encode()


@pytest.mark.parametrize("blob", [
    fake("-----BEGIN ", "RSA PRIVATE KEY-----\nMIIE..."),
    fake("AK", "IA", "ABCDEFGHIJKLMNOP"),
    fake("gh", "p_", "a" * 36),
    fake("AI", "za", "B" * 35),
    fake("sk", "-", "x" * 40),
    fake("api_key = '", "abcdef123456789", "'"),
])
def test_guard_finds_secrets(blob):
    hits = guard.check_content("notes/x.txt", blob, len(blob))
    assert any("secret" in h for h in hits)


def test_guard_finds_renamed_atd_dump():
    line = "{}:0.5,0.5,1.2;NaN,NaN,NaN;0.4,0.6,2.0:0.5,0.5,0\n"
    blob = "".join(line.format(i) for i in range(1, 200)).encode()
    hits = guard.check_content("notes/innocent.txt", blob, len(blob))
    assert any("ATD" in h for h in hits)


HOMOG_HEAD = b"frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n"
HOMOG_ROW = b"%d,1822.66,-511.2,44.48,1.69,169.29,255.73,-0.019,-0.52,1,0.29,1\n"


def test_guard_finds_homography_and_tagging_exports():
    homog = HOMOG_HEAD + b"".join(HOMOG_ROW % i for i in range(100))
    assert any("homography" in h for h in guard.check_content("a.csv", homog, len(homog)))
    tag = b"<file><ALL_INSTANCES>" + b"<instance><ID>1</ID></instance>" * 30
    assert any("SportsCode" in h for h in guard.check_content("b.xml", tag, len(tag)))
    meta = b"".join(b'<Player id="P%d" teamId="T1"/>' % i for i in range(46))
    assert any("metadata" in h for h in guard.check_content("c.xml", meta, len(meta)))


def test_guard_ignores_single_quoted_examples():
    """Docs quoting one header/row, and one-line synthetic templates, are not leaks."""
    blob = HOMOG_HEAD + HOMOG_ROW % 11185 + b'<Player id="P{0}" teamId="{1}">' + b"<instance>"
    assert guard.check_content("DATA.md", blob, len(blob)) == []


def test_guard_blocks_large_file():
    assert any("large" in h for h in guard.check_content("a.csv", b"", 6_000_000))


def test_guard_clean_code_passes():
    blob = b"import numpy as np\n\ndef f(x):\n    return x + 1\n"
    assert guard.check_content("mtah/x.py", blob, len(blob)) == []


def test_guard_submission_csv_is_not_flagged_as_atd():
    blob = b"frame,label\n" + b"".join(b"%d,H\n" % i for i in range(500))
    assert guard.check_content("evaluation/results/x.csv", blob, len(blob)) == []


def test_guard_passes_on_this_repo():
    """The working tree as git would commit it must be clean right now."""
    try:
        paths = guard.candidate_files(staged=False)
    except Exception:
        pytest.skip("git not available")
    assert guard.scan(paths) == []


# --- run.py: output path safety ---------------------------------------------------------------

@pytest.mark.parametrize("team", [
    "../evil", "..\\evil", "a/b", "a\\b", "C:evil", "/abs", "", ".hidden", "-flag",
    "x" * 65, "CON", "nul", "com1", "team name", "t\x00", "tëam",
])
def test_bad_team_names_rejected(team, tmp_path):
    with pytest.raises(ValueError):
        run.safe_output(str(tmp_path), team)


@pytest.mark.parametrize("team", ["nirakshak", "team_1", "Team-2", "a"])
def test_good_team_names_accepted(team, tmp_path):
    path = run.safe_output(str(tmp_path), team)
    assert os.path.dirname(path) == os.path.realpath(str(tmp_path))
    assert path.endswith(team + ".csv")


# --- features cache integrity -----------------------------------------------------------------

def good_cache(n, slots=3):
    f = {k: np.zeros(n, dtype=np.float32) for k in features.FIELDS}
    f["pos"] = np.zeros((n, slots, 2), dtype=np.float16)
    f["slot_team"] = np.zeros(slots, dtype=np.int8)
    return f


N = source_frames("tactical")


def test_cache_roundtrip_ok(tmp_path):
    p = str(tmp_path / "c.npz")
    np.savez_compressed(p, **good_cache(N))
    assert features._load_cache(p, N) is not None


def test_cache_with_pickled_object_is_refused(tmp_path):
    p = str(tmp_path / "c.npz")
    f = good_cache(N)
    f["ball_x"] = np.array([{"payload": 1}] * N, dtype=object)   # needs pickle to load
    np.savez(p, **f)
    assert features._load_cache(p, N) is None


@pytest.mark.parametrize("mutate", [
    lambda f: f.pop("pos"),                                      # missing key
    lambda f: f.update(extra=np.zeros(3)),                       # unknown key
    lambda f: f.update(ball_x=np.zeros(N - 1, np.float32)),      # wrong length
    lambda f: f.update(pos=np.zeros((N, 4, 2), np.float16)),     # slots mismatch
    lambda f: f.update(d_home=np.array(["a"] * N)),              # non-numeric
])
def test_cache_tampered_is_refused(tmp_path, mutate):
    p = str(tmp_path / "c.npz")
    f = good_cache(N)
    mutate(f)
    np.savez(p, **f)
    assert features._load_cache(p, N) is None


def test_cache_corrupt_file_is_refused(tmp_path):
    p = tmp_path / "c.npz"
    p.write_bytes(b"not a zip at all")
    assert features._load_cache(str(p), N) is None
