"""tracking.check_speed_unit: m/s passes; pitch-normalised and km/h speeds raise; all-NaN passes.

    python -m mtah.v2.tests.test_speed_guard
"""

import contextlib
import io

import numpy as np

from mtah.experiments.court import synth
from mtah.v2 import tracking


def with_speed(trk, ps):
    out = dict(trk)
    out["ps"] = np.ascontiguousarray(ps, dtype=np.float32)
    return out


def raises(trk):
    try:
        tracking.check_speed_unit(trk)
    except ValueError as err:
        assert "not m/s" in str(err)
        return str(err)
    raise AssertionError("expected ValueError")


def main():
    trk, _ = synth.make(1, 20, "realistic")
    tracking.check_speed_unit(trk)                       # valid m/s data passes silently
    ps = np.asarray(trk["ps"])
    print("valid m/s: median {:.2f} p95 {:.2f} -> passes".format(
        float(np.nanmedian(ps)), float(np.nanpercentile(ps, 95))))

    print("normalised:", raises(with_speed(trk, ps / 100.0))[:110], "...")
    print("km/h:      ", raises(with_speed(trk, ps * 3.6 * 3))[:110], "...")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tracking.check_speed_unit(with_speed(trk, np.full(ps.shape, np.nan)))
    assert "all NaN" in buf.getvalue(), "no notice printed"
    print("all-NaN: passes, notice:", buf.getvalue().strip())
    print("SPEED GUARD OK")


if __name__ == "__main__":
    main()
