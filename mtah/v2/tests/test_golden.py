"""Golden test: the v2 port equals experiments/stack_2_touch_deadball on synthetic seeds 1-3.

    python -m mtah.v2.tests.test_golden

The experiment and the synthetic generator are imported here only, never by the pipeline.
"""

import time

import numpy as np

from mtah.experiments import stack_2_touch_deadball as stack2
from mtah.experiments.court import synth
from mtah.v2 import decode, tracking

N_TILED = 199319
TILED_BLOCKS = [(1, 2000, 2000 + 79000 - 1), (2, 100000, 100000 + 79857 - 1)]
RUNTIME_LIMIT_S = 60.0


def check_seed(seed):
    trk, _ = synth.make(seed, 20, "realistic")
    e_new, e_old = decode.emissions(trk), stack2.emissions(trk)
    assert np.allclose(e_new, e_old, rtol=0, atol=1e-9), "emissions differ, seed %d" % seed
    assert not np.isnan(e_new).any() and not (e_new == np.inf).any(), "NaN/+inf, seed %d" % seed
    labels = decode.decode(trk, e_new)
    assert np.array_equal(labels, stack2.decode(trk, e_old)), "labels differ, seed %d" % seed
    assert np.array_equal(labels, decode.decode(trk, decode.emissions(trk))), "not deterministic"
    assert np.array_equal(e_new, decode.emissions(trk)), "emissions not bit-exact twice"
    inside = np.zeros(len(labels), bool)
    for _, lo, hi in trk["blocks"]:
        inside[lo:hi + 1] = True
    assert (labels[~inside] == "X").all() and (labels[inside] != "X").all(), "X not on blocks"
    print("seed {}: emissions allclose (max |diff| {:.2e}), labels equal, deterministic, "
          "{} frames".format(seed, float(np.abs(e_new - e_old).max()), len(labels)))


def check_runtime():
    trk, _ = synth.make(1, 20, "realistic")
    reps = -(-N_TILED // len(trk["bx"]))
    tile = lambda a: np.concatenate([a] * reps, axis=0)[:N_TILED]  # noqa: E731
    big = tracking.build(tile(trk["px"]), tile(trk["py"]), tile(trk["ps"]), trk["team"],
                         tile(trk["bx"]), tile(trk["by"]), trk["pitch"], TILED_BLOCKS)
    t0 = time.perf_counter()
    e = decode.emissions(big)
    t_em = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels = decode.decode(big, e)
    t_dec = time.perf_counter() - t0
    assert len(labels) == N_TILED and t_em + t_dec < RUNTIME_LIMIT_S, "too slow"
    print("tiled {} frames: emissions {:.1f}s, decode {:.1f}s (limit {:.0f}s)".format(
        N_TILED, t_em, t_dec, RUNTIME_LIMIT_S))


def main():
    for seed in (1, 2, 3):
        check_seed(seed)
    check_runtime()
    print("GOLDEN OK")


if __name__ == "__main__":
    main()
