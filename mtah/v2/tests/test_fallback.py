"""Fallback variants equal their experiments on synthetic seeds 1-3 (labels), plus runtime.

    python -m mtah.v2.tests.test_fallback

The experiments and the synthetic generator are imported here only, never by the pipeline.
"""

import time

import numpy as np

from mtah.experiments import cand_02_touch_events as c02
from mtah.experiments import stack_1_touch_semimarkov as stack1
from mtah.experiments.court import run as court
from mtah.experiments.court import synth
from mtah.v2 import fallback, tracking

N_TILED = 199319
TILED_BLOCKS = [(1, 2000, 2000 + 79000 - 1), (2, 100000, 100000 + 79857 - 1)]
LIMIT_S = {"touch_only": 60.0, "semi_markov": 120.0}


def check_seed(seed):
    trk, _ = synth.make(seed, 20, "realistic")
    old = {"touch_only": court.run_candidate(c02, trk)[0],
           "semi_markov": court.run_candidate(stack1, trk)[0]}
    e = fallback.emissions(trk)
    assert np.allclose(e, c02.emissions(trk), rtol=0, atol=1e-9), "emissions differ, seed %d" % seed
    assert not np.isnan(e).any() and not (e == np.inf).any(), "NaN/+inf, seed %d" % seed
    inside = np.zeros(len(e), bool)
    for _, lo, hi in trk["blocks"]:
        inside[lo:hi + 1] = True
    for variant in fallback.VARIANTS:
        labels = fallback.decode(trk, variant)
        assert np.array_equal(labels, old[variant]), "%s labels differ, seed %d" % (variant, seed)
        assert np.array_equal(labels, fallback.decode(trk, variant)), "%s not deterministic" % variant
        assert (labels[~inside] == "X").all() and (labels[inside] != "X").all(), "X not on blocks"
        print("seed {}: {} labels equal the experiment, deterministic, {} frames".format(
            seed, variant, len(labels)))


def check_runtime():
    trk, _ = synth.make(1, 20, "realistic")
    reps = -(-N_TILED // len(trk["bx"]))
    tile = lambda a: np.concatenate([a] * reps, axis=0)[:N_TILED]  # noqa: E731
    big = tracking.build(tile(trk["px"]), tile(trk["py"]), tile(trk["ps"]), trk["team"],
                         tile(trk["bx"]), tile(trk["by"]), trk["pitch"], TILED_BLOCKS)
    for variant in fallback.VARIANTS:
        t0 = time.perf_counter()
        labels = fallback.decode(big, variant)
        secs = time.perf_counter() - t0
        assert len(labels) == N_TILED and secs < LIMIT_S[variant], "%s too slow" % variant
        print("tiled {} frames, {}: {:.1f}s (limit {:.0f}s)".format(
            N_TILED, variant, secs, LIMIT_S[variant]))


def main():
    for seed in (1, 2, 3):
        check_seed(seed)
    check_runtime()
    print("FALLBACK OK")


if __name__ == "__main__":
    main()
