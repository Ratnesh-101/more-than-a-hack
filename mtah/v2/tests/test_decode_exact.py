"""The O(n) min-stoppage Viterbi equals a literal state-expanded dense Viterbi (score and path).

    python -m mtah.v2.tests.test_decode_exact

States: H, A, D1..Dk (Di = i-th consecutive dead frame, Dk absorbing). A block starts in H, A or
Dk (it may begin mid-stoppage) and ends in ANY state (it may end mid-stoppage, shorter than k).
"""

import numpy as np

from mtah.v2.decode import DEAD_COST, SWITCH_COST, _viterbi_min_dead

N_BLOCKS = 300


def dense(e, k, sw, dc):
    """(best score, path) by a full (k+2)-state Viterbi; ties prefer staying in the state."""
    s = k + 2
    last = s - 1
    trans = np.full((s, s), -np.inf)
    trans[0, 0] = trans[1, 1] = 0.0
    trans[0, 1] = trans[1, 0] = -sw
    trans[0, 2] = trans[1, 2] = -dc
    for i in range(2, last):
        trans[i, i + 1] = 0.0
    trans[last, last] = 0.0
    trans[last, 0] = trans[last, 1] = -dc
    em = np.concatenate([e[:, :2], np.repeat(e[:, 2:3], k, axis=1)], axis=1)
    n = len(e)
    idx = np.arange(s)
    score = np.full(s, -np.inf)
    score[[0, 1, last]] = em[0, [0, 1, last]]
    back = np.zeros((n, s), dtype=np.int64)
    for t in range(1, n):
        cand = score[:, None] + trans
        best = np.argmax(cand, axis=0)
        stay = cand[idx, idx] >= cand[best, idx]
        back[t] = np.where(stay, idx, best)
        score = cand[back[t], idx] + em[t]
    state = int(np.argmax(score))
    top = float(score[state])
    path = np.empty(n, dtype=np.int64)
    path[-1] = state
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return top, np.minimum(path, 2).astype(np.int8)


def path_score(e, path, k, sw, dc):
    """Score of a label path under the model, or None if it breaks the minimum stoppage."""
    n = len(e)
    total = float(e[np.arange(n), path].sum())
    edges = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [n]))
    for a, b in zip(starts, ends):
        if path[a] == 2 and a > 0 and b < n and b - a < k:
            return None
    for t in edges:
        total -= sw if path[t - 1] != 2 and path[t] != 2 else dc
    return total


def random_block(rng):
    k = int(rng.choice([2, 3, 5, 8, 75]))
    n = int(rng.integers(1, 3 * k + 60)) if rng.random() < 0.5 else int(rng.integers(1, 500))
    kind = int(rng.integers(3))
    e = rng.normal(0.0, rng.choice([0.3, 1.0, 3.0]), (n, 3))
    if kind == 0:                                  # integer emissions: many ties
        e = np.round(e)
    elif kind == 1:                                # stretches of strong dead / team evidence
        for _ in range(int(rng.integers(1, 6))):
            a = int(rng.integers(0, n))
            b = min(n, a + int(rng.integers(1, 2 * k + 1)))
            e[a:b, int(rng.integers(3))] += rng.choice([2.0, 6.0])
    else:                                          # block ends in a (possibly short) dead stretch
        e[max(0, n - int(rng.integers(1, 2 * k + 1))):, 2] += 8.0
    return e, k


def main():
    rng = np.random.default_rng(12345)
    n_path_equal = n_short = n_dead_end = 0
    for i in range(N_BLOCKS):
        e, k = random_block(rng)
        want, ref_path = dense(e, k, SWITCH_COST, DEAD_COST)
        path = _viterbi_min_dead(e, k, SWITCH_COST, DEAD_COST)
        got = path_score(e, path, k, SWITCH_COST, DEAD_COST)
        assert got is not None, "block {}: path breaks the minimum stoppage".format(i)
        assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), \
            "block {} (n={}, k={}): score {} != reference {}".format(i, len(e), k, got, want)
        assert path_score(e, ref_path, k, SWITCH_COST, DEAD_COST) is not None
        if np.array_equal(path, ref_path):
            n_path_equal += 1
        else:       # a different path is acceptable only as an exact tie of the optimum
            assert abs(path_score(e, ref_path, k, SWITCH_COST, DEAD_COST) - want) <= 1e-9 * max(
                1.0, abs(want))
        n_short += len(e) < k
        n_dead_end += path[-1] == 2 and len(e) > 1
    print("{} blocks: score equals the dense reference on all; path identical on {}; "
          "{} blocks shorter than k, {} ending in D".format(N_BLOCKS, n_path_equal, n_short,
                                                          n_dead_end))
    # continuous emissions have no ties: the path must be unique, hence identical
    rng = np.random.default_rng(777)
    for i in range(100):
        e, k = rng.normal(0, 2.0, (int(rng.integers(1, 300)), 3)), int(rng.choice([3, 8, 75]))
        e[-int(rng.integers(1, 2 * k)):, 2] += 3.0
        assert np.array_equal(_viterbi_min_dead(e, k, SWITCH_COST, DEAD_COST),
                              dense(e, k, SWITCH_COST, DEAD_COST)[1]), "continuous block %d" % i
    print("100 tie-free blocks: path identical to the dense reference")
    print("EXACT OK")


if __name__ == "__main__":
    main()
