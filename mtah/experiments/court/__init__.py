"""The court: one shared arena in which feature candidates are built and judged.

CANDIDATE CONTRACT
  File  mtah/experiments/cand_<nn>_<slug>.py  exposing:
    NAME: str                      display name
    PARAMS: dict                   may override ONLY switch_cost and dead_cost of mtah.decode.PARAMS;
                                   every other key is ignored
    emissions(trk) -> ndarray (n, 3) float64
                                   log-likelihoods for H, A, D on ALL n frames (finite or -inf, no
                                   NaN / +inf); the harness restricts them to the blocks
    decode(trk, e) -> labels       OPTIONAL, for candidates with their own decoder; (n,) array of
                                   'H' 'A' 'D' 'X'. Default: mtah.decode.viterbi per block with
                                   the merged params. Frames outside the blocks are forced to 'X'.

  trk (see court/trk.py) is a read-only dict:
    px, py, ps  float32 (n, 46)  player x / y in metres, speed in m/s, NaN where missing
    team        int8 (46,)       0 home, 1 away, -1 unknown (slots are tracklets, not players)
    bx, by      float32 (n,)     ball in metres, NaN where missing
    pitch (L, W), fps = 25, blocks [(half, lo, hi)] inclusive frame ranges that get decoded

  Rules (inviolable): a candidate reads ONLY trk. No ground truth of any kind, no data files, no
  absolute paths, no import of evaluation/. Smart Tagging use needs a line
  `# lint-allow: smart_tagging <reason>`. `--rules-lint` greps for violations and a normal run
  disqualifies violators. Candidates must be deterministic and must not mutate trk.

  Scoring: synthetic matches with known truth (spellF1 / macroF1 via evaluation.possession_metrics,
  X forward-filled as in run_evaluation.py) and, with --real, UNLABELED diagnostics only.
"""
