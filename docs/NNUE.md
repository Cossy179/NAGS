# NNUE evaluation

`nags_fast` and `nags_enhanced` can evaluate positions with an NNUE network
instead of the hand-written evaluation (material, piece-square tables,
bishop pair). The pipeline has three parts.

## 1. Training data: `nags_datagen`

```bash
build/nags_datagen --out data/selfplay.txt --games 100000 --threads 3 --nodes 5000
```

Each thread plays games with its own `nags_enhanced` search (FastBoard,
16 MB hash). A game starts with 8 random moves (`--random-plies`); openings
the search already scores beyond ±400 cp at depth 6 are discarded. Every
move is then searched to a fixed node budget (`--nodes`). Games end by the
rules, or by adjudication: ±2000 cp for 6 plies (win), |score| ≤ 10 cp for
20 plies after ply 80 (draw), or 400 plies (draw).

Kept positions: not in check, a quiet best move, no mate or tablebase
score. Each is one line,

```
<fen> | <score> | <result>
```

with the search score in centipawns and the result (1.0 / 0.5 / 0.0), both
from White's point of view. Output is appended, so several runs (use a
different `--seed`) can go into one file.

## 2. Training: `tools/nnue/train.py`

```bash
python tools/nnue/train.py --data data/selfplay.txt --cache data/selfplay.npz \
    --epochs 30 --out nets/nags.nnue
```

Network: 768 inputs per perspective (colour relative to the side whose
perspective it is × 6 piece types × 64 squares, mirrored for Black), one
768 → 256 feature transformer shared by both perspectives, a clipped ReLU,
and one output from the two accumulators concatenated (side to move first).
The target is `λ·sigmoid(score/400) + (1 − λ)·result` for the side to move
(`--lambda`, default 0.75), with a squared error on `sigmoid(output)`.
Training uses Adam with a cosine learning-rate schedule and holds out 1% of
the positions for a validation loss. The network is exported after every
epoch, quantized (feature weights × 255, output weights × 64), in the format
described at the top of `train.py`.

## 3. Inference in the engine

`src/Nnue.h` / `src/Nnue.cpp`. FastBoard keeps both accumulators up to date
as pieces are put and removed (so unmaking a move restores them exactly),
and `eval::evaluate` uses the network whenever one is active.

* The build embeds `nets/nags.nnue` if it exists (CMake option
  `NAGS_NNUE_FILE`). Without it the engines use the hand-written evaluation.
* UCI options (`nags_fast`, `nags_enhanced`): `UseNNUE` (default true) and
  `EvalFile` (a network file, or `<embedded>`).
* The UCI command `eval` prints the static evaluation of the current
  position and which evaluation produced it.

Checks: `nnue_tests` (incremental accumulators equal recomputed ones through
random games, colour-mirrored positions evaluate the same, the search runs
on a network) and `tests/python/test_nnue.py` (features, export, and exact
agreement between the engine's `eval` and the trainer's integer reference
`quantized_eval`).
