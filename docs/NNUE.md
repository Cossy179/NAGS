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
described at the top of `train.py`, and the training state is saved to
`<out>.ckpt`, so an interrupted run continues with `--resume` (same data and
settings).

## 3. Inference in the engine

`src/Nnue.h` / `src/Nnue.cpp`. FastBoard keeps a stack of accumulators:
`makeMove` writes the new position's accumulators from the previous ones in
one pass (adding and removing the moved, captured and castling pieces), and
`unmakeMove` just drops them. `eval::evaluate` uses the network whenever one
is active. The update and evaluation kernels are also compiled for AVX2 and
chosen at run time with GCC on x86-64 Linux; other builds use the portable
code. With the test network, `bench` runs at about 0.8× the speed of the
hand-written evaluation (portable build; about 0.5× before the accumulator
stack and the AVX2 kernels).

* The hidden layer size is fixed at compile time: CMake option
  `NAGS_NNUE_HIDDEN` (default 256; train with the same `--hidden`). Networks
  of another size are rejected when loaded.
* The build embeds `nets/nags.nnue` (CMake option `NAGS_NNUE_FILE`), so
  `nags_fast` and `nags_enhanced` use NNUE by default. Without the file the
  engines use the hand-written evaluation.
* UCI options (`nags_fast`, `nags_enhanced`): `UseNNUE` (default true) and
  `EvalFile` (a network file, or `<embedded>`).
* The UCI command `eval` prints the static evaluation of the current
  position and which evaluation produced it.

Checks: `nnue_tests` (incremental accumulators equal recomputed ones through
random games, colour-mirrored positions evaluate the same, the search runs
on a network) and `tests/python/test_nnue.py` (features, export, and exact
agreement between the engine's `eval` and the trainer's integer reference
`quantized_eval`).

## Networks

All networks so far use the architecture and training settings above (20
epochs, batch 16384, Adam 1e-3 with a cosine schedule to 1e-5, λ = 0.75)
and differ only in their data. Each was admitted by a match against the
previous default (`docs/TESTING.md`).

| Network | Data | Validation loss | Result |
|---------|------|-----------------|--------|
| 1 | 4.5M positions from 49,714 `nags_datagen` games by the hand-written-evaluation engine (seed 1, 5000 nodes, engine at `efec5f8`) | 0.00646 | +143 ± 71 Elo at 20000 nodes and +103 ± 56 at 3+0.03 against the hand-written evaluation |
| 2 | network 1's data plus 3.6M positions from 40,000 more games (seed 2): 8.1M | 0.00643 | +29.9 ± 17.9 against network 1 at 20000 nodes |
| 3 | network 2's data plus 8.0M positions from ~90,000 games by the network-1 engine (seed 3, engine at `20797e4`): 16.1M | 0.00723 | +258 ± 51 against network 2 and +354 ± 132 against the hand-written evaluation, both at 3+0.03 |
| 4 | network 3's data plus 2.7M positions from the network-2 engine (seed 4) and 8.2M from the network-3 engine (seed 5): 26.9M | 0.00780 | +92.3 ± 32.1 against network 3 at 3+0.03; +15.7 ± 11.8 at 20000 nodes against the same network trained without the 8.1M hand-written-evaluation-engine positions |
| 5 (`nets/nags.nnue`) | network 4's data plus 11.5M positions from ~137,000 games by the network-4 engine (seed 6): 38.4M (trained in two runs with `--stop-after` / `--resume`, 4 threads) | 0.00815 | +68.8 ± 28.1 against network 4 at 20000 nodes |

A 512-wide network trained on network 5's data reached a lower validation
loss (0.00778 against 0.00815 on the same held-out positions), but its build
searches about 25% fewer nodes per second and lost to network 5 by
−48.2 ± 25.4 Elo at 3+0.03, so 256 stays the default size. It might pay off
at longer time controls or with faster inference (e.g. AVX2 builds by
default, or an int8 output layer).

Validation losses are not comparable across rows once NNUE-engine positions
enter the validation set (from network 3 on). Labels from games played by
NNUE engines gave the biggest gains: each round of data from the current
network, then retraining, has added a lot, and keeping the older data
helped as well.
