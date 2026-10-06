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

With an output name ending in `.bin` each position is a 32-byte binary
record instead (occupancy, 4-bit piece codes, score, result, side to move;
the layout is at the top of `src/datagen_main.cpp`). That is about half the
size of text and loads in seconds, so it is the format to use for large
data sets. `tools/nnue/pack.py` converts existing text files (or `.npz`
caches): `python tools/nnue/pack.py "data/selfplay*.txt"` writes a `.bin`
next to each.

## 2. Training: `tools/nnue/train.py`

```bash
python tools/nnue/train.py --data data/selfplay.txt --cache data/selfplay.npz \
    --epochs 30 --out nets/nags.nnue
python tools/nnue/train.py --data "data/*.bin" --epochs 30 --out nets/nags.nnue
```

Text input is parsed into memory (`--cache` saves the parsed arrays as
`.npz`). Binary input (`.bin`, which cannot be mixed with text) is kept in
GPU memory when it fits; otherwise it is read from disk in groups of 1M-
position chunks, in a new random order every epoch, shuffled within each
group and loaded while the previous group trains, so data sets much larger
than memory train at full speed. Its validation set is the last 1% of the
records (at most 1M). Wildcards in `--data` are expanded by the trainer.

Network: 768 inputs per perspective (colour relative to the side whose
perspective it is × 6 piece types × 64 squares, mirrored for Black), one
768 → 256 feature transformer shared by both perspectives, a clipped ReLU,
and one output from the two accumulators concatenated (side to move first).
With `--buckets N` (up to 8) there are N output layers, picked by the number
of pieces on the board (bucket = (pieces − 1) · N / 32), so the opening,
middlegame and endgame get their own final weights at almost no cost.

Two more options change the design (format version 3; the engine reads all
versions):

* `--king-buckets 4|8|16|mirror` gives each perspective several 768-feature
  sets, chosen by its own king's square, and mirrors its board left-right
  when that king is on files e–h (`mirror` only mirrors). The layouts are in
  `KING_LAYOUTS` in `train.py`; 32 comma-separated numbers give a custom one.
  During training a shared factor set is added to every bucket, so what the
  buckets have in common is learned from all positions; it is folded into
  the buckets on export. More buckets need more data.
* `--activation screlu` squares the clipped ReLU (SCReLU), which usually
  evaluates better at the same size and speed.

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
`makeMove` only records the pieces the move adds and removes, the
accumulators are computed when an evaluation first needs them (in one pass
per move from the nearest computed position below), and `unmakeMove` just
drops them. With king buckets a perspective whose king changes feature set
is recomputed instead, from a per-board cache that keeps one accumulator per
perspective and king bucket together with the pieces it was computed for,
so only the difference to the current pieces is applied. `eval::evaluate`
uses the network whenever one is active. The update and evaluation kernels are also compiled for AVX2 and
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

## Running it on your own machine (GPU training)

Training is by far the slowest step on a CPU; an NVIDIA GPU with CUDA does it
many times faster. Data generation and test matches run on the CPU, so more
cores help those. `tools/nnue/run_round.py` runs a whole round with one
command.

**Setup (Windows; Linux is the same with `build/` paths):**

1. Install Git, CMake 3.15+, Visual Studio 2022 (or its Build Tools) with
   "Desktop development with C++", Python 3.10–3.12, and a current NVIDIA
   driver.
2. Clone and build:
   ```
   git clone https://github.com/Cossy179/NAGS.git
   cd NAGS
   cmake -B build
   cmake --build build --config Release --parallel
   ```
   The engines land in `build\Release\`. `build\Release\nags_enhanced.exe bench`
   should print the node count listed in `docs/TESTING.md`.
3. Python with the CUDA build of PyTorch (take the exact install command for
   your CUDA version from pytorch.org, for example):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install numpy chess
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   The last line should print `True` and the GPU's name.

**First, build up data.** The training data is not in the repository (it is
large and easy to regenerate). A network trained on a few million positions
is weaker than the built-in one, so generate several batches first, each
with a new seed (about 90 positions per game, 32 bytes each). For the
current design 40–50M positions are enough; bigger designs (king buckets, a
wider layer) need 200M or more:
```
build\Release\nags_datagen.exe --out data\selfplay_101.bin --games 150000 --threads 11 --seed 101
```
(`--threads`: one less than your CPU's thread count.) In PowerShell, a loop
over seeds runs unattended, e.g. 15 batches of about 13.5M positions:
```
foreach ($s in 101..115) { build\Release\nags_datagen.exe --out data\selfplay_$s.bin --games 150000 --threads 11 --seed $s }
```

**Then run rounds:**
```
python tools\nnue\run_round.py --games 150000 --threads 11 --seed 102
```
This generates new games, trains on every `data\selfplay*.bin` (on the GPU,
`--device auto`; older `selfplay*.txt` files are converted to `.bin` once),
and plays the new network against the built-in one (SPRT [0, 10] at
3+0.03). `--king-buckets` and `--activation` choose the network design, e.g.
`--king-buckets 4 --activation screlu`. If the new one wins it is copied to
`nets\nags.nnue`: rebuild, check `bench`, record the match in
`docs/TESTING.md` and commit. `--skip-datagen` trains on the existing data
only; `--skip-test` stops after training.

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
| 5 | network 4's data plus 11.5M positions from ~137,000 games by the network-4 engine (seed 6): 38.4M (trained in two runs with `--stop-after` / `--resume`, 4 threads) | 0.00815 | +68.8 ± 28.1 against network 4 at 20000 nodes |
| 6 | network 5's data plus 9.3M positions from ~110,000 games by the network-5 engine (seeds 7 and 8): 47.7M. First network with **8 output buckets** (`--buckets 8`, format version 2) | 0.00830 | +101.0 ± 34.2 against network 5 at 3+0.03 (the gain combines the new data and the buckets) |
| 7 (`nets/nags.nnue`) | network 6's data plus 8.86M positions from ~104,000 games by the network-6 engine (seed 9): 56.5M, 8 buckets (trained in three runs with `--stop-after` / `--resume`) | 0.00843 | +27.4 ± 16.9 against network 6 at 3+0.03 |

A 512-wide network trained on network 5's data reached a lower validation
loss (0.00778 against 0.00815 on the same held-out positions), but its build
searches about 25% fewer nodes per second and lost to network 5 by
−48.2 ± 25.4 Elo at 3+0.03, so 256 stays the default size. It might pay off
at longer time controls or with faster inference (e.g. AVX2 builds by
default, or an int8 output layer).

King buckets and SCReLU (format version 3) are supported but not yet in the
default network. A first check on a small data set (network 6's 8.9M
newest positions, 6 epochs, 8 output buckets, all three trained alike):

| Design | Validation loss | Match |
|--------|-----------------|-------|
| 768 → 256, clipped ReLU | 0.01021 | – |
| 4 king buckets (mirrored, factorised) + SCReLU | 0.01012 | −6.9 ± 12.5 against the first (SPRT [0, 10] at 3+0.03, H0 after 2158 games) |
| SCReLU only | 0.01023 | – |

The king-bucket network fits these positions better but did not play
better: with four times the input weights it overfits so little data
(training loss 0.0074 against 0.0094). Whether it pays off needs a test at
a few hundred million positions.

Validation losses are not comparable across rows once NNUE-engine positions
enter the validation set (from network 3 on). Labels from games played by
NNUE engines gave the biggest gains: each round of data from the current
network, then retraining, has added a lot, and keeping the older data
helped as well.
