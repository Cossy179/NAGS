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

**Shards** (`--first-game N`, opt-in): the games are numbered N to N +
`--games` − 1 and each one draws its random opening from its own stream,
seeded by the seed and its number, so a game is the same whichever thread
plays it (only the order in the file depends on the threads; with
`--threads 1` the file is in game order). `<out>.games` lists the settings
and every finished game (number, records, bytes, result). Run the same
command again after a stop and it resumes: a game cut off mid-write is
truncated and only the missing numbers are played; other settings are
refused. Several machines can share one seed with disjoint ranges
(`--first-game 0 --games 100000`, `--first-game 100000 --games 100000`, ...)
and never write the same game, and the index gives the game boundaries
(for a validation split by game). Without `--first-game` the output of a
command is exactly what it was.

Where the time goes (callgrind, network 8, 5000 nodes, one thread): the
search is 99% of the instructions; openings, filtering, packing and output
together about 1%. Of the positions searched (36,122 in 300 games), 73.7%
are kept, 8.5% are in check, 17.3% have a capture or promotion as the best
move and 0.5% a mate or tablebase score; 16% of the random openings are
rejected (each costs a depth-6 search). So data generation gets faster with
the search itself, and at most about 6% from searching the in-check
positions with a smaller budget (which would change the games). One thread
here: 353 kept positions per second.

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
chosen at run time with GCC on x86-64 Linux. With MSVC on x86-64 Windows
they, and the add/subtract kernels used by refreshes, have hand-written AVX2
versions chosen at run time by CPUID (about 2.6× faster `bench`); other
builds use the portable code. All versions give exactly the same results. With the test network, `bench` runs at about 0.8× the speed of the
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
on a network), `tests/python/test_nnue.py` (features, export, and exact
agreement between the engine's `eval` and the trainer's integer reference
`quantized_eval`) and `tests/python/test_nnue_v4.py` (the same for format
version 4, through castling, promotions and en passant).

## 4. Experimental: format version 4 (a hidden stack)

`tools/nnue/train_v4.py` trains, and the engine reads, a fourth format next
to versions 1-3 (which are unchanged: `docs/COMPATIBILITY.md`). It keeps
everything up to the accumulators (features, king buckets, mirroring, the
feature transformer and its quantization), so the incremental updates are
the same code, and replaces the single output layer by two small hidden
layers per output bucket:

    accumulators -> inputs (2H clipped values, their squares with
    --activation screlu, or H pairwise products with --activation pairwise)
    -> L1 (--l1, default 16) -> L2 (--l2, default 32) -> 1

all in integers (the file format and the exact arithmetic are described at
the top of `train_v4.py`; `quantized_eval` there is the reference the
engine matches exactly). Nothing uses version 4 unless such a file is
loaded with `EvalFile`, and `train.py`, its checkpoints and `run_round.py`
are not involved.

Its cost is the price of a deeper network. Most inputs are zero after the
clipped ReLU, so the engine's AVX2 code visits only the nonzero pairs of
inputs of each layer. With untrained H = 256, 8-bucket, 4-king-bucket
networks (loaded machine, minimum of 6 `speedtest` runs):

| Network | `evaluate` | `go nodes 1500000` from a middlegame |
|---------|-----------:|-------------------------------------:|
| version 3, embedded network 7 | 33 ns | 1.78M nodes/s |
| version 4, pairwise, 16-32 | 117 ns | 1.41M nodes/s (−21%) |
| version 4, clipped ReLU, 16-32 | 241 ns | 1.32M nodes/s (−26%) |

(A trained network's inputs are usually sparser, so faster.) The 512-wide
network above lost 48 Elo with a 25% speed loss, so a version 4 network
must buy more than its speed costs. Test it as any design: train it and a
version 3 network on the same data with the same epochs and settings, then
an SPRT between the two engines at 3+0.03 (`tools/sprt.py`, each engine
with its `EvalFile`); validation losses alone do not decide it.

```bash
python tools/nnue/train_v4.py --data "data/*.bin" --epochs 20 --buckets 8 \
    --king-buckets 4 --activation pairwise --out nets/v4.nnue
```

## Running it on your own machine (GPU training)

Training is by far the slowest step on a CPU; an NVIDIA GPU with CUDA does it
many times faster. Data generation and test matches run on the CPU, so more
cores help those. `tools/nnue/run_round.py` runs a whole round with one
command.

**Setup (Windows; Linux is the same with `build/` paths):**

1. Install Git, CMake 3.15+, Visual Studio 2022 (or its Build Tools; the
   2019 Build Tools, MSVC 14.29, work too) with "Desktop development with
   C++", Python 3.10–3.13, and a current NVIDIA driver.
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

**Then one command does everything:**
```
python tools\nnue\run_round.py --target-positions 200000000 --threads 11
```
(`--threads`: one less than your CPU's thread count.) It

1. generates self-play data (`data\selfplay_<seed>.bin`, batches of
   `--games` 150,000 games, about 13.5M positions each) until `data\` holds
   `--target-positions` positions; older `selfplay*.txt` files are converted
   to `.bin` once and count too;
2. trains each network design in `--designs` on all of it (on the GPU,
   `--device auto`): by default `current` (the built-in network's design)
   and `kb4` (4 king buckets with SCReLU, which needs this much data to pay
   off); `kb8` and `kb16` are also available;
3. plays each new network against the built-in one (SPRT [0, 10] at
   3+0.03); if several win, the best two play each other; the winner is
   copied to `nets\nags.nnue`.

It can be stopped at any time (Ctrl+C, a restart): run the same command
again and it carries on where it was. Finished batches are kept, training
continues from its checkpoint, and test matches resume from their logs.
Each step's result is appended to `data\round_<positions>.txt`. Expect
about 38 hours of data generation for 200M positions on a 6-core CPU, then
about an hour (`current`) to two hours (`kb4`) per design on an RTX 3060;
test matches take minutes to hours, depending on how far apart the networks
are (measured figures below).

If a new network was copied to `nets\nags.nnue`: rebuild
(`cmake --build build --config Release --parallel`), check `bench`, record
the match in `docs/TESTING.md` and the network in the table below, and
commit. For the next round raise `--target-positions` (new games come from
the stronger engine) and run the command again. `--skip-datagen` trains on
the data that is there; `--skip-test` stops after training.

### Measured on a Windows PC (October 2026)

The round that produced network 8 ran on an AMD Ryzen 5 3500X (6 cores,
6 threads: no SMT), an RTX 3060 12 GB, 16 GB of RAM, the Visual Studio 2019
Build Tools and Python 3.13 with PyTorch 2.11 (`cu128` wheels for a driver
reporting CUDA 12.8):

| Step | Measured |
|------|----------|
| `nags_datagen`, `--threads 6` | 1,465 positions/s (5 threads: 1,301; 12 threads: 1,414, so more threads than the CPU has do not help); about 87 positions a game |
| 201.7M positions | about 38 hours of generation, spread over eight sittings (stopped and resumed each time) |
| Loading them for training (6.4 GB, from a SATA hard disk) | 50–77 s; kept in GPU memory (8.7 GB of 12 GB in use) |
| Training `current`, 20 epochs | 212 s an epoch (about 950,000 positions/s): 71 minutes |
| Training `kb4`, 20 epochs | 365 s an epoch (about 550,000 positions/s): 2 hours |
| Test matches at 3+0.03, 5 at a time | 5–11 minutes each here (208–410 games), because the networks were far apart; close networks need thousands of games |

Practical notes from that round:

* **Data on another drive.** `--data-dir D:\NAGS\data` puts everything
  (data, networks, checkpoints, match logs, the round summary) on that
  drive. A SATA hard disk (about 100 MB/s, 3M positions/s) is fast enough:
  training holds the data in GPU memory, and streaming would still read
  faster than training uses it.
* **The target overshoots by up to one batch.** The total is checked
  between batches, so the last batch is played to its end (201.7M for a
  200M target).
* **Resuming starts a new batch.** After a stop, the next batch gets a new
  seed and `nags_datagen`'s counter starts at zero; the running total is
  the `<positions> of <target> positions` line `run_round.py` prints before
  each batch. With `--resumable-batches` (numbered games, `nags_datagen
  --first-game 0`) the stopped batch is finished under its own seed instead.
* **Start long runs from your own console** (Explorer, the Start menu, a
  `.cmd` file), not as a child of another application: a run started from
  inside an app that then updated itself (a packaged Store app) was ended
  with it.
* **A click in a classic console window pauses its output** ("Select" in
  the title bar). A paused window stalls a program that prints, such as the
  trainer's epoch lines: press Esc, or turn QuickEdit off.
* **Python.** The Microsoft Store build of Python 3.10 hung in a venv; the
  python.org 3.13 build works. The PyTorch CUDA wheel is about 2.75 GB: if
  the download times out, `pip install --timeout 120 --retries 10 ...`; pip's
  cache grew to about 20 GB (`pip cache purge`).

Known issues on Windows (at `0291007`):

* `tools/calibrate.py` still splits `--engine` with POSIX rules, which drop
  the backslashes of a Windows path (`tools/sprt.py` was fixed): give the
  engine path with forward slashes, e.g. `C:/.../nags_enhanced.exe`.
* Before `sprt.py` was fixed, every game of a match failed to start
  (`FileNotFoundError`), and the runner still scored the 204 failures as a
  tie and accepted H0. `run_round.py` now checks that a network loads before
  a match, but a match in which every game is an engine failure still ends
  with a decision: check the `Endings:` line of a surprising result.
* `tests/python/test_sprt.py::test_engine_paths_stay_whole` fails on
  Windows because the test's own file name contains a backslash, which
  Windows reads as a directory separator (the function under test is fine);
  `tests/python/test_ladder.py::test_anchor_and_rate` fails because the
  recorded bench is `None`. The GNN tests need `torch_geometric`.
* Calibrations from different machines are not comparable: network 7
  rated 3010 ± 31 on this PC and 3111 ± 34 on the VM (`docs/TESTING.md`).

## Networks

All networks use the training settings above (20 epochs, batch 16384, Adam
1e-3 with a cosine schedule to 1e-5, λ = 0.75). Networks 1-7 share the
768 → 256 clipped-ReLU design and differ only in their data; network 8 adds
king buckets and SCReLU. Each was admitted by a match against the previous
default (`docs/TESTING.md`).

| Network | Data | Validation loss | Result |
|---------|------|-----------------|--------|
| 1 | 4.5M positions from 49,714 `nags_datagen` games by the hand-written-evaluation engine (seed 1, 5000 nodes, engine at `efec5f8`) | 0.00646 | +143 ± 71 Elo at 20000 nodes and +103 ± 56 at 3+0.03 against the hand-written evaluation |
| 2 | network 1's data plus 3.6M positions from 40,000 more games (seed 2): 8.1M | 0.00643 | +29.9 ± 17.9 against network 1 at 20000 nodes |
| 3 | network 2's data plus 8.0M positions from ~90,000 games by the network-1 engine (seed 3, engine at `20797e4`): 16.1M | 0.00723 | +258 ± 51 against network 2 and +354 ± 132 against the hand-written evaluation, both at 3+0.03 |
| 4 | network 3's data plus 2.7M positions from the network-2 engine (seed 4) and 8.2M from the network-3 engine (seed 5): 26.9M | 0.00780 | +92.3 ± 32.1 against network 3 at 3+0.03; +15.7 ± 11.8 at 20000 nodes against the same network trained without the 8.1M hand-written-evaluation-engine positions |
| 5 | network 4's data plus 11.5M positions from ~137,000 games by the network-4 engine (seed 6): 38.4M (trained in two runs with `--stop-after` / `--resume`, 4 threads) | 0.00815 | +68.8 ± 28.1 against network 4 at 20000 nodes |
| 6 | network 5's data plus 9.3M positions from ~110,000 games by the network-5 engine (seeds 7 and 8): 47.7M. First network with **8 output buckets** (`--buckets 8`, format version 2) | 0.00830 | +101.0 ± 34.2 against network 5 at 3+0.03 (the gain combines the new data and the buckets) |
| 7 | network 6's data plus 8.86M positions from ~104,000 games by the network-6 engine (seed 9): 56.5M, 8 buckets (trained in three runs with `--stop-after` / `--resume`) | 0.00843 | +27.4 ± 16.9 against network 6 at 3+0.03 |
| 8 (`nets/nags.nnue`) | 201.7M new positions only (none of network 7's data): about 2.3M games by the network-7 engine at `bb1e03c` (5000 nodes, seeds 9001–9024), generated on the 6-core Windows PC and trained on its RTX 3060 (`run_round.py`, design `kb4`, about 2 h). **4 king buckets** (mirrored) + **SCReLU**, 8 output buckets, format version 3 | 0.00843 (0.00859 for the clipped-ReLU design on the same data and held-out positions) | +209.6 ± 41.7 against network 7 at 3+0.03; +48.6 ± 25.0 against the clipped-ReLU design trained on the same data, which itself scored +160.9 ± 35.8 against network 7 |

A 512-wide network trained on network 5's data reached a lower validation
loss (0.00778 against 0.00815 on the same held-out positions), but its build
searches about 25% fewer nodes per second and lost to network 5 by
−48.2 ± 25.4 Elo at 3+0.03, so 256 stays the default size. It might pay off
at longer time controls or with faster inference (e.g. AVX2 builds by
default, or an int8 output layer).

King buckets and SCReLU (format version 3) are in the default network from
network 8 on. A first check on a small data set (network 6's 8.9M
newest positions, 6 epochs, 8 output buckets, all three trained alike):

| Design | Validation loss | Match |
|--------|-----------------|-------|
| 768 → 256, clipped ReLU | 0.01021 | – |
| 4 king buckets (mirrored, factorised) + SCReLU | 0.01012 | −6.9 ± 12.5 against the first (SPRT [0, 10] at 3+0.03, H0 after 2158 games) |
| SCReLU only | 0.01023 | – |

The king-bucket network fits these positions better but did not play
better: with four times the input weights it overfits so little data
(training loss 0.0074 against 0.0094). Whether it pays off needs a test at
a few hundred million positions. At 201.7M positions it does: trained on
the same data, it beat the clipped-ReLU design by +48.6 ± 25.0 and became
network 8. (A first attempt at these matches played no games: `sprt.py`
then split Windows engine paths with POSIX rules, since fixed.)

Validation losses are not comparable across rows once NNUE-engine positions
enter the validation set (from network 3 on). Labels from games played by
NNUE engines gave the biggest gains: each round of data from the current
network, then retraining, has added a lot, and keeping the older data
helped as well.
