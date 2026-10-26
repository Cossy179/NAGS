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
`--threads 1` the file is in game order). `<out>.games` is the shard's index.
Its first line records the settings, the network (FNV-1a 64 of its bytes)
and the search (the `nags_enhanced bench` node count, computed at start).
Then there is one line per finished game: number, records, bytes, result and
the FNV-1a 64 checksum of its bytes, each written after the game's bytes.

Run the same command again after a stop and it resumes:

* an index line cut off as it was written is dropped;
* the last 64 games are checked against their checksums, and those that
  don't match, or that the output is too short for, are played again (they
  are the same games);
* the output is cut back to the games kept, and only the missing numbers
  are played.

Other settings, another network or another search are refused, as is a
damaged index line further up. An output file that holds data but has no
index (an ordinary run, or a deleted index) is never truncated or extended:
nags_datagen stops with an error. Several machines can share one seed with
disjoint ranges (`--first-game 0 --games 100000`, `--first-game 100000
--games 100000`, ...) and never write the same game.
`python tools/nnue/shards.py check data/*.bin` verifies a set of shards
before training: each one's size, the checksums (`--checksums all` for every
game), the same settings, network and search in all of them apart from the
seed, and no game of a seed twice. The index gives the game boundaries (for
a validation split by game). Without `--first-game` the output of a command
is exactly what it was.

**Deeper labels for chosen positions** (`--relabel-nodes N
--relabel-select RULE`, opt-in). A kept position chosen by the rule is
searched again with N nodes by a second searcher with its own table, and
that score becomes its label. The game is played as without the option:
the same positions, order and results. Only the chosen labels change, and
a chosen position whose deeper search finds a mate is not kept, as in the
game search. `disagree:CP` chooses positions where the static evaluation
and the search score differ by at least CP; `random:P` chooses a fraction
P by a hash of the position, the equal-compute control for an experiment.
The shard header records the rule.

Measured on 100 games, 5000 nodes, relabelling with 40,000 (one thread,
on the loaded VM, so the times are relative):

| rule | positions relabelled | mean change of the label | time |
|---|---|---|---|
| none | - | - | 27 s |
| `disagree:50` | 57.9% | 117.8 cp | 126 s |
| `disagree:100` | 36.3% | 153.0 cp | 90 s |
| `disagree:200` | 16.8% | 210.7 cp | 57 s |
| `random:0.168` | 15.6% | 66.6 cp | 54 s |

At about the same cost (about 2x), the disagreeing positions are the ones
whose 5000-node label the deeper search revises most, by 3.2 times the
random ones. Whether training on them gives a stronger network is untested.

**What a run cost.** Every run ends with
`searches in this run: game moves N nodes, S thread-seconds` and, when
relabelling, `relabelling M nodes (x% of all), R thread-seconds (y% of
all)`: the nodes and the time of the searches, summed over the threads,
kept apart for the game's moves and the deeper searches (dropped ones
included). A resumed shard reports each sitting on its own. Arms of an
experiment are matched by these, not by the fraction of positions chosen:
a rule that picks harder positions can cost more per search. On 300 games
of seed 9201 (4 threads, the idle VM, 40,000 nodes): plain 104-105 game
thread-seconds; `disagree:200` relabelled 16.2% for 180.1M nodes and
114-120 thread-seconds (two runs) on top of 106-111 for the moves;
`random:0.165` relabelled 16.4% for 181.4M nodes and 116 thread-seconds.
Equal fractions cost the same here to within the timing noise, and a
relabelled game costs about 2.2 plain ones, so the fair comparisons are
`disagree` against `random` at equal measured relabelling cost, and either
against plain data with about twice the games (`docs/PC_TASKS.md`,
section 7).

Where the time goes (callgrind, network 8, 5000 nodes, one thread): the
search is 99% of the instructions; openings, filtering, packing and output
together about 1%. Of the positions searched (36,122 in 300 games), 73.7%
are kept, 8.5% are in check, 17.3% have a capture or promotion as the best
move and 0.5% a mate or tablebase score; 16% of the random openings are
rejected (each costs a depth-6 search). So data generation gets faster with
the search itself, and at most about 6% from searching the in-check
positions with a smaller budget (which would change the games). One thread
here: 353 kept positions per second.

**Start positions** (`--book FILE`, opt-in). Each game starts from a
position of the file (one FEN or EPD position a line; the first four
fields are read, EPD operations ignored), drawn by the game's random
stream, before the `--random-plies`. Without it every game starts from the
initial position and the output is byte-identical to before (checked: text,
binary and shards). The shard header records the book's FNV-1a 64 and its
number of positions. A book steers the data: `tools/nnue/mix.py book` draws
positions of a kind from existing data (imbalanced material, pawn endings,
a piece range), `tools/make_openings.py --pgn` makes one from real games,
and `tools/mistakes.py summary --epd` from positions NAGS got wrong. 2,000
imbalanced positions from network-8 self-play, with `--random-plies 0`, gave
games with material two or more pawns apart in 63.6% of the positions
instead of 35.8%. Raise `--max-opening-score` for lopsided books.

**Other sources of records** (all write the 32-byte format):

* `tools/nnue/rescore.py label`: the games of indexed shards again, with
  NAGS's scores from a deeper search (NAGS is the only source of training
  labels: `docs/STRENGTH_PLAN.md`; another engine can be given to measure
  how its scores differ, never to train on),
  resumable, with a consistent index and the engine time per position in
  `<out>.teacher.json`; `rescore.py compare` sets label sets of the same
  games side by side (their fit to the results, the factor between their
  scales, their differences and cost). The records hold no castling
  rights, so the teacher gets rights wherever king and rook are at home
  (3.8% of positions differ from the real rights).
* `tools/nnue/tb_data.py generate`: random quiet positions of chosen
  materials with the Syzygy result under the fifty-move rule and NAGS's
  search score; `tb_data.py measure` is the matching endgame test
  (`docs/STRENGTH_PLAN.md`, item 7).
* `tools/nnue/mix.py build`: a mixture of sources in given shares, by
  whole games, with at most `--max-copies` records of a position, in
  random game order with an index; `mix.py report` describes a data set's
  composition.

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

**Provenance.** The data is fingerprinted by content (the SHA-256 of each
file, cached next to it as `<file>.sha256` by size and modification time, so
200M positions are hashed once, about a minute from a hard disk), and the
checkpoint keeps the fingerprint: `--resume` refuses data with the same
number of positions but other content (before, the count was all it
compared). After every epoch `<out>.json` records what the network is: its
architecture (format version, hidden size, output and king buckets,
mirroring, activation, layout), the settings, the data's fingerprint and
file list, the train/validation split, the loss history, the weights it
started from and the trainer's commit and SHA-256.

**Validation by complete games.** The validation loss is measured on the
last 1% of the records (at most 1M). When the file that boundary falls in
is a shard with a complete index (`nags_datagen --first-game`, its
`.games` file), the boundary moves to the nearest start of a game in it,
so no game has positions on both sides. Without an index it stays where it
is: one game may be split, and the log says so. `<out>.json` and the
checkpoint record the split ("games" or "records"), and `--resume` refuses
another split. Text data is still split by random positions, as before,
so a game's positions fall on both sides; train on `.bin` shards for a
clean validation loss.

**A fixed evaluation set.** The validation loss changes with every round's
data. `--eval-set FILE` (repeatable; 32-byte records that are not trained
on) is evaluated after every epoch and reported overall, by output bucket
(piece count) and by the side to move's king bucket, in the log and in
`<out>.json`, so networks of different rounds and designs can be compared
on the same positions, and a change that helps the middlegame and hurts
the endgame shows as such. Make one once, from a seed no training batch
uses and under a name `run_round.py` does not train on (it trains on
`selfplay*.bin`), for example
`nags_datagen --out data/evalset_9999.bin --seed 9999 --first-game 0 --games 3000`
(about 250k positions); training with or without it gives the same network
(tested). A loss is not a strength result: it can rule designs out, and
matches decide.

**The best epoch and early stopping** (off by default; training is
unchanged by them, tested byte for byte). `--keep-best` also keeps the
network of the best epoch so far as `<out without .nnue>.best.nnue`, and
`--early-stop N` stops when the loss has not improved for N epochs. The loss
is the validation loss, or with `--select-by eval:NAME` an evaluation set's
overall loss, which doesn't move with the round's data. `<out>.json`
records the best epoch, its loss and the metric. A resumed run that had
already stopped stays stopped. The cosine schedule still runs over
`--epochs`, so a run that stops early ends at a higher learning rate than a
full one.

**Staged screening** (`tools/nnue/screen.py`). Candidates (designs or
settings, as train.py arguments in a JSON file) are first trained briefly
on a subset (stage 0) and ranked on a fixed evaluation set at their best
epoch. The best `--keep` are then trained in full with `--keep-best`
(stage 1). Every run resumes, and `results.json` and the printed table
hold both stages' losses. A lower loss is not more Elo: the 512-wide
network had the lower loss and lost 48 Elo. So the tool ends by printing
the equal-time `sprt.py` commands for the stage-1 networks. Whether stage 0
ranks candidates the way stage 1 does is itself unmeasured; the table
keeps both stages' losses so that can be checked as results accumulate.

An evaluation set must be independent of the training data. Before
training, each `--eval-set` is compared with the data, and the run is
refused if the set is one of the training files (or a copy of one), if
both have shard indexes and share a game (the same seed and number), or
if more than `--eval-overlap-limit` (default 1%) of its positions with 10
or more pieces occur in the training data too. Positions are compared by
placement and side to move, whatever their scores and results, so
training games relabelled (`--relabel-nodes`) or with other results are
found like copies; that check catches them without an index. It takes one
pass over the data, and the result is kept in `<eval>.overlap.json` (an
older file, which compared whole records, is computed again). The overlap
found, overall and with 10 or more pieces, is recorded in `<out>.json`.

Why 10 pieces: independent self-play shares simple endgames, and more of
them the more data there is. Against 3,000 games of another seed (275k
positions), 23 of a 300-game set's 26,168 positions (0.088%) occurred in
the training data, all with 3 to 5 pieces (0.9% of its 2,518 positions
with 2 to 5 pieces); none of its 23,650 positions with 6 or more pieces
did. The count grew about in proportion to the training data (2, 3, 7, 23
against 100, 300, 1,000, 3,000 games), so against a round's 100M
positions a limit on all positions would refuse an independent set. A
copied or relabelled game brings its middlegame along, which independent
games do not share. (Whole records, labels included, matched none of
these.)

**Resuming keeps the evaluation setup.** The checkpoint records the
evaluation sets (name and SHA-256), `--select-by`, `--early-stop` and
`--keep-best`, and `--resume` refuses any change: another file under the
same name would put two sets' losses into one history, and the best epoch
would be chosen across them. A checkpoint written before this was
recorded resumes only if its history has no evaluation-set results.

**An earlier run is never overwritten.** If `<out>`, its checkpoint, its
`.json` or its `.best.nnue` exist, train.py refuses to start unless
`--resume` continues the run from its checkpoint (files without a
checkpoint cannot be resumed: choose another `--out`). `--profile` writes
nothing and is not affected.

**Measuring the trainer.** `--profile N` times N batches by phase (batch
preparation, forward, backward, optimiser step, reading the loss back),
each synchronised with the GPU, and then N batches end to end reading the
loss after every batch (as training does by default, which makes the CPU
wait for the GPU each time) and after every 100 (or `--sync-every`); it
writes nothing. `--sync-every N` reads the loss every N batches in
training: the network trained is byte-identical (tested), only the logged
loss's rounding differs. On the 4-core VM's CPU (no GPU; one thread, kb4
SCReLU, batch 16384): 513 ms a batch, of which data 51, forward 147,
backward 308, step 7, and the loss read nothing measurable (1.01x end to
end), as expected without a GPU. Whether the per-batch read costs anything
on the RTX 3060 is what `--profile 200` there will say; the default stays
at 1 until it does.

**The fused feature transformer** (`--ft-impl fused`, opt-in). The default
path looks up each perspective's 32 piece slots (padding included) and,
with king buckets, their 32 factor rows: 128 rows a position, and the
backward pass scatters into all of them. The fused path folds the factor
rows into the bucket rows once per batch, as the export does, and looks up
both perspectives' pieces only, in one call. It computes the same outputs
and gradients to float rounding (tested for every design), with the same
parameters, so checkpoints are interchangeable and a run may switch paths
on `--resume`. On the 4-core VM's CPU (4 threads, kb4 SCReLU, batch 16384):
28,400 positions/s → 40,400 (backward 301 → 193 ms a batch, forward
184 → 104). The RTX 3060's figure is for `docs/PC_TASKS.md` section 9.

**Quantization-aware training** (`--qat`, opt-in; implies the fused path).
The forward pass rounds the folded feature weights and the feature biases
(× 255) and the output layer (× 64) to the export's integer grid, with the
gradient of the identity (straight-through), so the network trained is the
one the engine runs: its output equals the export's weights in floating
point to 1e-4 cp and the engine's integer evaluation to its truncations
(1 cp), tested for every design. The optimiser still updates float
weights, which drift around the grid. Use it to fine-tune a trained
network (`--init-from`); a `--qat` checkpoint resumes only with `--qat`,
and `quant_error.py` loads it as trained, so its weight-rounding error is
zero by design and the `loss against the labels` line (engine column) is
the measure to compare with a plain run's export. It costs about 5% of
training speed on the CPU. Whether it plays better is
`docs/STRENGTH_PLAN.md`, item 2.

**Exact resume and warm starts are separate.** `--resume` continues the
same run (data, settings, optimiser and schedule state). `--init-from NET`
starts a *new* run from NET's weights, an exported network such as
`nets/nags.nnue` or a training `.ckpt`, with a fresh Adam state and a fresh
cosine schedule over this run's epochs; NET must have the run's
architecture (hidden size, output buckets, king buckets and layout,
activation), or the trainer stops. An exported network is read back
exactly: exporting before any training writes the same file (tested for
every design and for network 8). A resumed warm start keeps the weights it
started from (recorded in the checkpoint).

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

**Quantization error** (`tools/nnue/quant_error.py`): a training
checkpoint's float evaluation against the exported network, computed both in
floating point and as the engine computes it, so the error splits into weight
rounding and integer arithmetic (overall and by output bucket). With only an
exported network, `--simulate N` estimates the weight part by rounding its
weights once more (uniform offsets of up to half a step). Network 8 on 26,406
new positions (`nags_datagen` seed 777, 300 games, 5000 nodes; |eval| mean
427 cp): the integer arithmetic costs 0.51 cp on average (at most 1.02, no
sign changes), which is the final truncation. Simulated weight rounding (5
trials): feature transformer 1.5–1.6 cp mean (p99 5.5–5.9), output layer
7.0–10.3 cp mean (p99 23–37), with a shift of up to ±5 cp over all positions
that depends on the particular rounding. So the output weights (× 64) carry
most of the error. Whether that costs strength is untested. The real figure
needs network 8's checkpoint (`quant_error.py --net <net>.nnue.ckpt --data
...`). The first, format-compatible experiment would be an export that
corrects each output bias for the rounding of its bucket's weights,
SPRT-tested against the same checkpoint exported as now.

**Bias-corrected export** (`train.py --export-bias-correction N`, off by
default; the checkpoint and training are unchanged, only the exported
output biases differ). Each output bias is corrected so that, on N
positions (the validation set's first, then training positions) and bucket
by bucket, the exported network's output is on average the float network's.
The mean shift its rounded weights make is taken off; it includes the
feature transformer's rounding, which moves the activations. The format
is unchanged (v3). Measured on a float network close to network 8 (network
8 fine-tuned for 3 epochs on 102,545 local self-play positions, so its
weights are no longer on the integer grid), on 34,353 positions of another
seed, with `quant_error.py --net <ckpt> --exported <nnue>`:

| export | mean error | p50 | bias | sign changes |
|---|---|---|---|---|
| plain | 17.0 cp | 13.6 cp | +6.9 cp | 598 (1.7%) |
| corrected (20,000 positions) | 12.3 cp | 7.7 cp | +0.03 cp | 215 (0.6%) |

Integer arithmetic stays 0.51 cp. This is not network 8's own checkpoint,
which needs the original `.ckpt`.

The two exports of that checkpoint, in main's engine (`b4baff0`), the
corrected one against the plain one: SPRT [0, 10] at 3+0.03, H1 after 1664
games (+434 =862 -368), **+13.8 ± 10.7**. That is one checkpoint, and an
early stop overstates the estimate. At 10+0.1 the same pair passed its
non-regression SPRT [−5, 0]: H1 after 4486 games (+1005 =2517 -964),
**+3.2 ± 6.1**. So the correction costs no strength at the longer time
control, and any gain there is smaller than the 3+0.03 estimate. Network 8
itself needs its own corrected export and SPRT against the plain network
before anything replaces `nets/nags.nnue`.

`tools/nnue/reexport.py` applies the correction to an existing checkpoint
without training:
`python tools/nnue/reexport.py --ckpt <net>.nnue.ckpt --data data/selfplay_*.bin --out <net>_corrected.nnue --check <net>.nnue`.
It first checks that the checkpoint's plain export is that network byte
for byte, then corrects on N random positions and reports the error on N
others. For network 8 this needs its checkpoint from the training PC. Like
any network, the result replaces `nets/nags.nnue` only after an SPRT
against it.

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
   `--device auto`): by default `crelu-b8` (768 → 256 clipped ReLU, the design
   of networks 1–7) and `kb4-screlu-b8` (4 king buckets with SCReLU, network
   8's design); `kb8-screlu-b8` and `kb16-screlu-b8` are also available. The
   historical names `current`, `kb4`, `kb8` and `kb16` still work and name the
   files (`net_current_...`, `net_kb4_...`), so a round started with them
   resumes either way. `current` never meant network 8's design; the run
   prints the built-in network's design (`built-in network: kb4-screlu-b8`).
   With `--warm-start builtin` a design with the built-in network's
   architecture starts from its weights (`train.py --init-from`, a new run in
   files ending `_warm`), the others from scratch;
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

Known issues on Windows (at `0291007`), and what has been done about them:

* `tools/calibrate.py` and `tools/ladder.py` split `--engine` (and the
  ladder's rung engines) with POSIX rules, which drop the backslashes of a
  Windows path. Fixed: both use `tools/sprt.py`'s path handling (an existing
  file is one argument as it is). Until the fix, giving the path with
  forward slashes worked.
* Before `sprt.py` was fixed, every game of a match failed to start
  (`FileNotFoundError`), and the runner still scored the 204 failures as a
  tie and accepted H0. Fixed: the runner now starts both engines before the
  first game (and the ladder and calibration before each of their matches),
  and an engine that cannot be started makes the match invalid (exit 3, no
  decision) instead of losing its games; `run_round.py` reports such a
  match as `error`.
* `tests/python/test_sprt.py::test_engine_paths_stay_whole` failed on
  Windows because the test's own file name contained a backslash, which
  Windows reads as a directory separator (the function under test was fine),
  and `tests/python/test_ladder.py::test_anchor_and_rate` because of the
  ladder's path splitting above (its recorded bench was `None`). Both fixed,
  not yet rerun on Windows. The GNN tests need `torch_geometric`.
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
