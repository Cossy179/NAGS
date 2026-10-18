# Compatibility with networks in training

Networks are trained over days on another machine while the engine keeps
changing. A network trained with the code of any commit must keep loading
and evaluating exactly as it did then, and an interrupted training round
must keep resuming after the repository is updated. This file lists what
that depends on, what may change and what may not, and the tests that hold
it in place. Check it before every merge to `main`.

## The contract

### NNUE network files (`src/Nnue.cpp`, `tools/nnue/train.py`)

Little endian. `"NAGSNNUE"`, uint32 version, uint32 hidden size H (must
equal the build's `NAGS_NNUE_HIDDEN`, default 256), then:

| Version | Header | Adds |
|---|---|---|
| 1 | 16 bytes | one output layer |
| 2 | 20 bytes | uint32 output buckets B (1-8) |
| 3 | 92 bytes | uint32 B, uint32 king buckets K (1-32), uint32 flags (bit 0 SCReLU, bit 1 left-right mirroring), uint8 layout[64] |
| 4 (experimental) | 100 bytes | as 3, flag bit 2 pairwise inputs, then uint32 L1, uint32 L2 (1-32); after the feature transformer a hidden stack per bucket instead of the output layer (`tools/nnue/train_v4.py`) |

Body: int16 feature weights [K][768][H], int16 feature bias [H], int16
output weights [B][2H] (side to move's half first), int32 output bias [B].
The file size must be exact. Version 4 has its own trainer and reference
(`train_v4.py`); `train.py` and `read_net` refuse it, and loading it never
changes how a version 1-3 file loaded afterwards evaluates
(`test_nnue_v4.py`). Its format is fixed from the first network trained
with it, like the others.

Fixed forever for versions 1-3: QA = 255, QB = 64, SCALE = 400; feature =
own/opponent × 384 + type (P N B R Q K) × 64 + square (a1 = 0; `^56` for
Black's perspective, then `^7` when mirrored); the king bucket is looked up
on the perspective's king square before mirroring; output bucket =
clamp((pieces − 1) · B / 32, 0, B − 1); clamped ReLU to [0, 255]; SCReLU is
`int16(c·w)·c` summed in int32 and truncated by 255 before the bias; the
result is `trunc((sum + bias) · 400 / 16320)` from the side to move's view.

**May change:** a new format version in a new parse branch; faster kernels
with identical integers; larger limits (B, K); clamping the *result* used by
the search (`eval::kEvalLimit`, far outside any real network's range).
**May not change:** anything above for versions 1-3, the meaning of the v3
flag bits or layout, or validation that would reject a file accepted today.

### Training data

32-byte records: uint64 occupancy, 16 bytes of 4-bit piece codes (P N B R Q
K p n b r q k = 0-11, ascending squares, low nibble first), int16 score
(White's view), uint8 result (0/1/2), uint8 side to move, 4 reserved bytes
(written 0, ignored when read). Text lines: `<FEN> | <score> | <result>`.
`nags_datagen` keeps its command line, and with `--threads 1` its output
is a function of the binary, the embedded network and the arguments; a
change that keeps the bench node count also keeps it byte-identical
(checked for every non-functional commit).

### `tools/nnue/train.py`

Its argparse defaults, constants (QA, QB, SCALE, WEIGHT_CLIP = 1.98), king
layouts, model parameter names and shapes, target (λσ(score/400) + (1 −
λ)·result, MSE of σ(output)), optimizer and per-batch cosine schedule. New
options must default to today's behaviour. Checkpoints (`<out>.ckpt`) keep
their keys; new keys may be added, and a newer trainer accepts checkpoints
without them exactly as before.

### `tools/nnue/run_round.py` and `tools/sprt.py`

The design names and their arguments (`current`, `kb4`, `kb8`, `kb16`), the
data/network/log file names, the resume rules, and the `sprt.py` status
and decision lines that `--resume` and `run_round` parse.

### GNN side (`chess_graph.py`, `gnn_evaluator.py`, `rpc_server.py`)

`nags-gnn-v2` checkpoints, the 27 node features and graph construction of
`ChessGraph`, the 4096-entry from-to policy (promotions share an index),
and the RPC protocol: the C++ client sends a FEN and its legal moves and
reads `move_priors` (one per move, same order), `value` (side to move,
[-1, 1]) and `uncertainty`. New model formats get a new checkpoint format
string and are served through the same protocol.

`nags-gnn-v3` (`gnn_v3.py`, `relational_graph.py`, `policy_encoding.py`) is
such a format, next to v2: 4228-entry promotion-aware policy, graph
`relational-v1`. `load_checkpoint` still accepts only v2;
`load_any_checkpoint` gives each format its own builder and encoding.
Replies gained one field, `model_format`; with a v2 checkpoint or no model
every other field is as before (`policy_dim` 4096), and for v3 `policy_dim`
is 4228. Any change to the v3 graph features or relations needs a new
`graph` version, and to its move encoding a new `policy_encoding`, so that
v3 checkpoints in use keep their meaning (the loader refuses versions it
does not know). If `gnn_v3.py` cannot be imported, `rpc_server.py` logs it
and serves v2 checkpoints through its own legacy path. `promote` writes a
v3 network to `models/production_v3.pth` (a name the legacy `*_model_*.pth`
lookups never match), never over the v2
`production_model.pth`; `supervised_legal_mask` (default off) is the only
change to the legacy training steps, and it is opt-in.

Serving speed-ups (`relational_fast.py`, `gnn_infer.py`, the new
`rpc_server.py` options) change how fast a reply is computed, never what it
says. `relational_fast.build_arrays` must give exactly
`relational_graph.build_graph`'s features, edges and edge order (the
reference stays and is what training uses), and `TorchBackendV3` exactly
`RelationalGNNEvaluator.evaluate`'s outputs, MC-dropout draws included;
`rpc_server.py` serves v3 through them by default (`--backend auto`), and
v2 networks only through the legacy path. The ONNX backend (`--backend
onnx`, optional dependency) agrees to float precision and is opt-in, as
are `--batch-wait-ms` and `--cache-size`. The protocol is unchanged
(several FENs per request were always allowed; the engine now sends them
with `NNBatch` > 1). `--threads` defaults to 1, which changes the server's
speed, not its replies beyond float rounding. A change to the v3 model's
layers must be mirrored in `TorchBackendV3` and the ONNX export (the tests
fail otherwise); a change to the graph needs a new `graph` version anyway.

The engine's `NNBatch` (default 1), `NNPipeline` (default off) and
`NNCache` (32 MB) only affect the hybrid's MCTS arm, which is off by
default; with `NNBatch` 1 its search is the one-request-per-simulation MCTS
as before, the cache only skipping repeated round trips. The training
pipeline sets none of them, so self-play games are played as before.

### Engine options used by the tools

`EvalFile` (prints `NNUE evaluation: <name>` or `EvalFile: <error>`),
`UseNNUE`, the `eval` command's `info string eval <cp>` output, and the
`bench` output (`Nodes searched  :`) — `bench` uses the default hash size,
whatever `Hash` is set to.

## Tests that hold it

| Test | Holds |
|---|---|
| `tests/python/test_compat.py` | Networks of every format (v1, v2, v3 SCReLU / mirror / 4 and 16 king buckets) with evaluations recorded from the engine and trainer of `bb1e03c`, in both the trainer's reference and the engine, through incremental updates (castling, en passant, promotions, king-bucket changes); the trainer's defaults, constants and layouts; `run_round.py`'s designs; resuming a checkpoint written by the `bb1e03c` trainer; loading and serving a `nags-gnn-v2` checkpoint with the same priors and value. Fixtures in `tests/data/compat` (never regenerate them). |
| `tests/python/test_nnue.py` | Engine = trainer reference on freshly trained networks of five designs; binary records round-trip; datagen binary = text; resume reproduces a full run and refuses changed settings. |
| `tests/cpp/nnue_tests.cpp` | Incremental accumulators = refreshed ones through random games (also across null moves); colour and left-right symmetry; the embedded network loads in this build; evaluations stay below the TB/mate scores. |
| `tests/python/test_relational_fast.py` | The fast graph builder equals `relational_graph.build_graph` exactly (features, edges, edge order, legal mask) on about 4,000 positions and special cases, and its batches equal `Batch.from_data_list`'s. |
| `tests/python/test_gnn_infer.py`, `test_rpc_fast.py` | The torch backend equals `evaluate()` bit for bit (MC dropout included) and its replies the legacy server's byte for byte; ONNX within 1e-5; batching, pipelining and the cache keep replies and their order. |
| `nags_datagen` identity | Not a permanent test (functional search changes legitimately change the games): for each non-functional commit, `nags_datagen --threads 1` with a fixed seed must write the same bytes as before. |

## Before merging to `main`

1. A network in training (any of the formats above, any `NAGS_NNUE_HIDDEN`)
   still loads, and evaluates the same (`test_compat.py`).
2. The trainer still resumes its checkpoints, with unchanged defaults.
3. Nothing under `nets/` or the training data is replaced without a
   strength test (`nets/nags.nnue` changes only through `run_round.py`).
4. Functional search changes are in `main` only with SPRT evidence
   (`docs/TESTING.md`); they change future training games but not their
   format.
5. New architectures (NNUE or GNN) are added as new, versioned formats
   next to the old ones, never in their place.
