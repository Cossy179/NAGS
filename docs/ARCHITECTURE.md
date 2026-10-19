# NAGS architecture

This document describes how the engine and the learning components work. It
replaces the earlier per-feature reports (`*_REPORT.md`, `README_BASIC.md`),
which described code that has since been rewritten.

## Source layout

```
src/
  ChessTypes.h      Piece / Color / Move types shared by everything (a1 = 0 ... h8 = 63)
  BitOps.h          portable lsb / msb / popcount (MSVC, GCC, Clang, fallback)
  Board.h/.cpp      bitboard board with ray-based sliding attacks   (nags_basic)
  FastBoard.h/.cpp  bitboard board with magic bitboards + mailbox   (nags, nags_fast, nags_enhanced)
  Eval.h            static evaluation: NNUE when active, else material, piece-square tables, bishop pair
  Nnue.h/.cpp       NNUE network loading, accumulators and evaluation
  Syzygy.h/.cpp     Syzygy tablebase probing (third_party/fathom)
  TT.h/.cpp         lockless transposition table shared by search threads
  Search.h          iterative-deepening alpha-beta (template over the board type)
  Uci.h/.cpp        UCI front end: worker-thread search, go parsing, time management
  ClassicEngine.h   UCI engine around Search (all engines; nags adds the hybrid on top)
  NAGS.h/.cpp       hybrid controller: alpha-beta plus a GNN-guided MCTS arm, evaluators
  Net.h/.cpp        TCP line client with timeouts + minimal JSON helpers
  MetaClient.h/.cpp client for meta_learner.py
  main.cpp          nags (hybrid) entry point
  *_main.cpp        nags_basic / nags_fast / nags_enhanced entry points
  rpc_client.cpp    smoke test for rpc_server.py
tests/cpp           perft, hashing, FEN, draws, eval symmetry, TT, time manager, search
tests/uci           end-to-end UCI scripts run by ctest
tests/python        pytest suite for the Python side
```

## Boards

`Board` and `FastBoard` expose the same interface (legal move generation,
optionally captures/promotions only, make/unmake, Zobrist hash, FEN, draw
detection, perft), so the search and evaluation are templates that work with
either.

* Legal moves are generated as pseudo-legal moves and filtered by making each
  one on a scratch copy of the position (without the game history, so the
  cost does not grow with game length).
* The Zobrist hash is updated incrementally for every change, including
  captured pieces, en-passant captures and castling rooks. The tests compare
  it with a freshly computed hash after every move of thousands of random
  game positions.
* `FastBoard` finds its magic multipliers at startup by seeded trial and error
  (deterministic, about 0.25 s) instead of trusting hard-coded constants.
* Draws: fifty-move rule (checkmate takes precedence), repetition and
  insufficient material. Inside the search a single earlier occurrence of the
  position counts as a repetition.
* `setFromFEN` validates the FEN (8x8 board, exactly one king per side, no
  pawns on the back ranks, side to move, castling, en passant, the side not to
  move is not in check) and leaves the board unchanged on failure. Castling
  rights without the king/rook on its home square are dropped.

## Search (`Search.h`)

* Iterative deepening with aspiration windows (from depth 5) and principal
  variation search.
* Extensions: check extension; singular extension of the TT move (depth ≥ 6,
  verified by a reduced search without it, with a multi-cut when even the
  alternatives beat beta). Far below the singular margin (by more than 20)
  the extension is two plies outside PV nodes, at most six times along a
  line; when the move is not singular but its TT score already beats beta it
  is searched a ply shallower.
* Correction history: per worker, running averages of how far the static
  evaluation was from the search result in positions with the same pawn
  structure, the same non-pawn pieces of each side, and after the same
  previous move (per side to move). Their weighted mean, at most ±256 cp,
  corrects the static evaluation used for pruning, "improving" and the
  quiescence stand-pat. Updated at nodes not in check whose result is not
  decided by a capture and lies on the right side of the evaluation.
* Pruning and reductions (outside PV nodes and check where it matters):
  reverse futility pruning (depth ≤ 6), null-move pruning (R = 3 + depth/6),
  late-move pruning and futility pruning of quiet moves (depth ≤ 3),
  logarithmic late-move reductions, mate distance pruning, and internal
  iterative reduction (one ply less at depth ≥ 4 without a TT move).
  "Improving" (the static evaluation is higher than two plies earlier)
  tightens reverse futility pruning, and when not improving late-move
  pruning keeps half as many quiet moves and reductions are one ply deeper.
  Quiet moves with a history score below −4096·depth are pruned (depth ≤ 3)
  and reductions shrink (or grow) by one ply per 8192 of history. SEE
  pruning (depth ≤ 8, after the first legal move, not the TT move) skips
  quiet moves losing more than 60·depth and captures losing more than
  20·depth² in the static exchange.
* Move ordering: transposition-table move, then winning/equal captures by
  MVV/LVA (with promotions), killer moves, the countermove, losing captures
  (negative static exchange evaluation), and quiet moves by a history table
  with a malus for quiet moves that failed to cut off (bonus
  min(150·depth − 100, 1500), with values kept within ±16384), plus
  continuation history: the same for (piece, target) pairs following the
  move one and two plies earlier. Their sum orders quiet moves and drives
  history pruning and reductions. A staged move picker gives the moves in
  this order (ties in generation order) without scoring everything up
  front: captures get their static exchange evaluation only when the
  capture stage is reached, and each pick scans one stage. Killers,
  countermove and quiet scores are taken when the node starts.
* Quiescence search over captures and promotions with delta pruning and
  without losing captures (SEE < 0). When in check, all evasions are
  searched and there is no stand-pat. Results go to the transposition table
  at depth 0; non-PV nodes take cutoffs from it, its move is tried first,
  and a stored score bounded on the right side replaces the stand-pat.
* Time management: no new iteration after a soft limit that grows when the
  best move just changed or the score dropped and shrinks when the best move
  is stable or took most of the last iteration's nodes; a hard limit aborts
  the search.
* Syzygy tablebases (WDL in the search, DTZ at the root), MultiPV and
  pondering.
* Transposition table entries record whether the score is exact or a bound,
  and mate scores are stored relative to the node, so they stay correct at
  other plies. The root never takes a TT cutoff, so every search returns a
  move from its own principal variation.
* If time runs out mid-iteration, the result of the last completed iteration
  is used, unless a root move in the interrupted iteration was fully searched
  and proved better than the previous best.
* `Searcher` runs the main thread plus `Threads - 1` Lazy SMP helpers that
  share the transposition table (`nags_enhanced`).

### Transposition table

Each entry holds two 64-bit atomics: packed data (move, score, depth, bound,
generation) and `key XOR data`. A reader accepts an entry only if the XOR
matches, so a torn read from a concurrent write is rejected rather than
misread. Sizes are rounded *down* to a power of two entries (16 bytes each), so
`Hash` is an upper bound on memory. `hashfull` is the share of sampled entries
written in the current search.

### Time management (`Uci.cpp`)

For `wtime/btime` the target is `time / movestogo (default 30) + 3/4 inc`,
after subtracting `Move Overhead`. No new iteration starts after 55% of the
target, scaled in the search (`Searcher::search`) by the best move's
stability (×2.0 right after it changed, down to ×0.85 after four stable
iterations), by score drops of more than 20 cp (up to ×1.5) and, from depth
7, by its share of the last iteration's nodes over all aspiration attempts:
×(1.5 − share)·1.35, from ×0.68 for a move that took all of them to ×2.0
for one that took none. The hard limit (abort) is the smaller of 3x the
target and half the remaining time (90% of the remaining time when
`movestogo` is 1). `movetime`
uses the given time minus the overhead (without the node-share factor).
`go infinite` (or a bare `go`)
searches until `stop`, and `bestmove` is never sent before that.

### Evaluation (`Eval.h`)

Material, piece-square tables and a bishop-pair bonus. The king's table is
tapered from a middlegame table to an endgame (centralisation) table with the
amount of material left. The tables are written in the usual printed layout
(rank 8 first, White's view), so a White piece on square `s` reads entry
`s ^ 56`. The tests check that evaluation is colour-symmetric and identical on
both boards.

## The hybrid controller (`NAGS.cpp`)

`nags` is `nags_enhanced` (FastBoard, NNUE, the full search, transposition
table, threads, tablebases, MultiPV, pondering) with a second search arm.
Per move:

1. **Meta-learner.** With `UseMetaLearner`, the controller sends the FEN, the
   remaining clock time, the previous root uncertainty and the tactical ratio
   (the share of legal moves that capture, promote or give check). It gets
   back three deltas in [-1, 1]:
   * verification depth: the alpha-beta depth + round(delta)
     (`dfs_depth_delta`);
   * MCTS simulation budget: 2000 + 1000·delta (`mcts_budget_delta`);
   * PUCT exploration constant: 1.4 + 0.5·delta
     (`bandit_exploration_delta`; the name predates the parallel design).

   In self-play, `MetaExploration` adds Gaussian noise to the deltas so the
   meta-learner sees varied choices.
2. **Is MCTS worth running?** Not with less soft time than `MctsMinTime`
   (default 1000 ms: every network evaluation is a round trip to
   `rpc_server.py`, so short moves get only a handful of simulations), not
   while the GUI is pondering, and not with a single legal move. Otherwise
   the root position is sent to `rpc_server.py`; if the GNN does not answer,
   MCTS does not run either and the move is the alpha-beta search's.
3. **Both arms in parallel.** The alpha-beta search runs as in
   `nags_enhanced`, with the time already used (meta-learner, root
   evaluation) subtracted from its limits and, when MCTS runs, 85% of the
   rest, so some time is left for verification. Each network request waits
   at most a tenth of the hard limit, so a slow answer cannot overrun the
   move. Meanwhile an MCTS thread runs PUCT simulations until the
   alpha-beta search finishes or the simulation budget is used. Leaves are
   evaluated by the GNN (FEN plus legal moves in, value plus priors for
   those moves out; the heuristic evaluator takes over if the service fails
   mid-search): one request per leaf by default, or `NNBatch` leaves per
   request, selected with virtual loss (see "Neural inference" below).
   Network answers are kept in an evaluation cache (`NNCache`, 32 MB), so a
   position met again costs no round trip. Each node stores its value from the
   point of view of the player who moved into it, so selection maximises for
   the side to move. Checkmate, stalemate and draws are terminal nodes. The
   tree is capped at about 2M nodes.
4. **Decision: MCTS proposes, alpha-beta verifies.** The engine plays the
   alpha-beta move unless all of these hold:
   * MCTS was guided by the network;
   * at least half the root visits (and at least 64) went to a different move;
   * the alpha-beta score is not a mate score;
   * a null-window verification search (on the shared transposition table)
     shows that move is worth at least the alpha-beta score minus 25 cp,
     within the remaining time.
5. **Reporting.** Standard `info` lines from the alpha-beta search, an
   `info string nags ...` summary (MCTS simulations and top move, and why
   the move was chosen), and an `info string nags_meta ...` line with the
   meta-learner inputs and the deltas used. The training pipeline reads the
   `nags_meta` line to build meta-learner samples.

Earlier versions ran both arms on one thread, with a Thompson-sampling bandit
handing out time slices, and searched on the slower ray-based board with the
hand-written evaluation. That design scored 1/60 against `nags_enhanced`
(see `docs/TESTING.md`); the current one gives up nothing when the network
has nothing to add.

Network failures never stall the engine. Every socket operation has a
timeout, writes cannot raise SIGPIPE, and a failed connection is retried at
most once a minute (every 30 s for the meta-learner).

## Neural network (`chess_graph.py`, `gnn_evaluator.py`)

A position becomes a graph with nodes and edges as follows:

* **Nodes:** 64 squares, one per piece, and 13 metadata nodes (side to move,
  4 castling rights, 8 pawn files).
* **Edges:** occupancy (piece–square), attacks (piece–attacked square), pawn
  file membership, pawn chains and side-to-move membership.
* **Node features (27):** node kind, piece type, colour, square coordinates,
  side to move, castling flags, pawn-file one-hot and an en-passant flag.

A residual GCN stack produces node embeddings, which are padded per graph for
batching. Two transformer heads read them:

* **Policy:** logits over the 64x64 from–to pairs, from bilinear scores of
  the square embeddings. Promotions share their from/to index.
* **Value:** tanh of a masked mean-pooled embedding, from the side to move's
  point of view in [-1, 1]. The uncertainty is the standard deviation over
  Monte-Carlo dropout samples.

Batched and single-position inference give identical results. Checkpoints
store the model configuration alongside the weights, so the server rebuilds
the model with the right dimensions.

## Neural network v3 (`nags-gnn-v3`, experimental)

A second network design, added next to the one above rather than replacing
it. **v2 checkpoints keep working unchanged:** `chess_graph.py`,
`gnn_evaluator.py` (format `nags-gnn-v2`, 4096-entry policy, `move_index`,
`legal_move_priors`, `save_checkpoint` / `load_checkpoint`), the training
steps `parse` … `full` and their defaults, `meta_learner.py`,
`training_config.json` and the RPC protocol are as they were, and
`tests/python/test_compat.py` pins a frozen v2 checkpoint's outputs. Nothing
uses v3 unless a v3 checkpoint is passed to `rpc_server.py --model` (or to
`training_pipeline.py --step evaluate --model`).

| Module | Contents |
|--------|----------|
| `policy_encoding.py` | promotion-aware move encoding (4228 entries) |
| `relational_graph.py` | `RelationalChessGraph`: typed, directed relations built with python-chess |
| `gnn_v3.py` | `RelationalGNNEvaluator`, v3 checkpoints, `load_any_checkpoint` and the per-format adapters |
| `distillation.py` | label rows, distillation targets and losses |
| `relational_fast.py` | the same graphs as `relational_graph.py`, bit for bit, as numpy arrays, about 3x faster (serving) |
| `gnn_infer.py` | inference backends for v3: torch (bit-identical to `evaluate()`) and ONNX Runtime |
| `tools/gnn_bench.py` | latency / throughput of v2 and v3, graph construction and model separately, per serving backend |

### Move encoding

The v2 policy gives a move the index `from·64 + to`, so `e7e8q`, `e7e8r`,
`e7e8b` and `e7e8n` share one probability. v3 keeps those 4096 indices for
every move that is not an underpromotion (queen promotions, castling as the
king's UCI from/to such as `e1g1`, en passant as the pawn's from/to) and adds
`4096 + pair·3 + {n: 0, b: 1, r: 2}` for underpromotions. `pair` numbers the
44 from/to pairs a promotion can use: White's (rank 7 → 8) then Black's
(rank 2 → 1), by the pawn's file a–h, and for one pawn by target file from
left to right (capture towards a, push, capture towards h); so pair 0 is
`a7a8`, 1 `a7b8`, 2 `b7a8`, …, 21 `h7h8`, 22 `a2a1`, …, 43 `h2h1`.
`POLICY_V3_DIM = 4228`. Every legal move has its own index (tested on
thousands of random positions, many with promotions); `index_to_uci_v3`
inverts it, given the board to tell `e7e8q` from a rook's `e7e8`.

### Graph

Built from a `chess.Board` (a FEN, or a board with its move history), so
attacks, pins, castling and en passant follow the rules exactly.

* **Nodes:** 64 squares (node *i* is square *i*), one per piece (in square
  order), one global node; `node_kind` marks which is which.
* **Relations** (21 directed edge types, `RELATIONS`): piece → its square and
  back; piece → enemy piece it attacks, and attacked-by back; piece → own
  piece it defends, and back; piece → empty square it controls, and back;
  side-to-move piece → square it can legally move to, and back; pawn ↔ pawns
  on its file; pawn → own pawn it protects (chain), and back; passed pawn →
  each square of its path to promotion, and back; king → the squares around
  it, and back; pinned piece → the slider pinning it to its king, and back;
  global node → every piece, and back.
* **Node features** (32 columns, documented one by one in `FEATURE_COLUMNS`),
  relative to the side to move ("own"/"opponent", ranks counted from the
  mover's side, so a position and its colour-flipped mirror image look the
  same apart from one `white_to_move` flag): node kind, piece type, own /
  opponent, file and relative rank, attackers of each side on the square
  (for a piece, attackers of its own colour are its defenders), legal
  en-passant square, pinned, mobility, legal moves of the piece, passed
  pawn, castling rook / king; on the global node: in check, castling rights
  of both sides, half-move clock / 100, repetitions of the position in the
  board's history (0 for a bare FEN), number of legal moves, material.

### Model

The trunk is a stack of R-GCN layers (basis decomposition, `num_bases`
shared matrices; messages averaged per relation, so a busy relation such as
"controls" does not drown a rare one such as "pins") with residual
connections, LayerNorm and a feed-forward block. It is written with one
gather and a few `index_add` calls, without a Python loop over relations or
torch-scatter. Heads:

* **Policy** (4228 logits): bilinear from-square × to-square scores of the
  square embeddings (plus the global embedding); an underpromotion's logit is
  its from/to logit plus a learned per-piece offset from the two square
  embeddings (initialised at −2).
* **WDL** (3 logits: win / draw / loss for the side to move) from the global
  node and the mean of all nodes; value = P(win) − P(loss).
* **Auxiliary search score** (optional, tanh), trained towards the search's
  expected score.

`evaluate()` returns what the v2 evaluator returns, with the same meaning:
policy probabilities over all entries (here 4228; with the legal masks that
`GNNAdapterV3.build_batch` attaches, as the server does, all of the mass is
on legal moves), the value from the side to move's view in [−1, 1], and the
uncertainty as the standard deviation of the value over `mc_samples`
Monte-Carlo dropout samples of the WDL head (not the WDL entropy, which is
also high in quiet drawn positions).

The value scale differs from v2's. A distilled v3 value is P(win) − P(loss)
≈ 2σ(s/400) − 1 = tanh(s/800) for a search score s, while v2's head is
trained towards game results. The engine uses the value only inside its MCTS
tree and converts it to centipawns, with 400·atanh(v), only for the
`info string` line, so a v3 network's printed MCTS score is about half the
search score; the search is unaffected. The meta-learner's `uncertainty`
input comes from the served network, so a meta-learner trained alongside a
v2 network must be retrained before it is used with a v3 one (it is off by
default).

### Checkpoints and serving

`save_checkpoint_v3` writes `{'format': 'nags-gnn-v3', 'config': {...,
'policy_encoding': 'v3', 'graph': 'relational-v1'}, 'state_dict': ...}`.
`load_checkpoint_v3` rejects anything else (and a config naming another
encoding or graph version); the legacy `load_checkpoint` still rejects
everything but `nags-gnn-v2`. `load_any_checkpoint(path)` returns an adapter
holding the model with its own graph builder and move encoding (v2:
`GNNEvaluator` + `ChessGraph` + `legal_move_priors`, the exact legacy code
path; v3: `RelationalGNNEvaluator` + `RelationalChessGraph` +
`legal_move_priors_v3`), so one format's outputs are never read with the
other's encoding.

`rpc_server.py --model` takes either format. Replies keep their fields;
`policy_dim` is the served model's (4096 for v2 and for the untrained
fallback, exactly as before; 4228 for v3) and a new `model_format` field
names the format. The engine reads only `move_priors` (one per requested
move, in the request's order), `value` and `uncertainty`, so a v3 network
works with the engine unchanged, and an underpromotion gets its own prior.

### Search-policy distillation (`label`, `distill`)

`label` takes positions from the existing jsonl datasets (`parse` and
self-play outputs, or any jsonl with a `fen` field; `--data`), keeps
distinct ones, and runs the alpha-beta engine (`nags_enhanced` unless
configured otherwise; one thread, hash cleared per position) with MultiPV k
at a fixed node budget. Each row has the FEN, the legal moves, the top-k
moves with their exact scores (`{"cp": x}` or `{"mate": n}` from the side to
move's view; mates are never folded into centipawns), `rest_upper_bound`
(the other moves are only known to score at most the k-th score), depth,
nodes, and the source game's result when known. NAGS's MultiPV searches
every line with a full window, so the k scores are exact for that search.

`distill` trains `RelationalGNNEvaluator` on those rows:

* **Policy:** cross-entropy against a soft target over the legal moves only
  (log-softmax restricted to the legal mask). The target is
  `softmax(score / T)` (T = `temperature_cp`, 50 cp) over the k scored
  moves, with mates mapped to large but finite scores (mate in n → 3000 −
  10·(n − 1), beyond every clipped cp score). The unscored moves are only
  known to score at most the k-th score, so none of them can be given a
  value of its own, but together they are not worthless either: they share
  one censored "rest" bucket of weight `rest_mass_scale` (0.02) × their
  number × exp((k-th score − best)/T), and the loss for it is
  −p_rest · log Σ_{unscored} π(m), so the network decides how to spread
  that mass among them. With `rest_mass_scale` 0 this is the plain target
  over the k moves.
* **Value:** cross-entropy of the WDL head against a WDL target from the best
  line's score s with a logistic model:
  `P(win) = σ((s − a)/b)`, `P(loss) = σ((−s − a)/b)`, `P(draw) = 1 − P(win) − P(loss)`,
  a = 400, b = 250, which puts the expected score P(win) + P(draw)/2 within
  0.034 of σ(s/400) at every score, the scale of the NNUE trainer, with a 66% draw
  probability at s = 0. `result_weight` (default 0) blends in the game
  result when the row has one. The auxiliary head regresses the search-only
  P(win) − P(loss).
* **Validation**, split by source game (by position when there is no game
  id): policy NLL, top-1 agreement with the search's best move, WDL NLL and
  value MSE, logged per epoch and stored in the checkpoint's `extra`. The
  best checkpoint is `models/gnn_v3_distill_<stamp>.pth`, a name the legacy
  steps' `*_model_*.pth` lookups never pick up.

Defaults are `DISTILL_DEFAULTS` in `training_pipeline.py`, merged under the
config's `distill_params` key; `training_config.json` does not change.

```bash
./run_training.sh label --data data/training_data.jsonl   # or: python training_pipeline.py --step label
./run_training.sh distill                                 # newest data/distill_*.jsonl
python rpc_server.py --model models/gnn_v3_distill_<stamp>.pth
./run_training.sh evaluate --model models/gnn_v3_distill_<stamp>.pth
```

### Speed

`tools/gnn_bench.py` times graph construction (on the CPU) and the model
(`evaluate`, 3 MC-dropout samples) separately for both formats at batch
sizes 1, 8, 64 and 256, on the CPU and on CUDA when available. Untrained
models of the default sizes (v2: 1.34M parameters, v3: 1.03M), one torch
thread, on a 4-core machine already loaded by an engine match (load
average 4.3), median of 3 runs, positions per second:

| batch | v2 build | v2 model | v2 total | v3 build | v3 model | v3 total |
|------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 1     | 216 | 14.4 | 13.5 | 245 | 62.2 | 49.6 |
| 8     | 304 | 16.0 | 15.2 | 385 | 137.5 | 101.3 |
| 64    | 446 | 16.8 | 16.2 | 429 | 135.4 | 102.9 |
| 256   | 393 | 13.6 | 13.1 | 526 | 63.5 | 56.6 |

A single position (the engine's per-leaf request) takes 74 ms with v2 and
20 ms with v3. Most of v2's time goes to its transformer heads; v3 has
none. Building the graphs costs about the same in both (2–5 ms a
position in Python), so at small batches it is a large share of a v3
request.

## Neural inference (`relational_fast.py`, `gnn_infer.py`, `rpc_server.py`, `NAGS.cpp`)

How the MCTS arm gets its network evaluations, and how fast. Nothing here
changes a checkpoint, the graph or the meaning of a reply; the v3 paths are
tested to give the reference's results bit for bit (or, for ONNX Runtime,
to float precision).

### Where the time went

Measured on the 4-core VM with an untrained default-size v3 network
(weights do not change the timing), `--mc-samples 3`, and `nags` searching
`go movetime 3000` from a Ruy Lopez position. Before these changes the
MCTS arm managed **41 simulations per second**, and each round trip to
`rpc_server.py` took 24 ms:

* **Threads.** The server ran torch with its default intra-op thread count
  (4 here, 6 on the Windows PC). A request is one position, about 200 small
  operations, and with 4 threads each one synchronises the pool: a request
  took 22.7 ms with 4 threads and 7.1 ms with 1 or 2. On the PC the
  alpha-beta threads compete for the same cores as well.
* **The request path** (one position, one thread): network 4.2 ms, graph
  construction 0.74 ms, collation into a torch_geometric batch 0.28 ms,
  switching the module between train and eval mode for MC dropout 0.35 ms,
  and the legal-move mask 0.07 ms (it parsed the FEN a second time).
* **The network itself** is about 168 MFLOP a position (90 nodes and 393
  edges on average; six R-GCN layers of width 128, mostly the 64 square
  nodes), which one core computes at about 40 GFLOP/s. So on the CPU the
  network is close to its floor: ONNX Runtime takes 3.5 ms at batch 1 where
  torch (without the mode switching) takes 4.2, and is no faster at batch 8
  or 64. On a GPU the same work is a small fraction
  of a millisecond per position in a batch, but only if the positions come in
  batches, and then graph construction in Python is what limits the rate.
* **One request per simulation**, and no evaluation was ever reused.

### The server

* `--threads` (default 1) sets torch's (and ONNX Runtime's) thread count.
* `relational_fast.build_arrays` builds the relational-v1 graph as numpy
  arrays: flat edge lists instead of a call per edge, features filled column
  by column, pins found from the sliders on the king's lines, and the legal
  moves of the same board reused for the mask. The reference
  (`relational_graph.build_graph`) is unchanged; the fast builder must equal
  it exactly, features, edges and edge order included
  (`test_relational_fast.py`, about 4,000 positions with promotions, en
  passant legal and illegal, pins on every line, castling rights without
  rooks, checkmate, stalemate and move histories with repetitions). Graph
  preparation for one position: 1.62 → 0.50 ms.
* `gnn_infer.TorchBackendV3` runs the model's own layers in the order
  `evaluate()` does, on those arrays, without switching modes (MC dropout
  with `F.dropout`, which draws the same random numbers as `nn.Dropout`).
  Its results are bit-identical to `evaluate()`, the Monte-Carlo samples
  included (`test_gnn_infer.py`), and its replies byte-identical
  (`test_rpc_fast.py`), so it is the default for v3 networks (`--backend
  auto`). v2 networks keep the legacy path.
* `gnn_infer.OnnxBackendV3` (`--backend onnx`, `pip install -r
  requirements-onnx.txt`) exports the network once per set of weights
  (`--onnx-cache`) with dynamic node, edge and batch counts. The exported
  graph takes the square and global nodes as gather indices, so it has no
  boolean indexing or data-dependent shapes, which TensorRT needs. MC dropout
  of the small WDL head is done in numpy (same distribution, different
  draws). Agreement with torch: about 1e-7.
* `--batch-wait-ms` evaluates the requests of all connections that arrive
  within the window as one batch, and lets each connection send its next
  request before the previous reply (the connection's graphs are built while
  the previous batch is evaluated; replies stay in order). It stops waiting
  once every open connection has a request in. It is for several clients
  sharing one GPU (parallel self-play games): with one engine on the CPU the
  hand-offs between threads cost about 3 ms a request.
* `--cache-size` keeps replies to repeated requests.
* The server sets `TCP_NODELAY`. With Nagle's algorithm a reply that follows
  another (pipelined requests) waited for the client's acknowledgement of
  the first, which Linux delays by up to 40 ms (Windows: 200 ms) when the
  client has nothing to send: in testing, the last reply of every pipelined
  search timed out. The engine also re-arms `TCP_QUICKACK` after each read
  on Linux, so it does not depend on the server for this.

One-position round trip, measured from a client: 24.8 ms before, 5.6 ms
with the defaults now (bit-identical replies), 4.7 ms with `--backend onnx`.

### The engine

* `NNBatch` (default 1, the original one request per simulation): the MCTS
  thread selects up to that many leaves before asking the network. Every
  node on a selected path is charged a virtual loss (a visit that lost,
  −1), so the following selections spread out; a selection that reaches a
  leaf already queued is a collision, and after as many collisions as the
  batch size the batch is sent as it is. Terminal positions and cached
  positions are backed up at once. When the reply comes, each leaf is
  expanded, its value backed up and the virtual loss taken back. The batch
  is also capped so that its expected round trip (from the measured time per
  position) stays under 40% of the request timeout, a tenth of the move's
  hard limit, so a slow server shrinks the batches instead of timing out.
* `NNPipeline` (default off): the next batch is sent before the previous
  one's reply arrives (two requests in flight), so the engine's selection
  and the server's work overlap; useful when the network is fast (GPU), not
  when it is the bottleneck.
* `NNCache` (MB, default 32; 0 = off): network answers by position, keyed by
  the Zobrist hash and the half-move clock (a v3 input), with the legal-move
  list checked as well. Within one search it catches transpositions (11% of
  the evaluations in the measurement above); the next move's search starts
  in the previous tree, which the cache mostly holds. It is cleared by
  `ucinewgame` and when the server address changes. With `--mc-samples` > 1
  a cached value is one draw of the MC dropout, as the network would give.
* If the server fails mid-search the requests in flight are evaluated by the
  heuristic and the rest of the move continues as with `NNBatch` 1 (the
  service is retried after a minute), as before.

The `info string nags` line gives `rpc_positions`, `rpc_batch` (positions
per request), `cache_hits` and `mcts_collisions` after the existing fields.

MCTS simulations per second, same measurement as above (CPU only):

| Setting | Simulations/s | Round trip |
|---|---:|---:|
| before | 41 | 24 ms (1 position) |
| `NNBatch 1` (server and cache defaults) | 196 | 5.7 ms |
| `NNBatch 4` | 252 | 16 ms |
| `NNBatch 8` | 250 | 32 ms |
| `NNBatch 16` | 258 | 61 ms |

On this CPU the network caps the rate at about 250 positions a second;
batching only removes the per-request overhead. The GPU case has not been
measured (no GPU here), and that is where batching pays: `python
tools/gnn_bench.py --models v3 --backends legacy,torch,onnx --device cuda
--providers tensorrt,cuda,cpu` times every path and checks each one
against the legacy outputs (`max_dp`, `max_dv`); then `rpc_server.py
--device cuda` (optionally `--backend onnx --providers tensorrt,cuda,cpu
--fp16`) with `NNBatch` 32-64 and `NNPipeline` on. With a fast network
the server's graph construction (about 0.4 ms a position in Python, one
core) becomes the limit, near 2,500 positions a second, unless the engine
sends the graphs (`NNGraphs`, below).

### Graphs built by the engine (`NNGraphs`)

`src/RelGraph.cpp` builds the relational-v1 graph in C++: the same float32
features, the same edges in the same order (the reference's insertion
order, then the same stable sort by destination) and the legal moves' v3
policy indices. It is bit-identical to `relational_graph.build_graph` for
the position's FEN (`test_relgraph.py`: about 3,000 random positions and
the special positions of `test_relational_fast.py`, through
`relgraph_dump`), and takes about 30 µs a position with the encoding.

With `NNGraphs` on (default off), each request carries the positions'
graphs (`"graphs"`, base64; layout in `RelGraph.h`), and the server decodes
them (`relational_fast.decode_graph`, 0.06 ms) instead of building them
(0.39 ms). The replies are byte-identical either way. Only a server that
would build the same graph uses them (a v3 network on the gnn_infer
backends, not trained with histories) and says so in that reply (`"graphs":
true`). Any other server, older ones included, ignores the field, and the
engine stops sending graphs on that connection after the first reply. A
request grows from about 0.3 KB to 5.8 KB a position, which is fine on one
machine, less so over a slow network.

Per position at batch 64, one core (CPU, full-size v3 network): building
0.39 ms vs decoding 0.06 ms plus 0.01 ms more JSON, against 4.2 ms for the
network. So on the CPU it saves about 8%. On a GPU, where the network takes
a fraction of that, graph building was the limit, and this removes most of
it. That case has not been measured here.

### Hybrid search options

Ways for the two arms to share work and for MCTS to spend its effort
better. All of them are off (or at the old value) by default: they change
the hybrid's play, and none is admitted until a trained network shows a
gain with it (docs/TESTING.md, "The NAGS hybrid"). The summary line
reports each one (`mcts_reused`, `tt_values`, `smart_stop`, `root_order`).

| Option | Default | What it does |
|---|---|---|
| `MctsReuse` | off | Keeps the tree after the move. The next search starts from the node of the new root's position (the root itself, a child, or our move plus the reply), with its visits and expanded subtree; the rest is freed. `ucinewgame` drops it. |
| `MctsCpuctBase` | 0 (constant) | With b > 0 the PUCT constant grows with the parent's visits N as c + ln((1 + N + b) / b) (AlphaZero), so a long search explores more. |
| `MctsFpu` | 20 | First-play urgency: an unvisited child starts this many hundredths below its parent's value. |
| `MctsPolicyTemp` | 100 | Priors become p^(1/T), renormalised, when a node is expanded (T = value/100): above 100 flatter, below sharper. |
| `MctsSmartStop` | off | MCTS stops once the most visited root move leads the second by more than the simulations still possible (the rest of the budget, and the current rate until the alpha-beta arm's hard limit), so its final choice cannot change, and whether that move is proposed to alpha-beta (64 visits and half the root's) is settled as well; the core is freed for the rest of the move. |
| `MctsAbShare` | 85 | The alpha-beta arm's share of the move's time when MCTS runs (the rest is kept for verifying a proposal). |
| `MctsTTValue` | 0 (off) | A leaf whose position has an exact score in the shared transposition table from a search at least this deep takes that score as its value instead of the network's: P(win) − P(loss) under the logistic WDL model of the distillation targets (a = 400, b = 250), mates ±1. The alpha-beta arm has already searched most positions near the root, so MCTS stops re-guessing them. |
| `NNRootOrder` | off | The alpha-beta search tries the root moves in the order of the network's prior (one evaluation per move, usually from the cache; also when MCTS does not run, from a hard limit of 50 ms; connecting, like a request, may take at most a tenth of the hard limit). Root moves late in the order are reduced (LMR), so this is neural guidance of the alpha-beta search itself. |

### Game histories

The engine sends FENs, so relational-v1's `repetitions` feature has always
been 0 for the network, in training (label rows are FENs) and in play. The
protocol can now carry each position's history (the moves since the last
capture or pawn move and the position before them; `rpc_server.py` turns
them into a board with its move stack), but only for a network trained
with them (checkpoint flag `history`, `distill_params.use_history`), and the
engine sends them only when the server's replies ask for them.

This alone gives the MCTS arm little: a position that repeats an earlier one
is a terminal draw in the tree and is never sent to the network, so at the
leaves the feature is still 0; only a root that repeats the game differs.
What the policy would gain from is knowing which legal moves repeat a
position (draw now, or avoid it), a relation per legal move, which is a
new graph version (`relational-v2`) with training rows that carry their
histories (`nags_datagen` writes FENs without moves). The plumbing above
is what such a network needs to be served and trained; none is trained.

**TensorRT.** Through ONNX Runtime's TensorRT execution provider
(`--providers tensorrt,cuda,cpu`). The exported graph (opset 18) is Gemm,
MatMul, Gather / GatherND, LayerNormalization, Softmax, elementwise
operations and seven ScatterND nodes with `reduction="add"` (the
`index_add` message aggregation of each layer and the mean pooling).
Whether a given TensorRT version's ONNX parser takes ScatterND with a
reduction has not been checked here; nodes it refuses are run by the CUDA
provider, with the rest of the graph in TensorRT. Dynamic shapes come with
an optimisation profile from 1 to 256 positions, engines are cached under
`--onnx-cache`, and `--fp16` turns on half precision. Not run here: whether
TensorRT beats the CUDA provider for this graph, and how far `--fp16` moves
the outputs, are for `gnn_bench.py` on the GPU (the `max_dp` / `max_dv`
columns).

### What the v3 network and distillation buy (move priors and values)

`tools/gnn_compare.py` trains the networks in several ways on the same positions and
compares them on held-out games. One run (2026-10-17): 300 `nags_enhanced`
self-play games at 2000 nodes per move, 4000 positions labelled with
MultiPV 4 at 15000 nodes, 3544 for training and 456 held out (30 whole
games), 8 epochs each, default model sizes, CPU:

| Arm | Top-1 = engine's best | Top-1 = move played | NLL of engine's best | Value MSE vs search | Value MSE vs game result |
|-----|----:|----:|----:|----:|----:|
| A: v2, played moves (legacy `supervised`) | 10.1% | 12.5% | 3.29 | 0.231 | 0.574 |
| A2: A with `supervised_legal_mask` | 11.8% | 11.6% | 2.92 | 0.313 | 0.660 |
| B: v3, distilled from the search | 22.8% | 23.9% | 2.50 | 0.058 | 0.379 |
| C: v3, played moves and results | 22.8% | 25.7% | 2.47 | 0.090 | 0.434 |
| B with `rest_mass_scale` 0.1 | 23.0% | 25.9% | 2.46 | 0.060 | 0.388 |
| B with `rest_mass_scale` 0.02 (the default since) | 24.8% | 25.4% | 2.45 | 0.058 | 0.387 |
| B with `rest_mass_scale` 0 | 24.1% | 24.6% | 2.45 | 0.070 | 0.413 |
| uniform prior / constant 0 value | 8.4% | – | 2.94 | – | 0.658 |

* The v3 network is the large gain for the move prior: about twice the
  top-1 agreement of v2 and a much lower NLL with the same data and
  epochs; masking illegal moves alone helps v2 little.
* For the policy, distillation with the first default (`rest_mass_scale`
  0.5, B) did not beat training on the moves played (B and C equal; the
  moves played came from the same engine at 2000 nodes, so they are
  already search moves). That default gave the unscored moves' bucket 39%
  of the target on average, more than the engine's best move (37%);
  smaller weights improved the held-out policy steadily (NLL 2.50, 2.46,
  2.45 at 0.5, 0.1, 0.02), and 0.02 is ahead of training on played moves
  on both policy and value. With no bucket at all (0) the value got worse.
  The policy margins are small against the noise of 456 positions.
* For the value, distillation is better even judged by game results
  (0.379 against 0.434), and both v2 value heads ended no better than
  predicting 0.
* Small and correlated samples (456 positions from 30 games: about ±2-4%
  on top-1), self-play positions only, no playing strength measured: these
  say which training to use for a first real v3 network, not that it will
  help the hybrid play better.

### Why PPO on these games is not on-policy

The `ppo` step updates the network on self-play positions with the move
the engine played. PPO's ratio π_new(a|s) / π_old(a|s) is an importance
weight only if π_old chose a. In NAGS the move is chosen by the alpha-beta
search (MCTS, guided by the network, can only replace it after a
verification search), so the behaviour policy is the search, not the network:
the ratio says nothing about how the data were sampled, clipping does not
bound a real policy change, and the update amounts to advantage-weighted
imitation of the engine's moves. The step is kept as it was, since a network
may be training with it, and logs this caveat. For v3, distillation is the
recommended objective: it trains on what the search actually knows, the
exact scores of its best k moves, instead of one sampled move and a game
result.

## Meta-learner (`meta_learner.py`)

A small MLP maps 8 position/clock features to the three deltas. Training is
advantage-weighted regression on (features, deltas used, reward) samples.
Samples whose reward beats the batch average get exponentially more weight,
so the predictions move towards deltas that won games. The training data is
stored as JSON.

**Why it is off by default.** The reward is not an informative signal for
the decision it is meant to train:

* Every move of a game gets the same reward (the result). Samples from one
  game are not independent, so the effective sample size is the number of
  games, not of moves.
* One move's deltas change the expected game result by a tiny amount ε,
  while a game result varies by about ±0.5 (variance about 0.25 with
  draws). Telling ε = 0.01 from 0 takes on the order of (0.5 / 0.01)² ≈
  2500 games per region of the feature space, before the noise exploration
  adds of its own.
* The result depends on all deltas of both engines over the whole game,
  and two of the three deltas only matter when the MCTS arm runs (it does
  not at short time controls, and without a trained network).
* Advantages are normalised per batch across unrelated positions, so the
  regression mostly fits noise.

So `UseMetaLearner` defaults to false, and the learner should only be
switched on with a match result that shows a gain. A signal that measures
the decision directly is the way forward: for sampled positions, search
with one budget or setting and with another, and score the move each
chooses with a much deeper reference search (the value lost, and how
often the extra effort changed the move), per unit of time spent. That
gives a per-move reward with no game-length credit assignment, and it is
also what time management needs (when is more search unlikely to change
the decision?).

## Training pipeline (`training_pipeline.py`)

See the README for the steps. Notes:

* Supervised training splits train/validation by game and reports
  validation policy loss, value loss and top-1 move accuracy. It keeps the
  checkpoint with the best validation loss.
* Self-play uses random opening plies for variety. Games are adjudicated as
  draws after `max_game_plies`; threefold repetition and the fifty-move rule
  are claimed.
* PPO is a clipped surrogate over the engine's own moves, with the policy
  restricted to legal moves:
  * old log-probabilities and values come from the starting checkpoint;
  * advantage = result − old value, normalised;
  * losses: value loss (MSE against the result) plus an entropy bonus.

  The moves come from the alpha-beta search, not from the network, so this
  is not an on-policy update (see "Why PPO on these games is not on-policy"
  above); `distill` is the recommended objective for the v3 network.
* `supervised_legal_mask` (default off, so the step is unchanged) restricts
  the supervised policy loss to the legal moves.
* Evaluation plays each opening (from the polyglot book if present, otherwise
  random plies) with both colours under the configured clock, and reports
  `+W =D -L`, the score, and Elo with a 95% interval. Promotion requires Elo
  above `elo_threshold`. With few games the interval is wide, so use enough
  games before trusting a promotion.
