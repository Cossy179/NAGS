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
* Extensions: check extension; singular extension of the TT move (depth ≥ 8,
  verified by a reduced search without it, with a multi-cut when even the
  alternatives beat beta).
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
  history pruning and reductions.
* Quiescence search over captures and promotions with delta pruning and
  without losing captures (SEE < 0). When in check, all evasions are
  searched and there is no stand-pat. Results go to the transposition table
  at depth 0; non-PV nodes take cutoffs from it, its move is tried first,
  and a stored score bounded on the right side replaces the stand-pat.
* Time management: no new iteration after a soft limit that grows when the
  best move just changed or the score dropped and shrinks when the best move
  is stable; a hard limit aborts the search.
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
target. The hard limit (abort) is the smaller of 3x the target and half the
remaining time (90% of the remaining time when `movestogo` is 1). `movetime`
uses the given time minus the overhead. `go infinite` (or a bare `go`)
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
   evaluated by the GNN (one request per leaf: FEN plus legal moves in,
   value plus priors for those moves out; the heuristic evaluator takes over
   if the service fails mid-search). Each node stores its value from the
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

## Meta-learner (`meta_learner.py`)

A small MLP maps 8 position/clock features to the three deltas. Training is
advantage-weighted regression on (features, deltas used, reward) samples.
Samples whose reward beats the batch average get exponentially more weight,
so the predictions move towards deltas that won games. The training data is
stored as JSON.

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
* Evaluation plays each opening (from the polyglot book if present, otherwise
  random plies) with both colours under the configured clock, and reports
  `+W =D -L`, the score, and Elo with a 95% interval. Promotion requires Elo
  above `elo_threshold`. With few games the interval is wide, so use enough
  games before trusting a promotion.
