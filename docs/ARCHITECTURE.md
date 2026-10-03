# NAGS architecture

This document describes how the engine and the learning components work. It
replaces the earlier per-feature reports (`*_REPORT.md`, `README_BASIC.md`),
which described code that has since been rewritten.

## Source layout

```
src/
  ChessTypes.h      Piece / Color / Move types shared by everything (a1 = 0 ... h8 = 63)
  BitOps.h          portable lsb / msb / popcount (MSVC, GCC, Clang, fallback)
  Board.h/.cpp      bitboard board with ray-based sliding attacks   (nags, nags_basic)
  FastBoard.h/.cpp  bitboard board with magic bitboards + mailbox   (nags_fast, nags_enhanced)
  Eval.h            static evaluation (material, piece-square tables, bishop pair)
  TT.h/.cpp         lockless transposition table shared by search threads
  Search.h          iterative-deepening alpha-beta (template over the board type)
  Uci.h/.cpp        UCI front end: worker-thread search, go parsing, time management
  ClassicEngine.h   UCI engine around Search (nags_basic / nags_fast / nags_enhanced)
  NAGS.h/.cpp       hybrid controller: bandit over alpha-beta and MCTS, evaluators
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
* Check extension, late-move reductions for quiet moves, mate distance pruning.
* Move ordering: transposition-table move, then captures by MVV/LVA (with
  promotions), killer moves, history heuristic.
* Quiescence search over captures and promotions with delta pruning. When in
  check, all evasions are searched and there is no stand-pat.
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

Per move:

1. **Meta-learner.** With `UseMetaLearner`, the controller sends the FEN, the
   remaining clock time, the previous root uncertainty and the tactical ratio
   (the share of legal moves that capture, promote or give check). It gets
   back three deltas in [-1, 1]:
   * DFS depth cap: 10 + 4·delta, unless `go depth` fixes it;
   * MCTS simulation budget: 2000 + 1000·delta;
   * PUCT exploration constant: 1.4 + 0.5·delta.

   In self-play, `MetaExploration` adds Gaussian noise to the deltas so the
   meta-learner sees varied choices.
2. **Depth 1.** A depth-1 alpha-beta search always completes first, so there
   is always a reasonable move.
3. **Bandit loop.** A Thompson-sampling bandit (Beta posteriors with mild
   forgetting; statistics persist across moves within a game) picks an arm
   for each pull:
   * **DFS:** one more iteration of the shared alpha-beta worker.
   * **MCTS:** a batch of PUCT simulations, capped by both a simulation count
     and a wall-clock slice. Leaves are evaluated by the GNN via
     `rpc_server.py` (one request per leaf: FEN plus legal moves in, value
     plus priors for those moves out) or by the heuristic evaluator. Each node
     stores its value from the point of view of the player who moved into it,
     so selection maximises for the side to move. Checkmate, stalemate and
     draws are terminal nodes. The tree is capped at about 2M nodes.

   A pull counts as a success if it changed that arm's best move or moved its
   value estimate noticeably (30 cp for DFS, 0.05 for the MCTS Q-value). So
   time flows to the arm that is still discovering something about the
   position.
4. **Decision.** The engine plays the deepest completed alpha-beta move,
   unless all of these hold:
   * MCTS was guided by the network;
   * at least half the root visits (and at least 64) went to a different move;
   * a verification search scores that move within 25 cp of the alpha-beta
     best.
5. **Reporting.** Standard `info` lines for each completed DFS iteration, an
   `info string nags ...` summary, and an `info string nags_meta ...` line
   with the meta-learner inputs and the deltas used. The training pipeline
   reads the `nags_meta` line to build meta-learner samples.

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
