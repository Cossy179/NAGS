You are being tasked with **designing and implementing NAGS**, the Neuro-Adaptive Graph Search chess engine. NAGS must be the strongest engine ever, running as a UCI-compatible executable under the Arena GUI.

> **Prerequisite:** the file `AJ-CORR-PGN-000.pgn` is available locally for all supervised-training stages.

---

### 1. Project Goals and Overview

1. Achieve Elo ≥ 3600 by combining tactical speed with deep positional understanding.
2. Integrate supervised learning from GM correspondence games (`AJ-CORR-PGN-000.pgn`) and self-play RL.
3. Adapt search strategy per position using online meta-learning.

---

### 2. Board Encoding: Hypergraph + GNN

* **Hypergraph Construction**

  * Nodes: 64 squares + piece nodes + metadata nodes (castling rights, en passant).
  * Edges: occupancy, attack/defense interactions, pawn-chain adjacency, king-safety zones.
* **GNN Backbone**

  * PyTorch Geometric with 6–8 message-passing layers, residual & batch-norm.
  * Outputs a **global graph embedding** + **per-node embeddings**.

---

### 3. Dual-Head Transformer-GNN Evaluator

* **Policy Head**

  * 4-layer Transformer over node embeddings → move priors (softmax over legal moves).
* **Value Head**

  * Transformer + MLP → scalar centipawn evaluation + Monte-Carlo Dropout for uncertainty.
* **Warm-Start**

  * Initialize weights via supervised pre-training on `AJ-CORR-PGN-000.pgn`.

---

### 4. Hybrid Search Controller

* **Bayesian Bandit Scheduler**

  * Arms = {DFS probe, MCTS playout}; reward = value-uncertainty reduction + tactical-shot detection.
* **Tactical α–β DFS**

  * Depth 4–6, quiescence, killer moves, history heuristics in optimized C++.
* **Policy-Guided MCTS**

  * PUCT selection w/ policy priors; expansion/eval by dual-head net; backup values & visits.

---

### 5. Online Meta-Learner

* **Purpose:** continuously tune search hyperparameters per position.
* **Inputs:**

  * Position embedding, time remaining, search-variance, tactical-volatility metrics.
  * **Initial data source:** embeddings & policy/value targets derived from `AJ-CORR-PGN-000.pgn`.
* **Architecture:** small 2-layer MLP that outputs

  * DFS depth cap
  * MCTS playout budget ratio
  * Bandit exploration constant
* **Training:** log top-move improvements and Elo deltas during self-play; optimize MLP for max Elo/sec gains.

---

### 6. Data Pipeline & Curriculum

* **Source:** `AJ-CORR-PGN-000.pgn` (1.6 M correspondence games with GM tags).
* **Filtering:** PGN headers where ≥ 1 player = “GM” and FIDE rating ≥ 2500.
* **Parsing:** convert to `(FEN_before_move, move_index, outcome)` triples.
* **Curriculum:**

  1. Pawn/minor-piece endgames
  2. Blitz & rapid self-play
  3. Full classical self-play with increasing time controls

---

### 7. Training Regimen

1. **Supervised Warm-Start**

   * Loss = CE(policy) + MSE(value), AdamW 1e-4, batch 512.
2. **Self-Play RL**

   * PPO with MCTS-derived target policies; blend in RAVE for cold-start.
3. **Continuous Retraining**

   * Nightly self-play → auto-retrain if Elo improves.

---

### 8. Integration & Deployment

* **UCI Stub (C++)**: parse `uci`, `isready`, `position`, `go`, `stop`, `quit`; call search controller.
* **RPC Inference**: batch board graphs → GNN server (gRPC/ZeroMQ) to maximize GPU throughput.
* **Build**: CMake for core + RPC client; static-link for Windows/Linux.
* **Arena Setup**: install `nags_v1.exe` as UCI engine; configure hash, threads, RPC endpoint.
* **Readme.md file:** helps users from start to finish on how they can make the engine and configure it to work on Arena

---

### 9. Benchmarking & Validation

* **Speed:** DFS nodes/sec; GNN positions/sec on target GPU.
* **Elo Matches:** round-robins vs. Stockfish 17.1 & Lc0 2023 builds.
* **Novelty:** measure top-move novelty vs. large OTB databases.

---

### 10. Optimization &  Extensions

* Parallel MCTS (YBWC/PVS)
* Quantum-inspired playout sampling
* Deeper Transformer-GNN variants vs. latency tradeoffs
* Adversarial training for robustness
* Attention-based explainability module

---

**Deliverable:** an end-to-end NAGS v1 prompt that, when executed by a capable AI or developer team, yields a UCI-compatible engine in Arena that smashes Stockfish & Leela.
