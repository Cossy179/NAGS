# NAGS - Neuro-Adaptive Graph Search

A chess engine project that combines classic alpha-beta search with a graph
neural network (GNN) evaluator, Monte-Carlo tree search and a meta-learner
that tunes search hyperparameters per move.

The repository contains four UCI engines, two Python services and a training
pipeline:

| Component | What it is |
|-----------|------------|
| `nags` | Hybrid engine: alpha-beta and MCTS share the time via a Thompson-sampling bandit; MCTS uses the GNN (via `rpc_server.py`) when available; `meta_learner.py` adjusts search hyperparameters per move. |
| `nags_enhanced` | Alpha-beta on magic bitboards with a shared transposition table and Lazy SMP threads. The strongest pure alpha-beta build. |
| `nags_fast` | Same search on magic bitboards, without a transposition table. |
| `nags_basic` | Same search on the simpler ray-based board, without a transposition table. |
| `rpc_client` | Smoke test for the GNN service. |
| `rpc_server.py` | Serves GNN policy priors / values / uncertainty over TCP. |
| `meta_learner.py` | Serves (and learns) per-move hyperparameter adjustments. |
| `training_pipeline.py` | PGN parsing, supervised training, self-play, PPO, match-based evaluation and model promotion. |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces work and
fit together.

## Requirements

- C++17 compiler (GCC, Clang or MSVC) and CMake 3.15+
- Python 3.9+ for the services and training (`pip install -r requirements.txt`)
- Optional: a CUDA GPU for faster training

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Executables end up in `build/` with single-configuration generators (Linux,
macOS: `build/nags`) and in `build/Release/` with Visual Studio
(`build\Release\nags.exe`).

## Test

```bash
(cd build && ctest -C Release --output-on-failure)                  # C++: perft, hashing, search, UCI end-to-end
python -m pytest -q                                                  # Python: graph, network, services, pipeline
```

Before merging a change that affects playing strength, follow
[docs/TESTING.md](docs/TESTING.md): check the `bench` fingerprint
(`build/nags_enhanced bench`), then prove the change in an SPRT match with
`tools/sprt.py`. `tools/calibrate.py` estimates an absolute rating against
Stockfish's strength-limited mode.

The C++ tests check move generation against reference perft counts, that the
incremental Zobrist hash always matches a freshly computed one, search results
on known positions (mates, perpetual check, stalemate), the time manager, and
every engine's UCI behaviour (side to move, bad input, `movetime`, threads,
MultiPV, pondering), and tablebase probing.
CI runs both suites on Linux, Windows and macOS
(`.github/workflows/nags-ci.yml`).

## Using the engines

Any UCI GUI (Arena, Cute Chess, BanksiaGUI, ...) can load the executables.
From a terminal:

```bash
printf 'uci\nposition startpos moves e2e4\ngo depth 8\n' | build/nags_enhanced
```

The search runs on its own thread, so `stop`, `isready` and `quit` are handled
while it thinks. Supported `go` parameters: `wtime btime winc binc movestogo
movetime depth nodes mate infinite ponder`, plus the non-standard `go perft N`
(also `perft N`) and `d` (print the FEN). The alpha-beta engines support
pondering (`go ponder`, then `ponderhit` or `stop`); `nags` treats
`go ponder` as a normal search.

UCI options:

| Option | Engines | Meaning |
|--------|---------|---------|
| `Hash` (MB, default 64) | `nags`, `nags_enhanced` | Transposition table size |
| `Threads` (default 1) | `nags_enhanced` | Lazy SMP search threads |
| `Clear Hash` | `nags`, `nags_enhanced` | Empty the transposition table |
| `Move Overhead` (ms, default 50) | all | Time kept in reserve per move for GUI/network lag |
| `MultiPV` (default 1) | `nags_basic`, `nags_fast`, `nags_enhanced` | Number of best lines to report |
| `Ponder` | `nags_basic`, `nags_fast`, `nags_enhanced` | Lets the GUI know it may ponder |
| `SyzygyPath` | all | Directories with Syzygy tablebase files (`:`-separated, `;` on Windows) |
| `SyzygyProbeLimit` (default 7) | all | Only probe positions with at most this many pieces |
| `UseNN`, `NNHost`, `NNPort` | `nags` | Use `rpc_server.py` for MCTS priors/values (default `127.0.0.1:5555`) |
| `UseMetaLearner`, `MetaHost`, `MetaPort` | `nags` | Ask `meta_learner.py` for per-move deltas (default `127.0.0.1:5556`) |
| `MetaExploration` (0-100) | `nags` | Gaussian noise (std = value/100) added to the deltas; used in self-play |

**Endgame tablebases.** With `SyzygyPath` set, the search probes the
win/draw/loss tables after captures and pawn moves, and with the DTZ tables
present a tablebase position at the root is played straight from the tables
(the move that keeps the result under the fifty-move rule and makes
progress). `nags` probes inside its alpha-beta arm only. Probing uses
[Fathom](https://github.com/jdart1/Fathom) (MIT, `third_party/fathom`). The
3-4-5 piece tables are about 1 GB, for example from
<http://tablebase.sesse.net/syzygy/>; the tests use the 3-piece tables in
`tests/data/syzygy`.

`nags` works without the Python services: if they are not running it falls
back (after a 150 ms connection attempt, retried at most once a minute) to a
deterministic heuristic evaluator and default hyperparameters. MCTS is only
allowed to override the alpha-beta move when the trained network is guiding
it.

## Python services

```bash
python rpc_server.py --model models/production_model.pth   # GNN service on 127.0.0.1:5555
python meta_learner.py                                      # meta-learner on 127.0.0.1:5556
build/rpc_client                                            # checks the GNN service end to end
```

Both speak newline-delimited JSON over TCP and keep connections open across
requests (protocols are documented at the top of each file). Without
`--model`, `rpc_server.py` serves an untrained network and says so in its log.
`python meta_learner.py --selftest` checks that the meta-learner's training
works on synthetic data.

## Training

The training data, `AJ-CORR-PGN-000.pgn` (about 1 GB), is stored with Git LFS.
Fetch it first with `git lfs pull`.

```bash
./run_training.sh full                 # Windows: run_training.bat full
./run_training.sh selfplay --games 20  # individual steps; extra options go to training_pipeline.py
python training_pipeline.py --help
```

| Step | What it does |
|------|--------------|
| `parse` | Extracts (position, move played, game result) samples from the PGN. |
| `supervised` | Trains the GNN policy (cross-entropy on the played move) and value head (MSE on the result, in [-1, 1] from the side to move's view), with a by-game validation split. |
| `selfplay` | Plays real `nags` vs `nags` games with the current network (`rpc_server.py` and `meta_learner.py` are started automatically) and records positions, moves, results and meta-learner samples. |
| `ppo` | Clipped PPO update of the network on the self-play games (advantage = result - old value estimate, legal-move-masked policy). |
| `evaluate` | Plays a match (paired openings, both colours) against the baseline, estimates Elo with a 95% interval, and promotes the model to `models/production_model.pth` if it beats `elo_threshold`. |
| `meta` | Trains the meta-learner on the self-play samples (reward = game result). |

`training_config.json` holds every setting (directories, engine path, batch
size, epochs, self-play games and time per move, `model_params`, `ppo_params`,
evaluation games, time control, opening book, `baseline_engine`, Slack
webhook). `baseline_engine` is `heuristic` (`nags` without the network, which
measures whether the network helps), `production` (the current production
model) or the command of any UCI engine, e.g. `stockfish`.

## License

MIT, see [LICENSE](LICENSE).
