# NAGS - Neuro-Adaptive Graph Search

A hybrid chess engine combining traditional alpha-beta search with graph neural networks and adaptive meta-learning.

## 🏗️ Architecture

- **C++ UCI Engine**: Bitboard-based move generation with transposition tables
- **Hybrid Search**: Bayesian bandit selection between DFS (alpha-beta) and MCTS
- **Graph Neural Network**: PyTorch Geometric-based position evaluation
- **Meta-Learning**: Online hyperparameter adaptation based on position characteristics
- **Training Pipeline**: Automated supervised + reinforcement learning with model promotion

## 🚀 Quick Start

### Prerequisites

- **C++**: CMake 3.15+, C++17 compiler (MSVC/GCC/Clang)
- **Python**: 3.9+ with pip
- **Optional**: CUDA-capable GPU for training acceleration

### Installation

1. **Clone and build C++ engine:**
   ```bash
   git clone <repo-url>
   cd NAGS
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test basic functionality:**
   ```bash
   # Test UCI engine
   echo -e "uci\nisready\nposition startpos\ngo depth 3\nquit" | build/Release/nags.exe
   
   # Test Python components
   python -c "from chess_graph import test_graph_dims_and_forward; test_graph_dims_and_forward()"
   ```

## 🎯 Usage

### As UCI Engine

```bash
# Basic usage
build/Release/nags.exe

# UCI commands
uci                           # Engine identification
setoption name Hash value 256 # Set hash table size
setoption name Threads value 4 # Set thread count
position startpos moves e2e4   # Set position
go wtime 60000 btime 60000    # Search with time control
go depth 6                    # Search to fixed depth
```

### Training Pipeline

```bash
# Full training pipeline
./run_training.sh full

# Individual steps
./run_training.sh parse      # Parse PGN to training data
./run_training.sh supervised # Supervised pre-training
./run_training.sh selfplay   # Self-play data collection
./run_training.sh ppo        # PPO reinforcement learning
./run_training.sh evaluate   # Evaluate against baseline

# Windows
run_training.bat full
```

### RPC Services

```bash
# Start neural network evaluator
python rpc_server.py

# Start meta-learner service  
python meta_learner.py

# Test RPC client
build/Release/rpc_client.exe
```

## 📊 Components

### Core Engine (`src/`)

- **Board.h/cpp**: Bitboard representation with Zobrist hashing
- **Search.h/cpp**: Alpha-beta search with quiescence and transposition tables
- **TT.h/cpp**: Transposition table for position caching
- **NAGS.h/cpp**: Hybrid search controller with bandit selection
- **MetaClient.h/cpp**: RPC client for meta-learning queries

### Neural Networks

- **chess_graph.py**: Graph representation of chess positions
- **gnn_evaluator.py**: Dual-head transformer (policy + value)
- **meta_learner.py**: Online hyperparameter adaptation
- **rpc_server.py**: Neural network inference service

### Training Infrastructure

- **training_pipeline.py**: End-to-end training orchestration
- **run_training.sh/.bat**: Cross-platform training scripts
- **.github/workflows/nags-ci.yml**: Continuous integration

## 🧠 How It Works

1. **Position Encoding**: Chess positions → graph with nodes (squares, pieces, metadata) and edges (attacks, occupancy, pawn chains)

2. **Neural Evaluation**: 6-layer GNN → dual-head transformer → policy probabilities + position value + uncertainty

3. **Hybrid Search**: Bayesian bandit chooses between:
   - **DFS**: Traditional alpha-beta with quiescence
   - **MCTS**: Policy-guided tree search with neural priors

4. **Meta-Learning**: Adapts search hyperparameters (depth, budget, exploration) based on position features and performance history

5. **Training Loop**: 
   - Supervised pre-training on master games
   - Self-play reinforcement learning with PPO
   - Continuous evaluation and model promotion

## 📈 Performance

- **Move Generation**: ~2M+ nodes/second (bitboards + magic attacks)
- **Search**: Adaptive depth/time allocation via meta-learning
- **Neural Inference**: Batched evaluation with uncertainty estimation
- **Training**: Automated pipeline with Elo-based model promotion

## 🔧 Configuration

Edit `training_config.json`:

```json
{
  "max_positions": 100000,
  "batch_size": 32,
  "learning_rate": 0.001,
  "epochs": 10,
  "self_play_games": 100,
  "elo_threshold": 25,
  "model_params": {
    "hidden_dim": 128,
    "gnn_layers": 6
  }
}
```

## 🧪 Testing

```bash
# C++ tests
cd build && ctest

# Python tests  
pytest -v

# Integration test
python -c "from gnn_evaluator import warm_start_and_run_example; print(warm_start_and_run_example())"

# RPC test
python rpc_server.py &
build/Release/rpc_client.exe
```

## 🚀 Deployment

### Arena/GUI Integration

1. Add `build/Release/nags.exe` as UCI engine
2. Configure hash size and threads in engine settings
3. Use time controls or fixed depth for analysis

### Production Training

1. Set up nightly CI runs with `nags-ci.yml`
2. Monitor training logs in `logs/`
3. Models auto-promote when Elo threshold is exceeded
4. Notifications sent on successful promotions

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open pull request

## 📞 Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: team@company.com
