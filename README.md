# NAGS - Neuro-Adaptive Graph Search Chess Engine

NAGS is a state-of-the-art chess engine combining deep neural networks, graph-based board representation, and adaptive search techniques to achieve superhuman chess performance.

## Features

- Hypergraph + GNN-based board representation
- Dual-Head Transformer-GNN position evaluator
- Hybrid search combining α-β and MCTS approaches
- Meta-learning for position-specific search optimization
- UCI compatibility for Arena chess GUI integration

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- C++17 compatible compiler
- CUDA 11.7+ (for GPU acceleration)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Cossy179/NAGS.git
   cd NAGS
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Build the C++ components:
   ```
   cd src/core
   mkdir build && cd build
   cmake ..
   make
   ```

## Training the Engine

1. Supervised learning from GM games:
   ```
   python scripts/train_supervised.py --pgn_path AJ-CORR-PGN-000.pgn
   ```

2. Self-play reinforcement learning:
   ```
   python scripts/train_rl.py --initial_model models/supervised_model.pt
   ```

## Using NAGS with Arena

1. Configure the UCI executable in Arena:
   - Open Arena Chess GUI
   - Select "Engines" > "Install New Engine"
   - Navigate to `bin/nags_v1.exe` and select it
   - Configure hash size, threads, and other parameters as needed

2. Start playing or analyzing with NAGS as your engine

## Project Structure

- `src/core/`: Core C++ engine components (board representation, search)
- `src/gnn/`: Graph Neural Network implementation
- `src/mcts/`: Monte Carlo Tree Search implementation
- `src/uci/`: Universal Chess Interface protocol support
- `src/meta/`: Meta-learning components for search adaptation
- `src/training/`: Training pipeline implementations
- `scripts/`: Utility scripts for training and evaluation
- `tests/`: Test suite
- `utils/`: Helper utilities

## License

MIT 