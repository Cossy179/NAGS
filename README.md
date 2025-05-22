# NAGS - Neuro-Adaptive Graph Search Chess Engine

NAGS is a state-of-the-art chess engine combining deep neural networks, graph-based board representation, and adaptive search techniques to achieve superhuman chess performance.

## Features

- Hypergraph + GNN-based board representation
- Dual-Head Transformer-GNN position evaluator
- Hybrid search combining α-β and MCTS approaches
- Meta-learning for position-specific search optimization
- Dual-phase training: supervised learning from GM games followed by RL self-play
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
   mkdir build && cd build
   cmake ..
   make
   ```
   
   For CUDA support, use:
   ```
   mkdir build && cd build
   cmake -DENABLE_CUDA=ON ..
   make
   ```

## Training the Engine

NAGS features a dual-phase training pipeline that combines supervised learning from GM games followed by self-play reinforcement learning.

### Using the Integrated Training Pipeline

The integrated C++ training pipeline can handle both supervised and reinforcement learning phases:

```
./bin/nags_train --pgn AJ-CORR-PGN-000.pgn --model-dir ./models
```

### Training Options

- Run only supervised learning phase:
  ```
  ./bin/nags_train --pgn AJ-CORR-PGN-000.pgn --supervised-only
  ```

- Run only reinforcement learning phase (requires a pre-trained model):
  ```
  ./bin/nags_train --checkpoint ./models/supervised_model.bin --rl-only
  ```

- Customize training parameters:
  ```
  ./bin/nags_train --pgn AJ-CORR-PGN-000.pgn --sl-batch 1024 --sl-epochs 20 --rl-games 2000 --rl-iter 30
  ```

- For help with all options:
  ```
  ./bin/nags_train --help
  ```

### Alternative Python Training Scripts

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
  - `supervised_trainer.h`: Supervised learning from GM games
  - `rl_trainer.h`: Self-play reinforcement learning
  - `dual_training_pipeline.h`: Combined training pipeline
- `scripts/`: Utility scripts for training and evaluation
- `tests/`: Test suite
- `utils/`: Helper utilities

## License

MIT 