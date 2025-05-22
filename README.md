# NAGS - Neuro-Adaptive Graph Search Chess Engine

NAGS is a novel chess engine that marries a hypergraph-based board representation with advanced neural architectures and adaptive search strategies. Below are its core components with equations rendered as images for GitHub compatibility.

## Features
- Hypergraph + GNN-based board representation
- Dual-Head Transformer-GNN position evaluator
- Hybrid search combining α-β pruning and MCTS with UCT selection
- Meta-learning for position-specific search optimization
- Dual-phase training: supervised learning from GM games followed by RL self-play
- UCI compatibility for Arena chess GUI integration

## System Architecture

### 1. Hypergraph Neural Network

This module processes a hypergraph representation of the board. The key update equations are:

![Equation](https://latex.codecogs.com/svg.image?%5Cmathbf%7BZ%7D%20%3D%20%5Csigma%5Cbigl%28%5Cmathbf%7BD%7D_e%5E%7B-1%7D%5Cmathbf%7BH%7D%5E%5Ctop%20%5Cmathbf%7BX%7D%5Cmathbf%7BW%7D_e%5Cbigr%29%2C)
![Equation](https://latex.codecogs.com/svg.image?%5Cquad%20%5Cmathbf%7BX%7D%27%20%3D%20%5Csigma%5Cbigl%28%5Cmathbf%7BD%7D_v%5E%7B-1%7D%5Cmathbf%7BH%7D%5Cmathbf%7BZ%7D%5Cmathbf%7BW%7D_v%5Cbigr%29.)

### 2. Message Passing Neural Network (MPNN)

The standard message-passing updates are:

![Equation](https://latex.codecogs.com/svg.image?%5Cmathbf%7Bm%7D_i%5E%7B%28t%2B1%29%7D%20%3D%20%5Csum_%7Bj%5Cin%20%5Cmathcal%7BN%7D%28i%29%7D%20M%28%5Cmathbf%7Bh%7D_i%5E%7B%28t%29%7D%2C%5Cmathbf%7Bh%7D_j%5E%7B%28t%29%7D%29%2C)
![Equation](https://latex.codecogs.com/svg.image?%5Cquad%20%5Cmathbf%7Bh%7D_i%5E%7B%28t%2B1%29%7D%20%3D%20U%28%5Cmathbf%7Bh%7D_i%5E%7B%28t%29%7D%2C%5Cmathbf%7Bm%7D_i%5E%7B%28t%2B1%29%7D%29.)

### 3. Multi-Head Self-Attention

The attention mechanism is given by:

![Equation](https://latex.codecogs.com/svg.image?%5Cmathrm%7BAttention%7D%28Q%2CK%2CV%29%20%3D%20%5Cmathrm%7Bsoftmax%7D%5CBigl%28%5Cfrac%7BQK%5E%5Ctop%7D%7B%5Csqrt%7Bd_k%7D%7D%5CBigr%29V)

### 4. UCT Selection (MCTS)

Child node selection uses the UCT rule:

![Equation](https://latex.codecogs.com/svg.image?i%5E%2A%20%3D%20%5Carg%5Cmax_i%20%5CBigl%28Q_i%20%2B%20c%20%5Csqrt%7B%5Cfrac%7B%5Cln%20N%7D%7Bn_i%7D%7D%5CBigr%29)

## Installation & Usage
1. Clone the repo:

   ```bash
   git clone https://github.com/Cossy179/NAGS.git
   cd NAGS
   ```

2. Install dependencies:

   ```bash
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

   ```bash
   python scripts/train_supervised.py --pgn_path AJ-CORR-PGN-000.pgn
   ```

5. **Reinforcement Learning**:

   ```bash
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