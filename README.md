# NAGS - Neuro-Adaptive Graph Search Chess Engine

NAGS is a state-of-the-art chess engine combining deep neural networks, graph-based board representation, and adaptive search techniques to achieve superhuman chess performance.

## Features

- Hypergraph + GNN-based board representation
- Dual-Head Transformer-GNN position evaluator
- Hybrid search combining α-β and MCTS approaches
- Meta-learning for position-specific search optimization
- Dual-phase training: supervised learning from GM games followed by RL self-play
- UCI compatibility for Arena chess GUI integration
- Advanced parallel search with dynamic work stealing
- Optimized SIMD-accelerated bitboard operations
- Position-aware dynamic search scheduling
- Ensemble neural network models with knowledge distillation
- Curriculum learning for progressive skill development

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- C++17 compatible compiler
- CUDA 11.7+ (for GPU acceleration)
- AVX2/BMI2 CPU instructions (for SIMD optimizations)

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
   
   For optimized AVX2/BMI2 support, use:
   ```
   mkdir build && cd build
   cmake -DENABLE_CUDA=ON -DENABLE_AVX2=ON -DENABLE_BMI2=ON ..
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

- Use curriculum learning for progressive training:
  ```
  ./bin/nags_train --pgn AJ-CORR-PGN-000.pgn --curriculum --difficulty-stages 4
  ```

- Train ensemble models:
  ```
  ./bin/nags_train --pgn AJ-CORR-PGN-000.pgn --ensemble --models 3
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

3. Curriculum learning:
   ```
   python scripts/curriculum_trainer.py --pgn_path AJ-CORR-PGN-000.pgn
   ```

4. Ensemble model training:
   ```
   python scripts/train_ensemble.py --pgn_path AJ-CORR-PGN-000.pgn --num_models 3
   ```

5. Knowledge distillation:
   ```
   python scripts/distill_model.py --teacher_models models/ensemble/*.pt --output models/distilled_model.pt
   ```

## Performance Optimizations

NAGS includes several performance optimizations for maximum strength:

1. **Parallel Search**: Dynamic multi-threaded search with work stealing for optimal CPU utilization.

2. **Bitboard Operations**: SIMD-accelerated bitboard operations for ultra-fast move generation:
   - AVX2 vector operations for parallel bit operations
   - BMI2/PEXT instructions for magic-less move generation
   - Optimized population count for fast piece counting

3. **Adaptive Search**: Position-aware search that adjusts parameters dynamically:
   - Late Move Reduction with adaptive parameters
   - Null Move Pruning with dynamic reduction factors
   - Aspiration windows for faster search convergence
   - Futility pruning with position-dependent margins

4. **Neural Network Optimizations**:
   - Mixed precision training (FP16/BF16)
   - CUDA graph capturing for faster inference
   - Tensor parallelism for large model training
   - Batch processing of positions for evaluation

5. **Memory Management**:
   - Lock-free transposition table with age-based replacement
   - Optimized hash table sizing for current hardware
   - Efficient memory layout for cache-friendly access

## Using NAGS with Arena

1. Configure the UCI executable in Arena:
   - Open Arena Chess GUI
   - Select "Engines" > "Install New Engine"
   - Navigate to `bin/nags_v1.exe` and select it
   - Configure hash size, threads, and other parameters as needed

2. Start playing or analyzing with NAGS as your engine

## Project Structure

- `src/core/`: Core C++ engine components (board representation, search)
  - `bitboard.h`: SIMD-optimized bitboard operations
  - `search.h`: Parallel alpha-beta and MCTS search
  - `evaluator.h`: Neural network integration
- `src/gnn/`: Graph Neural Network implementation
  - `evaluator_model.py`: Enhanced Transformer-GNN models
  - `board_encoder.py`: Position encoding for neural networks
- `src/mcts/`: Monte Carlo Tree Search implementation
- `src/uci/`: Universal Chess Interface protocol support
- `src/meta/`: Meta-learning components for search adaptation
  - `position_aware_scheduler.h`: Dynamic parameter tuning
- `src/training/`: Training pipeline implementations
  - `supervised_trainer.h`: Supervised learning from GM games
  - `rl_trainer.h`: Self-play reinforcement learning
  - `dual_training_pipeline.h`: Combined training pipeline
  - `curriculum_trainer.py`: Progressive difficulty training
- `scripts/`: Utility scripts for training and evaluation
- `tests/`: Test suite
- `utils/`: Helper utilities

## License

MIT