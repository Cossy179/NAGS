# Getting Started with NAGS

This guide will help you build and run the NAGS chess engine from source.

## Prerequisites

- **C++17 compatible compiler** (GCC 7+, Clang 5+, or MSVC 2019+)
- **CMake 3.14+** for building the C++ components
- **Python 3.8+** for the neural network components
- **CUDA Toolkit 11.7+** (optional, for GPU acceleration)
- **Arena Chess GUI** for using the engine in a chess interface

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/NAGS.git
cd NAGS
```

## Step 2: Install Python Dependencies

It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Build the Engine

```bash
# Create a build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release
```

## Step 4: Train or Download a Model

You can either train a model yourself or use a pre-trained one.

### Training a Model

```bash
python scripts/train_supervised.py --pgn_path AJ-CORR-PGN-000.pgn --output_dir models
```

### Using a Pre-trained Model

Place the pre-trained model file (`.pt` format) in the `models` directory.

## Step 5: Start the Neural Network Server

```bash
python scripts/start_nn_server.py --model_path models/best_model.pt
```

## Step 6: Configure Arena Chess GUI

You can either manually configure Arena or use our helper script:

### Using the Helper Script

```bash
python scripts/run_with_arena.py
```

### Manual Configuration

1. Open Arena Chess GUI
2. Go to Engines > Install New Engine
3. Navigate to and select the `nags_v1.exe` executable in the `build/bin` directory
4. Configure engine parameters:
   - Go to Engines > Manage > Details
   - Set Hash Size to the desired value (e.g., 128 MB)
   - Set Thread Count to your CPU's core count
   - Set GNN Server to `localhost:50051`

## Step 7: Play or Analyze with NAGS

1. In Arena, select NAGS from the engines list
2. Start a new game or analysis session
3. Enjoy!

## Troubleshooting

- **Missing PGN file**: Ensure the `AJ-CORR-PGN-000.pgn` file is in the root directory for training
- **Engine not found**: Check that the executable is in the `build/bin` directory
- **Neural network server crash**: Check CUDA compatibility and ensure required libraries are installed
- **Low performance**: Try increasing the Hash size in Arena's engine configuration

## Project Structure

- `src/core/`: Core engine components (board representation, search)
- `src/gnn/`: Graph Neural Network implementation
- `src/mcts/`: Monte Carlo Tree Search implementation
- `src/uci/`: Universal Chess Interface protocol support
- `src/meta/`: Meta-learning components
- `scripts/`: Utility scripts for training and evaluation
- `tests/`: Test suite

## Advanced Usage

### Self-Play Training

```bash
python scripts/train_rl.py --initial_model models/supervised_model.pt
```

### Benchmarking

```bash
python scripts/benchmark.py --engine_path build/bin/nags_v1.exe --opponent stockfish
```

### Tuning Meta-Learner

```bash
python scripts/tune_meta_learner.py --pgn_path AJ-CORR-PGN-000.pgn
``` 