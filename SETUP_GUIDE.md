# NAGS Chess Engine - Complete Setup Guide

## Prerequisites Installation

### 1. Install Python 3.8+
Download and install Python from https://www.python.org/downloads/
- **Important**: Check "Add Python to PATH" during installation
- Verify installation: `python --version` or `py --version`

### 2. Install Visual Studio Build Tools
Download from https://visualstudio.microsoft.com/downloads/
- Install "Build Tools for Visual Studio 2022"
- Select "C++ build tools" workload
- Include "Windows 10/11 SDK"

### 3. Install Git (if not already installed)
Download from https://git-scm.com/download/win

## Quick Start - Dual Training Pipeline

### Step 1: Setup Python Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Build the C++ Engine
```powershell
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the executables
cmake --build . --config Release

# Return to root directory
cd ..
```

### Step 3: Start Dual Training
```powershell
# Activate Python environment
.\venv\Scripts\activate

# Start supervised learning phase (using your PGN file)
python scripts/train_supervised.py --pgn_path AJ-CORR-PGN-000.pgn --output_dir models --epochs 50

# Alternative: Use the C++ dual training pipeline
.\build\bin\Release\nags_train.exe --pgn_path AJ-CORR-PGN-000.pgn --save_dir models --supervised_epochs 50 --rl_iterations 100
```

### Step 4: Test Your Engine
```powershell
# Your executable will be created at:
# build\bin\Release\nags_v1.exe

# Test the engine
.\build\bin\Release\nags_v1.exe
```

## Detailed Training Configuration

### Supervised Learning Phase
- **Input**: GM correspondence games from PGN file
- **Duration**: ~6-12 hours on modern hardware
- **Output**: Initial model with basic chess knowledge

### Reinforcement Learning Phase  
- **Input**: Supervised model + self-play games
- **Duration**: ~24-48 hours for significant improvement
- **Output**: Refined model with improved tactical play

## Troubleshooting

### Common Issues:
1. **Python not found**: Ensure Python is in your PATH
2. **CMake errors**: Install Visual Studio Build Tools
3. **Training crashes**: Check available RAM (needs 8GB+)
4. **Slow training**: Consider GPU acceleration with CUDA

### Performance Tips:
- Use SSD storage for faster data loading
- Close other applications during training
- Monitor CPU/GPU temperature during long training runs

## Next Steps After First Build

1. **Test in Arena Chess GUI**: Install and configure Arena
2. **Benchmark Performance**: Test against other engines
3. **Tune Parameters**: Adjust search depth, time controls
4. **Continuous Training**: Set up automated self-play cycles 