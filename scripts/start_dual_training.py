#!/usr/bin/env python3
"""
Start the complete dual training pipeline for NAGS chess engine.
This script orchestrates both supervised learning and reinforcement learning phases.
"""

import os
import sys
import argparse
import time
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("dual_training.log"),
        logging.StreamHandler()
    ]
)

def check_prerequisites():
    """Check if all required files and tools are available."""
    issues = []
    
    # Check for PGN file
    if not os.path.exists("AJ-CORR-PGN-000.pgn"):
        issues.append("AJ-CORR-PGN-000.pgn file not found")
    
    # Check for C++ executables
    if not os.path.exists("build/bin/Release/nags_train.exe"):
        issues.append("nags_train.exe not found - run quick_start.bat first")
    
    if not os.path.exists("build/bin/Release/nags_v1.exe"):
        issues.append("nags_v1.exe not found - run quick_start.bat first")
    
    # Check for output directory
    os.makedirs("models", exist_ok=True)
    
    return issues

def run_supervised_training(pgn_path, output_dir, epochs=50, batch_size=32):
    """Run the supervised learning phase."""
    logging.info("=== Starting Supervised Learning Phase ===")
    
    cmd = [
        "python", "scripts/train_supervised.py",
        "--pgn_path", pgn_path,
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", "0.001",
        "--validation_split", "0.2",
        "--max_games", "10000"  # Start with subset for faster initial training
    ]
    
    logging.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info("Supervised training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Supervised training failed: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False

def run_cpp_dual_training(pgn_path, save_dir, supervised_epochs=30, rl_iterations=50):
    """Run the C++ dual training pipeline."""
    logging.info("=== Starting C++ Dual Training Pipeline ===")
    
    cmd = [
        "build/bin/Release/nags_train.exe",
        "--pgn_path", pgn_path,
        "--save_dir", save_dir,
        "--supervised_epochs", str(supervised_epochs),
        "--rl_iterations", str(rl_iterations),
        "--batch_size", "64",
        "--learning_rate", "0.001",
        "--num_selfplay_games", "100",
        "--time_control_ms", "1000"
    ]
    
    logging.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run with real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
            logging.info(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            logging.info("C++ dual training completed successfully")
            return True
        else:
            logging.error(f"C++ dual training failed with code {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to run C++ dual training: {e}")
        return False

def test_trained_engine(model_path=None):
    """Test the trained engine with a simple position."""
    logging.info("=== Testing Trained Engine ===")
    
    cmd = ["build/bin/Release/nags_v1.exe"]
    
    try:
        # Start engine and send some UCI commands
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, universal_newlines=True)
        
        # Send basic UCI commands
        commands = [
            "uci",
            "ucinewgame",
            "position startpos moves e2e4",
            "go depth 5",
            "quit"
        ]
        
        for command in commands:
            process.stdin.write(command + "\n")
            process.stdin.flush()
            time.sleep(0.5)
        
        stdout, stderr = process.communicate(timeout=10)
        
        if "bestmove" in stdout:
            logging.info("Engine test successful - engine is responding to UCI commands")
            return True
        else:
            logging.warning("Engine test inconclusive")
            return False
            
    except Exception as e:
        logging.error(f"Engine test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Start NAGS dual training pipeline")
    parser.add_argument("--pgn_path", default="AJ-CORR-PGN-000.pgn", 
                       help="Path to PGN file with training games")
    parser.add_argument("--output_dir", default="models", 
                       help="Directory to save trained models")
    parser.add_argument("--supervised_epochs", type=int, default=30,
                       help="Number of supervised learning epochs")
    parser.add_argument("--rl_iterations", type=int, default=50,
                       help="Number of reinforcement learning iterations")
    parser.add_argument("--use_cpp", action="store_true",
                       help="Use C++ dual training pipeline instead of Python")
    parser.add_argument("--test_only", action="store_true",
                       help="Only test the existing engine without training")
    
    args = parser.parse_args()
    
    logging.info("Starting NAGS dual training pipeline")
    logging.info(f"PGN path: {args.pgn_path}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        logging.error("Prerequisites check failed:")
        for issue in issues:
            logging.error(f"  - {issue}")
        sys.exit(1)
    
    # Test only mode
    if args.test_only:
        success = test_trained_engine()
        sys.exit(0 if success else 1)
    
    start_time = time.time()
    
    if args.use_cpp:
        # Use C++ dual training pipeline
        success = run_cpp_dual_training(
            args.pgn_path, 
            args.output_dir,
            args.supervised_epochs,
            args.rl_iterations
        )
    else:
        # Use Python supervised training
        success = run_supervised_training(
            args.pgn_path,
            args.output_dir,
            args.supervised_epochs
        )
    
    if success:
        # Test the trained engine
        test_trained_engine()
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        logging.info("=== Training Complete! ===")
        logging.info(f"Total training time: {training_duration/3600:.2f} hours")
        logging.info(f"Models saved in: {args.output_dir}")
        logging.info(f"Engine executable: build/bin/Release/nags_v1.exe")
        logging.info("")
        logging.info("Next steps:")
        logging.info("1. Test your engine: .\\build\\bin\\Release\\nags_v1.exe")
        logging.info("2. Install Arena Chess GUI and configure NAGS as an engine")
        logging.info("3. Start playing and analyzing!")
        
    else:
        logging.error("Training failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 