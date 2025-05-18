#!/usr/bin/env python3
"""
Setup and run NAGS with Arena chess GUI.
"""

import os
import sys
import subprocess
import argparse
import logging
import threading
import time
import shutil
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("arena_setup.log"),
        logging.StreamHandler()
    ]
)

def find_arena_path():
    """Try to find Arena installation path."""
    if platform.system() == "Windows":
        # Common installation paths
        possible_paths = [
            "C:\\Program Files (x86)\\Arena",
            "C:\\Program Files\\Arena",
            os.path.expanduser("~\\Arena"),
            os.path.expanduser("~\\Documents\\Arena")
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "Arena.exe")):
                return path
    else:
        # Arena is primarily for Windows, but might work with Wine
        if shutil.which("wine"):
            wine_prefix = os.environ.get("WINEPREFIX", os.path.expanduser("~/.wine"))
            possible_paths = [
                os.path.join(wine_prefix, "drive_c/Program Files (x86)/Arena"),
                os.path.join(wine_prefix, "drive_c/Program Files/Arena")
            ]
            
            for path in possible_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "Arena.exe")):
                    return path
    
    return None

def start_nn_server(model_path, host="localhost", port=50051, cuda=False):
    """Start the neural network server."""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine the path to the server script
    server_script = os.path.join(script_dir, "start_nn_server.py")
    
    # Build the command
    cmd = [
        sys.executable,
        server_script,
        "--model_path", model_path,
        "--host", host,
        "--port", str(port)
    ]
    
    if cuda:
        cmd.append("--cuda")
    
    # Start the server in a new process
    logging.info(f"Starting NN server: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a bit for the server to start
    time.sleep(3)
    
    # Check if server is running
    if process.poll() is not None:
        # Server exited
        stdout, stderr = process.communicate()
        logging.error(f"NN server failed to start:\nStdout: {stdout}\nStderr: {stderr}")
        return None
    
    logging.info("NN server started successfully")
    return process

def register_engine_with_arena(engine_path, arena_path):
    """Register the NAGS engine with Arena."""
    if not os.path.exists(engine_path):
        logging.error(f"Engine executable not found at {engine_path}")
        return False
    
    if not arena_path:
        logging.error("Arena installation not found")
        return False
    
    # Check if Arena has an engines directory
    engines_dir = os.path.join(arena_path, "Engines")
    if not os.path.exists(engines_dir):
        os.makedirs(engines_dir, exist_ok=True)
    
    # Create a batch file to register the engine
    batch_file = os.path.join(os.path.dirname(engine_path), "register_nags.bat")
    with open(batch_file, 'w') as f:
        f.write(f'@echo off\n')
        f.write(f'echo Registering NAGS with Arena\n')
        f.write(f'copy "{engine_path}" "{engines_dir}\\nags_v1.exe" /Y\n')
        f.write(f'echo Engine registered successfully\n')
        f.write(f'pause\n')
    
    # Run the batch file
    logging.info(f"Running registration batch file: {batch_file}")
    process = subprocess.run(batch_file, shell=True)
    
    if process.returncode != 0:
        logging.error("Failed to register engine with Arena")
        return False
    
    logging.info("Engine registered with Arena successfully")
    return True

def start_arena(arena_path):
    """Start Arena with NAGS engine."""
    if not arena_path:
        logging.error("Arena installation not found")
        return False
    
    # Path to Arena executable
    arena_exe = os.path.join(arena_path, "Arena.exe")
    
    if not os.path.exists(arena_exe):
        logging.error(f"Arena executable not found at {arena_exe}")
        return False
    
    # Start Arena
    logging.info(f"Starting Arena from {arena_exe}")
    
    if platform.system() == "Windows":
        process = subprocess.Popen([arena_exe])
    else:
        # Try with Wine on Linux/macOS
        process = subprocess.Popen(["wine", arena_exe])
    
    time.sleep(2)
    
    if process.poll() is not None:
        logging.error("Failed to start Arena")
        return False
    
    logging.info("Arena started successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup and run NAGS with Arena")
    parser.add_argument("--engine_path", type=str, help="Path to NAGS engine executable")
    parser.add_argument("--model_path", type=str, help="Path to neural network model")
    parser.add_argument("--arena_path", type=str, help="Path to Arena installation")
    parser.add_argument("--host", type=str, default="localhost", help="NN server host")
    parser.add_argument("--port", type=int, default=50051, help="NN server port")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for neural network")
    parser.add_argument("--build_dir", type=str, default="build", help="Build directory for NAGS")
    args = parser.parse_args()
    
    # Determine engine path
    engine_path = args.engine_path
    if not engine_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        build_dir = os.path.join(project_dir, args.build_dir)
        
        if platform.system() == "Windows":
            engine_path = os.path.join(build_dir, "bin", "nags_v1.exe")
        else:
            engine_path = os.path.join(build_dir, "bin", "nags_v1")
    
    logging.info(f"Using engine at: {engine_path}")
    
    # Determine Arena path
    arena_path = args.arena_path
    if not arena_path:
        arena_path = find_arena_path()
        logging.info(f"Found Arena at: {arena_path}" if arena_path else "Arena not found")
    
    # Determine model path
    model_path = args.model_path
    if not model_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        models_dir = os.path.join(project_dir, "models")
        
        # Try to find a model
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".pt"):
                    model_path = os.path.join(models_dir, file)
                    break
    
    logging.info(f"Using model at: {model_path}" if model_path else "Model not found")
    
    # Start NN server if model is available
    nn_server_process = None
    if model_path:
        nn_server_process = start_nn_server(
            model_path=model_path,
            host=args.host,
            port=args.port,
            cuda=args.cuda
        )
    
    # Register engine with Arena
    if arena_path:
        register_engine_with_arena(engine_path, arena_path)
    
    # Start Arena
    if arena_path:
        start_arena(arena_path)
    
    # Keep the script running while NN server is active
    try:
        if nn_server_process:
            logging.info("Press Ctrl+C to stop the NN server and exit")
            nn_server_process.wait()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        if nn_server_process:
            nn_server_process.terminate()
            nn_server_process.wait()
    
    logging.info("Exited")


if __name__ == "__main__":
    main() 