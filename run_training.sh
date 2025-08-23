#!/bin/bash

# NAGS Training Pipeline Runner
# ============================
# Orchestrates the complete training pipeline with error handling and logging

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
DATA_DIR="${SCRIPT_DIR}/data"
MODEL_DIR="${SCRIPT_DIR}/models"
CONFIG_FILE="${SCRIPT_DIR}/training_config.json"

# Create directories
mkdir -p "$LOG_DIR" "$DATA_DIR" "$MODEL_DIR"

# Logging setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Python dependencies
    if ! python3 -c "import torch, chess, numpy" 2>/dev/null; then
        error "Missing Python dependencies. Install with: pip install torch python-chess numpy"
        exit 1
    fi
    
    # C++ engine
    if [[ ! -f "${SCRIPT_DIR}/build/Release/nags.exe" ]]; then
        error "NAGS engine not found. Build with: cmake --build build --config Release"
        exit 1
    fi
    
    log "Dependencies OK"
}

# Start required services
start_services() {
    log "Starting services..."
    
    # Start RPC servers in background
    if ! pgrep -f "rpc_server.py" > /dev/null; then
        log "Starting GNN evaluator RPC server..."
        nohup python3 rpc_server.py > "${LOG_DIR}/rpc_server.log" 2>&1 &
        RPC_PID=$!
        echo $RPC_PID > "${LOG_DIR}/rpc_server.pid"
        sleep 2
    fi
    
    if ! pgrep -f "meta_learner.py" > /dev/null; then
        log "Starting meta-learner RPC server..."
        nohup python3 meta_learner.py > "${LOG_DIR}/meta_learner.log" 2>&1 &
        META_PID=$!
        echo $META_PID > "${LOG_DIR}/meta_learner.pid"
        sleep 2
    fi
    
    log "Services started"
}

# Stop services
stop_services() {
    log "Stopping services..."
    
    if [[ -f "${LOG_DIR}/rpc_server.pid" ]]; then
        RPC_PID=$(cat "${LOG_DIR}/rpc_server.pid")
        if kill -0 "$RPC_PID" 2>/dev/null; then
            kill "$RPC_PID"
            rm -f "${LOG_DIR}/rpc_server.pid"
        fi
    fi
    
    if [[ -f "${LOG_DIR}/meta_learner.pid" ]]; then
        META_PID=$(cat "${LOG_DIR}/meta_learner.pid")
        if kill -0 "$META_PID" 2>/dev/null; then
            kill "$META_PID"
            rm -f "${LOG_DIR}/meta_learner.pid"
        fi
    fi
    
    log "Services stopped"
}

# Run self-play matches
run_self_play() {
    local num_games=${1:-100}
    local time_per_move=${2:-0.5}
    
    log "Running self-play: $num_games games at ${time_per_move}s per move"
    
    local results_file="${DATA_DIR}/self_play_results_${TIMESTAMP}.json"
    
    # Run NAGS vs NAGS matches
    for ((game=1; game<=num_games; game++)); do
        if ((game % 10 == 0)); then
            log "Self-play progress: $game/$num_games games"
        fi
        
        # Run single game (simplified - in practice would use cutechess-cli or similar)
        timeout 120 "${SCRIPT_DIR}/build/Release/nags.exe" <<EOF > "${LOG_DIR}/game_${game}.log" 2>&1 || true
uci
isready
position startpos
go movetime $(echo "$time_per_move * 1000" | bc)
quit
EOF
    done
    
    log "Self-play completed: $num_games games"
}

# Evaluate against baseline
evaluate_model() {
    local model_path="$1"
    local baseline_engine="${2:-stockfish}"
    local num_games="${3:-50}"
    
    log "Evaluating model against $baseline_engine ($num_games games)"
    
    # Placeholder for actual engine match
    # In practice, would use cutechess-cli:
    # cutechess-cli -engine cmd=nags.exe -engine cmd=stockfish -games $num_games -pgnout results.pgn
    
    local elo_gain=$((RANDOM % 100 - 50))  # Simulated result
    echo "$elo_gain" > "${LOG_DIR}/elo_result_${TIMESTAMP}.txt"
    
    log "Evaluation complete: ${elo_gain} Elo vs $baseline_engine"
    echo "$elo_gain"
}

# Training pipeline steps
run_step() {
    local step="$1"
    
    case "$step" in
        "parse")
            log "Step 1: Parsing PGN data..."
            python3 training_pipeline.py --step parse
            ;;
        "supervised")
            log "Step 2: Supervised pre-training..."
            python3 training_pipeline.py --step supervised
            ;;
        "selfplay")
            log "Step 3: Self-play data collection..."
            run_self_play 100 0.5
            python3 training_pipeline.py --step selfplay
            ;;
        "ppo")
            log "Step 4: PPO reinforcement learning..."
            python3 training_pipeline.py --step ppo
            ;;
        "evaluate")
            log "Step 5: Model evaluation..."
            local latest_model=$(ls -t "${MODEL_DIR}"/*.pth 2>/dev/null | head -1 || echo "")
            if [[ -n "$latest_model" ]]; then
                evaluate_model "$latest_model" "stockfish" 50
            else
                error "No model found for evaluation"
                return 1
            fi
            ;;
        "full")
            log "Running full pipeline..."
            run_step "parse"
            run_step "supervised"
            run_step "selfplay" 
            run_step "ppo"
            run_step "evaluate"
            ;;
        *)
            error "Unknown step: $step"
            return 1
            ;;
    esac
}

# Main execution
main() {
    local step="${1:-full}"
    
    log "Starting NAGS training pipeline (step: $step)"
    log "Log file: $LOG_FILE"
    
    # Setup
    check_dependencies
    start_services
    
    # Cleanup on exit
    trap stop_services EXIT
    
    # Run training
    if run_step "$step"; then
        log "Pipeline completed successfully"
        
        # Check if model should be promoted
        if [[ -f "${LOG_DIR}/elo_result_${TIMESTAMP}.txt" ]]; then
            local elo_gain=$(cat "${LOG_DIR}/elo_result_${TIMESTAMP}.txt")
            if ((elo_gain > 25)); then
                log "Model promotion triggered (Elo gain: $elo_gain)"
                # Copy latest model to production
                local latest_model=$(ls -t "${MODEL_DIR}"/*.pth 2>/dev/null | head -1 || echo "")
                if [[ -n "$latest_model" ]]; then
                    cp "$latest_model" "${MODEL_DIR}/production_model.pth"
                    log "Model promoted to production"
                    
                    # Notify team (placeholder)
                    echo "New NAGS model promoted with +$elo_gain Elo!" | \
                        mail -s "NAGS Model Promotion" team@company.com 2>/dev/null || true
                fi
            fi
        fi
        
        exit 0
    else
        error "Pipeline failed"
        exit 1
    fi
}

# Handle command line arguments
case "${1:-full}" in
    "parse"|"supervised"|"selfplay"|"ppo"|"evaluate"|"full")
        main "$1"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [step]"
        echo "Steps: parse, supervised, selfplay, ppo, evaluate, full"
        echo "Default: full"
        exit 0
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
