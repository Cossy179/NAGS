#!/usr/bin/env bash
# NAGS training pipeline runner (Linux / macOS).
#
# Thin wrapper around training_pipeline.py, which starts and stops the GNN and
# meta-learner services itself, plays the self-play / evaluation games and
# promotes models. Extra arguments are passed through, e.g.
#   ./run_training.sh selfplay --games 20
#   ./run_training.sh evaluate --model models/ppo_model_20250101_120000.pth

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python3}"

usage() {
    echo "Usage: $0 [step] [pipeline options]"
    echo "Steps: parse, supervised, selfplay, ppo, evaluate, meta, full (default: full)"
    echo "Options are passed to training_pipeline.py (see: $PYTHON training_pipeline.py --help)"
}

STEP="${1:-full}"
case "$STEP" in
    help|-h|--help) usage; exit 0 ;;
    parse|supervised|selfplay|ppo|evaluate|meta|full) ;;
    *) echo "Unknown step: $STEP" >&2; usage >&2; exit 1 ;;
esac
[[ $# -gt 0 ]] && shift

if ! "$PYTHON" -c "import torch, torch_geometric, chess" 2>/dev/null; then
    echo "ERROR: missing Python dependencies. Install with: $PYTHON -m pip install -r requirements.txt" >&2
    exit 1
fi

case "$STEP" in
    selfplay|evaluate|full)
        if [[ ! -x build/nags && ! -f build/Release/nags.exe && ! -x build/Release/nags ]]; then
            echo "ERROR: NAGS engine not found. Build it with:" >&2
            echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release" >&2
            exit 1
        fi
        ;;
esac

mkdir -p logs
LOG_FILE="logs/run_training_$(date +%Y%m%d_%H%M%S).log"
echo "Running step '$STEP' (log: $LOG_FILE)"
"$PYTHON" training_pipeline.py --step "$STEP" "$@" 2>&1 | tee -a "$LOG_FILE"
