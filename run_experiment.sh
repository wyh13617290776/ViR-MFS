#!/bin/bash
set -euo pipefail

# ==============================================================================
# ViR-MFS unified launcher.
# Usage:
#   ./run_experiment.sh [GPU_ID] [MODE] [PARAMS_PATH] [CONFIG_PATH]
# Examples:
#   ./run_experiment.sh 0 train
#   ./run_experiment.sh 0,1,2,3 train
#   ./run_experiment.sh 1 test
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_ID=${1:-0}
MODE=${2:-all}
PARAMS_PATH=${3:-config/params.yaml}
CONFIG_PATH=${4:-config/config.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ "$MODE" != "train" && "$MODE" != "test" && "$MODE" != "all" ]]; then
    echo "[ERROR] Unsupported mode: $MODE. Expected train, test, or all."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=========================================================="
echo "Starting ViR-MFS project"
echo "Project root: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
echo "GPU device(s): $GPU_ID (total: $NUM_GPUS)"
echo "Mode: $MODE"
echo "Config: $CONFIG_PATH"
echo "Params: $PARAMS_PATH"
echo "=========================================================="

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
    echo "[INFO] Starting Training..."
    
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        echo "[INFO] Multi-GPU detected. Starting DDP training with torchrun..."
        torchrun --nproc_per_node="$NUM_GPUS" train.py --config "$CONFIG_PATH" --params "$PARAMS_PATH"
    else
        echo "[INFO] Single-GPU/CPU training..."
        "$PYTHON_BIN" train.py --config "$CONFIG_PATH" --params "$PARAMS_PATH"
    fi
fi

if [[ "$MODE" == "test" || "$MODE" == "all" ]]; then
    echo "[INFO] Starting Testing..."
    export CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} 
    "$PYTHON_BIN" test.py --config "$CONFIG_PATH" --params "$PARAMS_PATH"
fi

echo "=========================================================="
echo "All tasks finished."
echo "=========================================================="
