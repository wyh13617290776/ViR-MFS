#!/bin/bash

# ==============================================================================
# ViR_MFS 实验启动脚本 (支持单卡/DDP多卡并行)
# 用法: 
#   ./run_experiment.sh [GPU_ID] [MODE]
# 示例:
#   ./run_experiment.sh 0 train        # 单卡训练
#   ./run_experiment.sh 0,1,2,3 train  # 4卡 DDP 并行训练
#   ./run_experiment.sh 1 test         # 1号卡测试
# ==============================================================================

# 1. 接收参数
GPU_ID=${1:-0}        # 默认使用 0 号显卡，支持逗号分隔如 0,1
MODE=${2:-"all"}      # 默认执行全部流程

# 2. 设置可见显卡
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 3. 巧妙地计算 GPU 数量 (通过逗号分割数组)
IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=========================================================="
echo "Starting ViR_MFS Project..."
echo "Using GPU Device(s): $GPU_ID (Total: $NUM_GPUS GPUs)"
echo "Running Mode: $MODE"
echo "=========================================================="

# 4. 执行训练 (根据 GPU 数量智能分流)
if [ "$MODE" == "train" ] || [ "$MODE" == "all" ]; then
    echo "[INFO] Starting Training..."
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "[INFO] 检测到多卡，启动 torchrun (DDP) 分布式训练..."
        # torchrun 是 PyTorch > 1.10 推荐的多卡拉起方式
        # 自动分配 RANK, WORLD_SIZE 等环境变量给你的 Python 脚本
        torchrun --nproc_per_node=$NUM_GPUS train.py
    else
        echo "[INFO] 检测到单卡，启动普通训练..."
        python train.py
    fi
fi

# 5. 执行测试 (测试通常用单卡即可，防止结果聚合麻烦)
if [ "$MODE" == "test" ] || [ "$MODE" == "all" ]; then
    echo "[INFO] Starting Testing..."
    # 强制将测试卡设置为你传入的第一张卡，避免多卡同时做推理导致日志混乱
    export CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} 
    python test.py
fi

echo "=========================================================="
echo "All tasks finished."
echo "=========================================================="