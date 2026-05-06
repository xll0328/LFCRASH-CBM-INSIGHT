#!/bin/bash
set -e

GPU_ID=6
OPTUNA_DIR="optuna_studies/gpu6_30trials"
LOG_DIR="logs/optuna_30trials_gpu6"

mkdir -p $OPTUNA_DIR
mkdir -p $LOG_DIR

echo "=========================================="
echo "部署三个数据集的Optuna搜参实验（30个trials）"
echo "GPU: ${GPU_ID}"
echo "Optuna目录: ${OPTUNA_DIR}"
echo "日志目录: ${LOG_DIR}"
echo "=========================================="

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i ${GPU_ID}

# 停止所有现有的训练进程
echo "停止所有现有的训练进程..."
pkill -f "train_best_params.py" 2>/dev/null || true
pkill -f "train_optuna_multi_dataset.py" 2>/dev/null || true
sleep 2

# 清理GPU缓存
echo "清理GPU缓存..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo ""
echo "开始部署Optuna搜参实验..."

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# ==========================================
# 启动三个数据集的Optuna搜参实验（并行运行）
# ==========================================
echo ""
echo "=========================================="
echo "启动三个数据集的Optuna搜参实验（每个30个trials）"
echo "=========================================="

# DAD数据集Optuna搜参
nohup python3 train_optuna_multi_dataset.py \
    --dataset dad \
    --gpu_id 0 \
    --n_trials 30 \
    --n_epochs 15 \
    > ${LOG_DIR}/optuna_dad_${TIMESTAMP}.log 2>&1 &
DAD_OPTUNA_PID=$!
echo "DAD Optuna搜参进程已启动，PID: ${DAD_OPTUNA_PID}"

# Crash数据集Optuna搜参
nohup python3 train_optuna_multi_dataset.py \
    --dataset crash \
    --gpu_id 0 \
    --n_trials 30 \
    --n_epochs 15 \
    > ${LOG_DIR}/optuna_crash_${TIMESTAMP}.log 2>&1 &
CRASH_OPTUNA_PID=$!
echo "Crash Optuna搜参进程已启动，PID: ${CRASH_OPTUNA_PID}"

# A3D数据集Optuna搜参
nohup python3 train_optuna_multi_dataset.py \
    --dataset a3d \
    --gpu_id 0 \
    --n_trials 30 \
    --n_epochs 15 \
    > ${LOG_DIR}/optuna_a3d_${TIMESTAMP}.log 2>&1 &
A3D_OPTUNA_PID=$!
echo "A3D Optuna搜参进程已启动，PID: ${A3D_OPTUNA_PID}"

echo ""
echo "=========================================="
echo "所有Optuna搜参实验已启动！"
echo "=========================================="
echo ""
echo "Optuna搜参进程："
echo "  DAD:   PID ${DAD_OPTUNA_PID} (30 trials)"
echo "  Crash: PID ${CRASH_OPTUNA_PID} (30 trials)"
echo "  A3D:   PID ${A3D_OPTUNA_PID} (30 trials)"
echo ""
echo "监控命令："
echo "  # 查看GPU使用情况"
echo "  nvidia-smi -i ${GPU_ID} -l 1"
echo ""
echo "  # 查看进程状态"
echo "  ps aux | grep -E 'train_optuna' | grep -v grep"
echo ""
echo "  # 查看Optuna搜参日志"
echo "  tail -f ${LOG_DIR}/optuna_dad_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/optuna_crash_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/optuna_a3d_${TIMESTAMP}.log"
echo ""
echo "输出目录："
echo "  Optuna结果: ${OPTUNA_DIR}"
echo ""



