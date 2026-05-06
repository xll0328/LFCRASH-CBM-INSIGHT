#!/bin/bash
set -e

GPU_ID=6
OUTPUT_DIR="output_gru/full_training_gpu6"
OPTUNA_DIR="optuna_studies/gpu6"
LOG_DIR="logs/full_training_and_optuna_gpu6"

mkdir -p $OUTPUT_DIR
mkdir -p $OPTUNA_DIR
mkdir -p $LOG_DIR

echo "=========================================="
echo "部署完整训练和Optuna搜参实验"
echo "GPU: ${GPU_ID}"
echo "输出目录: ${OUTPUT_DIR}"
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
echo "开始部署实验..."

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# ==========================================
# 第一部分：完整训练（使用各自最佳参数）
# ==========================================
echo ""
echo "=========================================="
echo "启动三个数据集的完整训练（使用各自最佳参数）"
echo "=========================================="

# DAD数据集完整训练
nohup python3 train_best_params.py \
    --gpu_id 0 \
    --datasets dad \
    --output_dir ${OUTPUT_DIR} \
    > ${LOG_DIR}/train_dad_${TIMESTAMP}.log 2>&1 &
DAD_TRAIN_PID=$!
echo "DAD完整训练进程已启动，PID: ${DAD_TRAIN_PID}"

# Crash数据集完整训练
nohup python3 train_best_params.py \
    --gpu_id 0 \
    --datasets crash \
    --output_dir ${OUTPUT_DIR} \
    > ${LOG_DIR}/train_crash_${TIMESTAMP}.log 2>&1 &
CRASH_TRAIN_PID=$!
echo "Crash完整训练进程已启动，PID: ${CRASH_TRAIN_PID}"

# A3D数据集完整训练
nohup python3 train_best_params.py \
    --gpu_id 0 \
    --datasets a3d \
    --output_dir ${OUTPUT_DIR} \
    > ${LOG_DIR}/train_a3d_${TIMESTAMP}.log 2>&1 &
A3D_TRAIN_PID=$!
echo "A3D完整训练进程已启动，PID: ${A3D_TRAIN_PID}"

# ==========================================
# 第二部分：Optuna搜参实验（并行运行）
# ==========================================
echo ""
echo "=========================================="
echo "启动三个数据集的Optuna搜参实验（并行运行）"
echo "=========================================="

# DAD数据集Optuna搜参
nohup python3 train_optuna_multi_dataset.py \
    --dataset dad \
    --gpu_id 0 \
    --n_trials 20 \
    --n_epochs 15 \
    > ${LOG_DIR}/optuna_dad_${TIMESTAMP}.log 2>&1 &
DAD_OPTUNA_PID=$!
echo "DAD Optuna搜参进程已启动，PID: ${DAD_OPTUNA_PID}"

# Crash数据集Optuna搜参
nohup python3 train_optuna_multi_dataset.py \
    --dataset crash \
    --gpu_id 0 \
    --n_trials 20 \
    --n_epochs 15 \
    > ${LOG_DIR}/optuna_crash_${TIMESTAMP}.log 2>&1 &
CRASH_OPTUNA_PID=$!
echo "Crash Optuna搜参进程已启动，PID: ${CRASH_OPTUNA_PID}"

# A3D数据集Optuna搜参
nohup python3 train_optuna_multi_dataset.py \
    --dataset a3d \
    --gpu_id 0 \
    --n_trials 20 \
    --n_epochs 15 \
    > ${LOG_DIR}/optuna_a3d_${TIMESTAMP}.log 2>&1 &
A3D_OPTUNA_PID=$!
echo "A3D Optuna搜参进程已启动，PID: ${A3D_OPTUNA_PID}"

echo ""
echo "=========================================="
echo "所有实验已启动！"
echo "=========================================="
echo ""
echo "完整训练进程："
echo "  DAD:   PID ${DAD_TRAIN_PID}"
echo "  Crash: PID ${CRASH_TRAIN_PID}"
echo "  A3D:   PID ${A3D_TRAIN_PID}"
echo ""
echo "Optuna搜参进程："
echo "  DAD:   PID ${DAD_OPTUNA_PID}"
echo "  Crash: PID ${CRASH_OPTUNA_PID}"
echo "  A3D:   PID ${A3D_OPTUNA_PID}"
echo ""
echo "监控命令："
echo "  # 查看GPU使用情况"
echo "  nvidia-smi -i ${GPU_ID} -l 1"
echo ""
echo "  # 查看进程状态"
echo "  ps aux | grep -E 'train_best_params|train_optuna' | grep -v grep"
echo ""
echo "  # 查看完整训练日志"
echo "  tail -f ${LOG_DIR}/train_dad_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/train_crash_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/train_a3d_${TIMESTAMP}.log"
echo ""
echo "  # 查看Optuna搜参日志"
echo "  tail -f ${LOG_DIR}/optuna_dad_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/optuna_crash_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/optuna_a3d_${TIMESTAMP}.log"
echo ""
echo "  # 查看所有日志的最新内容"
echo "  tail -f ${LOG_DIR}/*.log"
echo ""
echo "输出目录："
echo "  完整训练结果: ${OUTPUT_DIR}"
echo "  Optuna结果: ${OPTUNA_DIR}"
echo ""




