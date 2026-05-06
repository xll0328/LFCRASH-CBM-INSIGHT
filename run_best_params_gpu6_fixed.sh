#!/bin/bash
set -e

GPU_ID=6
OUTPUT_DIR="output_gru/best_params_gpu6_fixed"
LOG_DIR="logs/best_params_gpu6_fixed"

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "=========================================="
echo "使用修复后的代码并行训练三个数据集"
echo "GPU: ${GPU_ID}"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志目录: ${LOG_DIR}"
echo "=========================================="

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i ${GPU_ID}

# 停止所有现有的训练进程
echo "停止所有现有的训练进程..."
pkill -f "train_best_params.py" 2>/dev/null || true
sleep 2

# 清理GPU缓存
echo "清理GPU缓存..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo ""
echo "开始并行训练三个数据集..."

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 并行启动三个数据集的训练
# DAD数据集
nohup python3 train_best_params.py \
    --gpu_id 0 \
    --datasets dad \
    --output_dir ${OUTPUT_DIR} \
    > ${LOG_DIR}/train_dad_$(date +'%Y%m%d_%H%M%S').log 2>&1 &
DAD_PID=$!
echo "DAD训练进程已启动，PID: ${DAD_PID}"

# Crash数据集
nohup python3 train_best_params.py \
    --gpu_id 0 \
    --datasets crash \
    --output_dir ${OUTPUT_DIR} \
    > ${LOG_DIR}/train_crash_$(date +'%Y%m%d_%H%M%S').log 2>&1 &
CRASH_PID=$!
echo "Crash训练进程已启动，PID: ${CRASH_PID}"

# A3D数据集
nohup python3 train_best_params.py \
    --gpu_id 0 \
    --datasets a3d \
    --output_dir ${OUTPUT_DIR} \
    > ${LOG_DIR}/train_a3d_$(date +'%Y%m%d_%H%M%S').log 2>&1 &
A3D_PID=$!
echo "A3D训练进程已启动，PID: ${A3D_PID}"

echo ""
echo "所有训练进程已启动："
echo "  DAD:   PID ${DAD_PID}"
echo "  Crash: PID ${CRASH_PID}"
echo "  A3D:   PID ${A3D_PID}"
echo ""

echo "可以使用以下命令监控进度:"
echo "  tail -f ${LOG_DIR}/train_dad_*.log"
echo "  tail -f ${LOG_DIR}/train_crash_*.log"
echo "  tail -f ${LOG_DIR}/train_a3d_*.log"
echo "  nvidia-smi -i ${GPU_ID} -l 1"
echo "  ps aux | grep train_best_params"
echo ""




