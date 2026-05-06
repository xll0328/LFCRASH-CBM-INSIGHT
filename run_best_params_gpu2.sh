#!/bin/bash
set -e

GPU_ID=2
OUTPUT_DIR="output_gru/best_params_gpu2"
LOG_DIR="logs/best_params_gpu2"

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "=========================================="
echo "使用最佳参数训练三个数据集"
echo "GPU: ${GPU_ID}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i ${GPU_ID}

echo "开始训练..."

# 启动训练进程
CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python3 train_best_params.py \
    --gpu_id 0 \
    --datasets dad crash a3d \
    --output_dir ${OUTPUT_DIR} \
    > ${LOG_DIR}/train_$(date +'%Y%m%d_%H%M%S').log 2>&1 &
TRAIN_PID=$!

echo "训练进程已启动，PID: ${TRAIN_PID}"
echo "日志文件: ${LOG_DIR}/train_$(date +'%Y%m%d_%H%M%S').log"
echo ""

echo "可以使用以下命令监控进度:"
echo "  tail -f ${LOG_DIR}/train_*.log"
echo "  nvidia-smi -i ${GPU_ID} -l 1"
echo "  ps aux | grep ${TRAIN_PID}"




