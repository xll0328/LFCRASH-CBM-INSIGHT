#!/bin/bash
# 使用最佳参数训练三个数据集，部署到GPU 3

set -e

GPU_ID=3
OUTPUT_DIR="output_gru/best_params_gpu3"
LOG_DIR="logs/best_params_gpu3"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "使用最佳参数训练三个数据集"
echo "GPU: $GPU_ID"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
echo ""

# 检查GPU
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | grep "^$GPU_ID," || echo "警告: 无法查询GPU $GPU_ID"
echo ""

# 运行训练
echo "开始训练..."
echo ""

# 使用nohup在后台运行，并记录日志
nohup python3 train_best_params.py \
    --gpu_id $GPU_ID \
    --datasets dad crash a3d \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

PID=$!
echo "训练进程已启动，PID: $PID"
echo "日志文件: $LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "可以使用以下命令监控进度:"
echo "  tail -f $LOG_DIR/train_*.log"
echo "  nvidia-smi -i $GPU_ID -l 1"
echo "  ps aux | grep $PID"
echo ""




