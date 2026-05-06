#!/bin/bash
set -e

GPU_ID=6
OUTPUT_DIR="output_gru/quick_test"
LOG_DIR="logs/quick_test"

mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "=========================================="
echo "快速测试：1个epoch训练 + 测试 + 可视化"
echo "GPU: ${GPU_ID}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i ${GPU_ID}

# 停止所有现有的训练进程（可选）
# pkill -f "train_best_params.py" 2>/dev/null || true
# sleep 2

echo ""
echo "开始快速测试..."

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 运行快速测试（只测试DAD数据集，最快）
nohup python3 quick_test_1epoch.py \
    --gpu_id 0 \
    --datasets dad \
    --output_dir ${OUTPUT_DIR} \
    > ${LOG_DIR}/quick_test_$(date +'%Y%m%d_%H%M%S').log 2>&1 &

PID=$!
echo "快速测试进程已启动，PID: ${PID}"
echo "日志文件: ${LOG_DIR}/quick_test_*.log"
echo ""
echo "可以使用以下命令监控进度:"
echo "  tail -f ${LOG_DIR}/quick_test_*.log"
echo "  nvidia-smi -i ${GPU_ID} -l 1"
echo "  ps aux | grep ${PID}"
echo ""




