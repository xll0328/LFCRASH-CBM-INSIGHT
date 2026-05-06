#!/bin/bash
# 部署CRASH对齐后的完整训练实验
# GPU 7, 24小时训练任务

GPU_ID=7
OUTPUT_DIR="./output/crash_aligned_training_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs/crash_aligned_training_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "部署CRASH对齐后的完整训练实验"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "输出目录: $OUTPUT_DIR"
echo "日志目录: $LOG_DIR"
echo "时间: $(date)"
echo ""

# 部署三个数据集（并行）
datasets=("dad" "crash" "a3d")
pids=()

for dataset in "${datasets[@]}"; do
    echo "启动 $dataset 数据集训练..."
    log_file="$LOG_DIR/train_${dataset}.log"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python3 train_best_params.py \
        --dataset "$dataset" \
        --gpu_id 0 \
        --output_dir "$OUTPUT_DIR" \
        > "$log_file" 2>&1 &
    
    pid=$!
    pids+=($pid)
    echo "  PID: $pid"
    echo "  日志: $log_file"
    echo ""
    
    sleep 5  # 间隔启动
done

# 保存PID
echo "${pids[@]}" > "$LOG_DIR/pids.txt"
echo "所有进程PID: ${pids[@]}"
echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "监控命令:"
echo "  # 查看进程状态"
echo "  ps aux | grep 'train_best_params.py' | grep -v grep"
echo ""
echo "  # 查看GPU状态"
echo "  nvidia-smi -i $GPU_ID -l 1"
echo ""
echo "  # 查看日志（实时）"
echo "  tail -f $LOG_DIR/train_*.log"
echo ""
echo "  # 停止所有实验"
echo "  pkill -f 'train_best_params.py'"
echo ""
echo "预计完成时间: 约24小时"
echo "=========================================="
