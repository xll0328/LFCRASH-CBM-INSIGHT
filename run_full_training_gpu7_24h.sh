#!/bin/bash
# 部署24小时完整训练实验 - GPU 7
# 使用Optuna找到的最佳参数进行完整训练（50 epochs）

set -e

# 创建日志目录
LOG_DIR="./logs/full_training_gpu7_24h"
mkdir -p -m 777 "$LOG_DIR"
echo "日志目录: $LOG_DIR"

# 创建输出目录
OUTPUT_DIR="./output/full_training_gpu7_24h"
mkdir -p -m 777 "$OUTPUT_DIR"

# 获取当前时间戳
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

echo "=========================================="
echo "部署24小时完整训练实验 - GPU 7"
echo "时间戳: $TIMESTAMP"
echo "=========================================="
echo ""
echo "配置:"
echo "  - GPU: 7"
echo "  - Epochs: 50 per dataset"
echo "  - 数据集: DAD, Crash, A3D (并行)"
echo "  - 使用Optuna最佳参数"
echo ""

# 部署DAD数据集训练
echo "启动DAD数据集训练..."
export CUDA_VISIBLE_DEVICES=7
nohup python3 train_best_params.py \
    --dataset dad \
    --gpu_id 0 \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_DIR/train_dad_${TIMESTAMP}.log" 2>&1 &
DAD_PID=$!
echo "DAD训练进程 PID: $DAD_PID"
echo "日志文件: $LOG_DIR/train_dad_${TIMESTAMP}.log"
echo ""

# 等待2秒，避免同时启动导致资源竞争
sleep 2

# 部署Crash数据集训练
echo "启动Crash数据集训练..."
export CUDA_VISIBLE_DEVICES=7
nohup python3 train_best_params.py \
    --dataset crash \
    --gpu_id 0 \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_DIR/train_crash_${TIMESTAMP}.log" 2>&1 &
CRASH_PID=$!
echo "Crash训练进程 PID: $CRASH_PID"
echo "日志文件: $LOG_DIR/train_crash_${TIMESTAMP}.log"
echo ""

# 等待2秒
sleep 2

# 部署A3D数据集训练
echo "启动A3D数据集训练..."
export CUDA_VISIBLE_DEVICES=7
nohup python3 train_best_params.py \
    --dataset a3d \
    --gpu_id 0 \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_DIR/train_a3d_${TIMESTAMP}.log" 2>&1 &
A3D_PID=$!
echo "A3D训练进程 PID: $A3D_PID"
echo "日志文件: $LOG_DIR/train_a3d_${TIMESTAMP}.log"
echo ""

echo "=========================================="
echo "所有实验已启动！"
echo "=========================================="
echo ""
echo "进程信息:"
echo "  DAD:   PID $DAD_PID"
echo "  Crash: PID $CRASH_PID"
echo "  A3D:   PID $A3D_PID"
echo ""
echo "监控命令:"
echo "  # 查看进程状态"
echo "  ps aux | grep -E 'train_best_params' | grep -v grep"
echo ""
echo "  # 查看GPU状态"
echo "  nvidia-smi -i 7 -l 1"
echo ""
echo "  # 查看日志（实时）"
echo "  tail -f $LOG_DIR/train_dad_${TIMESTAMP}.log"
echo "  tail -f $LOG_DIR/train_crash_${TIMESTAMP}.log"
echo "  tail -f $LOG_DIR/train_a3d_${TIMESTAMP}.log"
echo ""
echo "  # 停止所有实验"
echo "  pkill -f 'train_best_params.py'"
echo ""
echo "预计完成时间: 约24小时"
echo "=========================================="

# 保存PID到文件
echo "$DAD_PID" > "$LOG_DIR/pids.txt"
echo "$CRASH_PID" >> "$LOG_DIR/pids.txt"
echo "$A3D_PID" >> "$LOG_DIR/pids.txt"
echo "PID已保存到: $LOG_DIR/pids.txt"


