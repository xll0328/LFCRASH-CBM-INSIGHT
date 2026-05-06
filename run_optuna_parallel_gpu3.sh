#!/bin/bash
# 在GPU 3上并行运行三个数据集的Optuna搜索
# 每个数据集20个trials，同时运行3个实验

set -e

GPU_ID=3
N_TRIALS=20
N_EPOCHS=15
NUM_WORKERS=2  # 每个实验使用2个worker，3个实验共6个，避免过多进程

# 创建输出目录
OUTPUT_DIR="optuna_studies/gpu${GPU_ID}"
mkdir -p "${OUTPUT_DIR}/logs"

# 日志文件
MAIN_LOG="${OUTPUT_DIR}/parallel_search_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee -a "${MAIN_LOG}"
echo "开始并行Optuna搜索" | tee -a "${MAIN_LOG}"
echo "GPU: ${GPU_ID}" | tee -a "${MAIN_LOG}"
echo "每个数据集: ${N_TRIALS} trials, ${N_EPOCHS} epochs" | tee -a "${MAIN_LOG}"
echo "时间: $(date)" | tee -a "${MAIN_LOG}"
echo "==========================================" | tee -a "${MAIN_LOG}"

# 检查GPU可用性
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi 未找到" | tee -a "${MAIN_LOG}"
    exit 1
fi

# 显示GPU状态
echo "GPU ${GPU_ID} 初始状态:" | tee -a "${MAIN_LOG}"
nvidia-smi --id=${GPU_ID} --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "${MAIN_LOG}"

# 数据集列表
DATASETS=("dad" "crash" "a3d")

# 启动三个数据集的搜索（后台运行）
PIDS=()
for dataset in "${DATASETS[@]}"; do
    echo "" | tee -a "${MAIN_LOG}"
    echo "启动 ${dataset} 数据集的Optuna搜索..." | tee -a "${MAIN_LOG}"
    
    # 为每个数据集创建独立的日志文件
    DATASET_LOG="${OUTPUT_DIR}/logs/optuna_${dataset}_$(date +%Y%m%d_%H%M%S).log"
    
    # 使用nohup在后台运行，并设置CUDA_VISIBLE_DEVICES
    nohup bash -c "
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        cd $(pwd)
        python3 train_optuna_multi_dataset.py \
            --dataset ${dataset} \
            --gpu_id 0 \
            --n_trials ${N_TRIALS} \
            --n_epochs ${N_EPOCHS} \
            --num_workers ${NUM_WORKERS} \
            > ${DATASET_LOG} 2>&1
    " &
    
    PID=$!
    PIDS+=($PID)
    
    echo "  ${dataset} 搜索已启动，PID: ${PID}, 日志: ${DATASET_LOG}" | tee -a "${MAIN_LOG}"
    
    # 稍微延迟，避免同时初始化导致内存峰值
    sleep 5
done

echo "" | tee -a "${MAIN_LOG}"
echo "所有搜索任务已启动!" | tee -a "${MAIN_LOG}"
echo "进程ID: ${PIDS[@]}" | tee -a "${MAIN_LOG}"
echo "" | tee -a "${MAIN_LOG}"

# 监控函数
monitor_processes() {
    while true; do
        # 检查所有进程是否还在运行
        all_running=true
        for pid in "${PIDS[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                all_running=false
            fi
        done
        
        if [ "$all_running" = false ]; then
            break
        fi
        
        # 显示GPU使用情况
        echo "[$(date +%H:%M:%S)] GPU ${GPU_ID} 状态:" | tee -a "${MAIN_LOG}"
        nvidia-smi --id=${GPU_ID} --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "${MAIN_LOG}"
        
        sleep 60  # 每分钟检查一次
    done
}

# 等待所有进程完成
echo "等待所有搜索任务完成..." | tee -a "${MAIN_LOG}"
echo "（按Ctrl+C可以中断，但会等待当前trial完成）" | tee -a "${MAIN_LOG}"

# 在后台启动监控
monitor_processes &
MONITOR_PID=$!

# 等待所有搜索进程
for pid in "${PIDS[@]}"; do
    wait $pid
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "进程 ${pid} 完成 (退出码: ${EXIT_CODE})" | tee -a "${MAIN_LOG}"
    else
        echo "进程 ${pid} 异常退出 (退出码: ${EXIT_CODE})" | tee -a "${MAIN_LOG}"
    fi
done

# 停止监控
kill $MONITOR_PID 2>/dev/null || true

echo "" | tee -a "${MAIN_LOG}"
echo "==========================================" | tee -a "${MAIN_LOG}"
echo "所有搜索任务已完成!" | tee -a "${MAIN_LOG}"
echo "完成时间: $(date)" | tee -a "${MAIN_LOG}"
echo "==========================================" | tee -a "${MAIN_LOG}"

# 显示最终GPU状态
echo "GPU ${GPU_ID} 最终状态:" | tee -a "${MAIN_LOG}"
nvidia-smi --id=${GPU_ID} --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "${MAIN_LOG}"

# 汇总结果
echo "" | tee -a "${MAIN_LOG}"
echo "结果汇总:" | tee -a "${MAIN_LOG}"
for dataset in "${DATASETS[@]}"; do
    RESULT_FILE=$(ls -t ${OUTPUT_DIR}/optuna_*_${dataset}_*_results.json 2>/dev/null | head -1)
    if [ -n "$RESULT_FILE" ] && [ -f "$RESULT_FILE" ]; then
        echo "  ${dataset}:" | tee -a "${MAIN_LOG}"
        python3 -c "
import json
with open('${RESULT_FILE}', 'r') as f:
    data = json.load(f)
    print(f\"    最佳AP: {data.get('best_value', 'N/A'):.4f}\")
    print(f\"    最佳参数: {data.get('best_params', {})}\")
    print(f\"    完成: {data.get('n_complete', 0)}, 剪枝: {data.get('n_pruned', 0)}, 失败: {data.get('n_fail', 0)}\")
" | tee -a "${MAIN_LOG}"
    else
        echo "  ${dataset}: 结果文件未找到" | tee -a "${MAIN_LOG}"
    fi
done

echo "" | tee -a "${MAIN_LOG}"
echo "主日志: ${MAIN_LOG}" | tee -a "${MAIN_LOG}"
echo "各数据集日志: ${OUTPUT_DIR}/logs/" | tee -a "${MAIN_LOG}"




