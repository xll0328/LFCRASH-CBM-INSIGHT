#!/bin/bash
# 在GPU 7上运行离散超参数搜索（每个数据集10个trials）

cd /data/sony/LFCRASH/LFCRASH-CBM

# 创建输出目录
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="output/optuna_discrete_${timestamp}"
mkdir -p "$output_dir"
mkdir -p "logs"

echo "=========================================="
echo "离散超参数搜索 - GPU 7"
echo "时间戳: $timestamp"
echo "输出目录: $output_dir"
echo "=========================================="
echo ""

# 检查GPU 7状态
echo "检查GPU 7状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{if ($1 == "7") print "GPU 7: 显存 " $3 "/" $4 " MB (" int($3*100/$4) "%) | 利用率 " $5 "%"}'
echo ""

# 部署三个数据集（顺序运行，因为只有10个trials，不会太久）
for dataset in dad crash a3d; do
    echo "=========================================="
    echo "部署 $dataset 数据集（10 trials）"
    echo "=========================================="
    
    log_file="logs/discrete_optuna_${dataset}_${timestamp}.log"
    study_name="discrete_${dataset}_${timestamp}"
    
    echo "日志文件: $log_file"
    echo "Study名称: $study_name"
    echo ""
    
    # 运行Optuna优化
    export CUDA_VISIBLE_DEVICES=7
    python3 optuna_optimize_discrete.py \
        --dataset $dataset \
        --n_trials 10 \
        --n_epochs 50 \
        --output_dir "$output_dir" \
        --study_name "$study_name" \
        > "$log_file" 2>&1
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $dataset 完成"
        echo "最佳结果:"
        tail -20 "$log_file" | grep -E "最佳AP|最佳参数|Trial" | head -15
    else
        echo "❌ $dataset 失败 (退出码: $exit_code)"
        echo "错误信息:"
        tail -20 "$log_file" | grep -E "Error|Exception|Traceback" | head -10
    fi
    
    echo ""
    echo "等待5秒后继续下一个数据集..."
    sleep 5
    echo ""
done

echo "=========================================="
echo "所有数据集完成！"
echo "=========================================="
echo ""
echo "结果文件:"
for dataset in dad crash a3d; do
    results_file="$output_dir/${dataset}_discrete_optuna_results.json"
    if [ -f "$results_file" ]; then
        echo "  ✅ $results_file"
    else
        echo "  ❌ $results_file (未找到)"
    fi
done
echo ""
echo "日志文件: logs/discrete_optuna_*_${timestamp}.log"
echo ""
