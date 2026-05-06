#!/bin/bash
# 在GPU 7上并行运行离散超参数搜索（每个数据集10个trials）

cd /data/sony/LFCRASH/LFCRASH-CBM

# 创建输出目录
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="output/optuna_discrete_${timestamp}"
mkdir -p "$output_dir"
mkdir -p "logs"

echo "=========================================="
echo "离散超参数搜索 - GPU 7 (并行运行)"
echo "时间戳: $timestamp"
echo "输出目录: $output_dir"
echo "=========================================="
echo ""

# 检查GPU 7状态
echo "检查GPU 7状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{if ($1 == "7") print "GPU 7: 显存 " $3 "/" $4 " MB (" int($3*100/$4) "%) | 利用率 " $5 "%"}'
echo ""

# 并行部署三个数据集
pids=()
for dataset in dad crash a3d; do
    echo "=========================================="
    echo "部署 $dataset 数据集（10 trials）- 并行运行"
    echo "=========================================="
    
    log_file="logs/discrete_optuna_${dataset}_${timestamp}.log"
    study_name="discrete_${dataset}_${timestamp}"
    
    echo "日志文件: $log_file"
    echo "Study名称: $study_name"
    echo ""
    
    # 后台运行Optuna优化
    export CUDA_VISIBLE_DEVICES=7
    nohup python3 optuna_optimize_discrete.py \
        --dataset $dataset \
        --n_trials 10 \
        --n_epochs 50 \
        --output_dir "$output_dir" \
        --study_name "$study_name" \
        > "$log_file" 2>&1 &
    
    pid=$!
    pids+=($pid)
    echo "✅ $dataset 已启动 (PID: $pid)"
    echo ""
    
    # 短暂延迟，避免同时启动导致资源竞争
    sleep 2
done

echo "=========================================="
echo "所有数据集已并行启动"
echo "=========================================="
echo ""
echo "进程ID:"
for i in "${!pids[@]}"; do
    dataset=("dad" "crash" "a3d")
    echo "  ${dataset[$i]}: PID ${pids[$i]}"
done
echo ""
echo "日志文件:"
for dataset in dad crash a3d; do
    echo "  $dataset: logs/discrete_optuna_${dataset}_${timestamp}.log"
done
echo ""
echo "输出目录: $output_dir"
echo ""
echo "💡 使用以下命令监控:"
echo "  ./check_discrete_optuna_status.sh"
echo "  tail -f logs/discrete_optuna_*_${timestamp}.log"
echo ""

# 保存PID到文件，方便后续检查
echo "${pids[@]}" > "$output_dir/pids.txt"
echo "PID已保存到: $output_dir/pids.txt"
