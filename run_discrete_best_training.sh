#!/bin/bash
# 使用离散超参数搜索找到的最佳参数训练三个数据集

cd /data/sony/LFCRASH/LFCRASH-CBM

# 创建输出目录
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="output/discrete_best_training_${timestamp}"
mkdir -p "$output_dir"
mkdir -p "logs"

echo "=========================================="
echo "使用离散超参数搜索最佳参数训练"
echo "时间戳: $timestamp"
echo "输出目录: $output_dir"
echo "=========================================="
echo ""

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print "GPU " $1 ": 显存 " $3 "/" $4 " MB (" int($3*100/$4) "%) | 利用率 " $5 "%"}'
echo ""

# 读取最佳超参数
echo "最佳超参数配置:"
echo "----------------------------------------"
echo "CRASH:"
echo "  AP=0.9993"
echo "  batch_size=32, lr=0.0002, h_dim=512, z_dim=512"
echo "  lambda_align=1e-05, lambda_sparse=0.002"
echo ""
echo "A3D:"
echo "  AP=0.9622"
echo "  batch_size=16, lr=5e-05, h_dim=1024, z_dim=512"
echo "  lambda_align=0.0002, lambda_sparse=0.05"
echo ""
echo "DAD:"
echo "  (等待Optuna搜索完成，使用之前最佳参数)"
echo "----------------------------------------"
echo ""

# 选择GPU（默认使用GPU 6，如果被占用则使用GPU 7）
gpu_id=6
if nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v gpu=$gpu_id '$1 == gpu && $2 > 10000 {exit 1}' ; then
    echo "使用GPU $gpu_id"
else
    gpu_id=7
    echo "GPU 6被占用，使用GPU $gpu_id"
fi

echo ""
echo "=========================================="
echo "开始训练（并行运行三个数据集）"
echo "=========================================="
echo ""

# 并行部署三个数据集
pids=()
for dataset in dad crash a3d; do
    echo "=========================================="
    echo "部署 $dataset 数据集训练"
    echo "=========================================="
    
    log_file="logs/discrete_best_training_${dataset}_${timestamp}.log"
    
    echo "日志文件: $log_file"
    echo ""
    
    # 后台运行训练
    export CUDA_VISIBLE_DEVICES=$gpu_id
    nohup python3 train_discrete_best_params.py \
        --dataset $dataset \
        --gpu $gpu_id \
        --output_dir "$output_dir" \
        > "$log_file" 2>&1 &
    
    pid=$!
    pids+=($pid)
    echo "✅ $dataset 已启动 (PID: $pid)"
    echo ""
    
    # 短暂延迟，避免同时启动导致资源竞争
    sleep 3
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
    echo "  $dataset: logs/discrete_best_training_${dataset}_${timestamp}.log"
done
echo ""
echo "输出目录: $output_dir"
echo ""
echo "💡 使用以下命令监控:"
echo "  tail -f logs/discrete_best_training_*_${timestamp}.log"
echo "  nvidia-smi"
echo ""

# 保存PID到文件
echo "${pids[@]}" > "$output_dir/pids.txt"
echo "PID已保存到: $output_dir/pids.txt"
