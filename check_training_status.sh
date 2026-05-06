#!/bin/bash
# 检查训练任务状态

cd /data/sony/LFCRASH/LFCRASH-CBM

echo "=== 训练任务状态检查 ==="
echo ""

# 检查进程
echo "📊 运行中的训练进程:"
ps aux | grep "[t]rain_best_params.py" | awk '{print "  PID: " $2 " | " $11 " " $12 " " $13}'
echo ""

# 检查GPU使用情况
echo "🎮 GPU 4使用情况:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{if ($1 == "4") print "  GPU 4: " $2 " | 显存: " $3 "/" $4 " MB (" int($3*100/$4) "%) | 利用率: " $5 "%"}'
echo ""

# 检查最新日志
echo "📋 最新日志文件:"
for dataset in dad crash a3d; do
    log_file=$(ls -t logs/full_training_${dataset}_*.log 2>/dev/null | head -1)
    if [ -f "$log_file" ]; then
        echo "  $dataset: $log_file"
        echo "    最后更新: $(stat -c %y "$log_file" | cut -d'.' -f1)"
        echo "    文件大小: $(du -h "$log_file" | cut -f1)"
        echo "    最后5行:"
        tail -5 "$log_file" | sed 's/^/      /'
        echo ""
    fi
done

# 检查输出目录
echo "📁 输出目录:"
latest_output=$(ls -td output/full_training_* 2>/dev/null | head -1)
if [ -d "$latest_output" ]; then
    echo "  最新输出: $latest_output"
    for dataset in dad crash a3d; do
        dataset_dir="$latest_output/best_${dataset}"
        if [ -d "$dataset_dir" ]; then
            echo "    $dataset: $dataset_dir"
            if [ -f "$dataset_dir/results.json" ]; then
                echo "      ✓ results.json 已存在"
            fi
            if [ -f "$dataset_dir/best_model.pth" ]; then
                echo "      ✓ best_model.pth 已存在"
            fi
        fi
    done
fi
