#!/bin/bash
# 检查GPU 6上的训练任务状态

cd /data/sony/LFCRASH/LFCRASH-CBM
timestamp=$(cat /tmp/lfcrash_timestamp.txt 2>/dev/null || echo "20260116_080327")

echo "=== GPU 6 训练任务状态检查 ==="
echo "实验时间戳: $timestamp"
echo ""

for dataset in dad crash a3d; do
    echo "📋 $dataset:"
    pid=$(ps aux | grep "[t]rain_best_params.py.*--dataset $dataset.*$timestamp" | awk '{print $2}' | head -1)
    if [ -n "$pid" ]; then
        echo "  ✅ 运行中 (PID: $pid)"
        log_file="logs/full_training_${dataset}_${timestamp}.log"
        if [ -f "$log_file" ]; then
            echo "  📝 日志: $log_file"
            echo "  📏 文件大小: $(du -h "$log_file" | cut -f1)"
            echo "  🕐 最后更新: $(stat -c %y "$log_file" | cut -d'.' -f1)"
            echo "  📊 当前进度:"
            tail -2 "$log_file" | grep -E "Epoch|AP|mTTA" | tail -2 | sed 's/^/    /' || tail -1 "$log_file" | sed 's/^/    /'
        fi
    else
        echo "  ⏹️  已停止"
        log_file="logs/full_training_${dataset}_${timestamp}.log"
        if [ -f "$log_file" ]; then
            echo "  📝 日志: $log_file"
            echo "  📊 最后内容:"
            tail -3 "$log_file" | sed 's/^/    /'
            # 检查是否完成
            if grep -q "训练完成\|所有训练完成" "$log_file"; then
                echo "  ✅ 训练已完成"
            fi
        fi
    fi
    echo ""
done

echo "🎮 GPU 6使用情况:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{if ($1 == "6") print "  显存: " $3 "/" $4 " MB (" int($3*100/$4) "%) | 利用率: " $5 "%"}'

echo ""
echo "💡 提示: 使用以下命令查看实时日志:"
echo "  tail -f logs/full_training_{dataset}_${timestamp}.log"
