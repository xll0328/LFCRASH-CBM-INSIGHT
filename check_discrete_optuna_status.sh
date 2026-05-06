#!/bin/bash
# 检查离散超参数搜索实验状态

cd /data/sony/LFCRASH/LFCRASH-CBM

echo "=========================================="
echo "离散超参数搜索实验状态检查"
echo "=========================================="
echo ""

# 检查进程
echo "📋 运行中的进程:"
ps aux | grep -E "[o]ptuna_optimize_discrete|[r]un_discrete_optuna" | grep -v grep | head -10
echo ""

# 检查日志文件
echo "📝 日志文件:"
ls -lth logs/discrete_optuna_*.log 2>/dev/null | head -5
echo ""

# 检查输出目录
echo "📁 输出目录:"
ls -lth output/optuna_discrete_*/ 2>/dev/null | head -5
echo ""

# 检查最新日志内容
latest_log=$(ls -t logs/discrete_optuna_all_*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "📊 最新日志 ($latest_log):"
    echo "----------------------------------------"
    tail -30 "$latest_log"
    echo "----------------------------------------"
else
    echo "⚠️  未找到日志文件"
fi

echo ""

# 检查GPU状态
echo "🎮 GPU 7状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{if ($1 == "7") print "  显存: " $3 "/" $4 " MB (" int($3*100/$4) "%) | 利用率: " $5 "%"}'

echo ""

# 检查结果文件
echo "📈 结果文件:"
for dataset in dad crash a3d; do
    result_file=$(find output/optuna_discrete_* -name "${dataset}_discrete_optuna_results.json" 2>/dev/null | head -1)
    if [ -n "$result_file" ]; then
        echo "  ✅ $dataset: $result_file"
        if command -v jq &> /dev/null; then
            echo "     最佳AP: $(jq -r '.best_ap' "$result_file" 2>/dev/null)"
        fi
    else
        echo "  ⏳ $dataset: 尚未完成"
    fi
done

echo ""
echo "💡 提示: 使用以下命令查看实时日志:"
echo "  tail -f logs/discrete_optuna_all_*.log"
