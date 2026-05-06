# GPU 6 并行训练部署

## 部署信息

- **GPU**: 6
- **模式**: 三个数据集并行训练
- **启动脚本**: `run_best_params_gpu6_parallel.sh`
- **输出目录**: `output_gru/best_params_gpu6/`
- **日志目录**: `logs/best_params_gpu6/`

## 并行训练配置

三个数据集同时运行在GPU 6上：
- **DAD**: 独立进程
- **Crash**: 独立进程  
- **A3D**: 独立进程

每个数据集使用相同的GPU，通过CUDA_VISIBLE_DEVICES=6统一管理。

## 监控命令

```bash
# 查看所有日志
tail -f logs/best_params_gpu6/train_*.log

# 查看特定数据集日志
tail -f logs/best_params_gpu6/train_dad_*.log
tail -f logs/best_params_gpu6/train_crash_*.log
tail -f logs/best_params_gpu6/train_a3d_*.log

# GPU监控
nvidia-smi -i 6 -l 1

# 检查进程
ps aux | grep train_best_params
```

## 最佳参数

- **DAD**: lambda_align=0.001, lambda_sparse=0.01 (预期AP: 0.5229)
- **Crash**: lambda_align=0.001, lambda_sparse=0.01 (预期AP: 0.9977)
- **A3D**: lambda_align=0.001, lambda_sparse=0.01 (预期AP: 0.9403)

