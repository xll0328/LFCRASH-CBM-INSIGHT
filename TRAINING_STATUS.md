# 最佳参数训练状态

## 训练已启动

**时间**: 2026-01-08 10:05
**GPU**: 3 (NVIDIA GeForce RTX 4090)
**进程ID**: 1555549

## 配置

使用experiments_gru_gpu7的最佳参数：
- **DAD**: lambda_align=0.001, lambda_sparse=0.01 (预期AP: 0.5229)
- **Crash**: lambda_align=0.001, lambda_sparse=0.01 (预期AP: 0.9977)
- **A3D**: lambda_align=0.001, lambda_sparse=0.01 (预期AP: 0.9403)

## 监控命令

```bash
# 查看训练日志
tail -f logs/best_params_gpu3/train_*.log

# 监控GPU
nvidia-smi -i 3 -l 1

# 检查进程
ps aux | grep train_best_params
```

## 修复的问题

1. ✅ CUDA设备索引问题（CUDA_VISIBLE_DEVICES=3时使用cuda:0）
2. ✅ DAD数据集labels处理问题（创建包装器）
3. ✅ toa数据类型问题（确保是tensor）

## 输出目录

- DAD: output_gru/best_params_gpu3/best_dad
- Crash: output_gru/best_params_gpu3/best_crash
- A3D: output_gru/best_params_gpu3/best_a3d

