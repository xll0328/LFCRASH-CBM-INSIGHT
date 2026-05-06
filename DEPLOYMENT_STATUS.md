# 最佳参数训练部署状态

## 当前状态

训练脚本已创建并部署到GPU 3，但遇到了一些数据加载问题需要修复：

### 遇到的问题

1. **DAD数据集**: labels处理问题（已创建包装器修复）
2. **Crash/A3D数据集**: toa数据类型问题（list of lists中的tensor转换）

### 已修复

1. ✅ CUDA设备索引（CUDA_VISIBLE_DEVICES=3时使用cuda:0）
2. ✅ DAD数据集包装器（修复labels处理）
3. ✅ toa处理逻辑（处理tensor类型）

### 训练脚本

- **文件**: train_best_params.py
- **启动**: run_best_params_gpu3.sh
- **输出**: output_gru/best_params_gpu3/
- **日志**: logs/best_params_gpu3/

### 最佳参数配置

- **DAD**: lambda_align=0.001, lambda_sparse=0.01 (预期AP: 0.5229)
- **Crash**: lambda_align=0.001, lambda_sparse=0.01 (预期AP: 0.9977)
- **A3D**: lambda_align=0.001, lambda_sparse=0.01 (预期AP: 0.9403)

### 监控

```bash
tail -f logs/best_params_gpu3/train_*.log
nvidia-smi -i 3 -l 1
```

