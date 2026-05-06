# 最佳参数训练状态

## 训练配置

### 最佳参数（基于experiments_gru_gpu7结果）

**DAD数据集**:
- lambda_align: 0.001
- lambda_sparse: 0.01
- batch_size: 16
- learning_rate: 1e-3
- weight_decay: 1e-5
- h_dim: 512, z_dim: 256
- n_epochs: 25
- 预期AP: 0.5229

**Crash数据集**:
- lambda_align: 0.001
- lambda_sparse: 0.01
- batch_size: 16
- learning_rate: 1e-3
- weight_decay: 1e-5
- h_dim: 512, z_dim: 256
- n_epochs: 25
- 预期AP: 0.9977

**A3D数据集**:
- lambda_align: 0.001
- lambda_sparse: 0.01
- batch_size: 16
- learning_rate: 1e-3
- weight_decay: 1e-5
- h_dim: 512, z_dim: 256
- n_epochs: 25
- 预期AP: 0.9403

## 部署信息

- **GPU**: 3 (NVIDIA GeForce RTX 4090)
- **输出目录**: output_gru/best_params_gpu3
- **日志目录**: logs/best_params_gpu3
- **训练脚本**: train_best_params.py
- **启动脚本**: run_best_params_gpu3.sh

## 监控命令

```bash
# 查看训练日志
tail -f logs/best_params_gpu3/train_*.log

# 监控GPU使用
nvidia-smi -i 3 -l 1

# 检查进程
ps aux | grep train_best_params
```

## 注意事项

1. **数据集特征**:
   - DAD: 使用res101特征 (2048维)
   - Crash: 使用vgg16特征 (4096维) - 因为没有res101
   - A3D: 使用vgg16特征 (4096维) - 因为没有res101

2. **数据量**:
   - DAD训练集: 128样本（可能是子集）
   - 如果数据量不对，需要检查数据路径

3. **训练时间**:
   - 每个数据集25个epoch
   - 预计每个数据集需要数小时

