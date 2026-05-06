# GPU 6 并行训练状态

## ✅ 部署成功

三个数据集已并行启动在GPU 6上：

### 进程信息
- **DAD**: PID 1643228
- **Crash**: PID 1643229  
- **A3D**: PID 1643231

### 日志文件
- `logs/best_params_gpu6/train_dad_*.log`
- `logs/best_params_gpu6/train_crash_*.log`
- `logs/best_params_gpu6/train_a3d_*.log`

### 输出目录
- `output_gru/best_params_gpu6/`

## 监控

```bash
# 实时查看所有日志
tail -f logs/best_params_gpu6/train_*.log

# 查看特定数据集
tail -f logs/best_params_gpu6/train_dad_*.log
tail -f logs/best_params_gpu6/train_crash_*.log
tail -f logs/best_params_gpu6/train_a3d_*.log

# GPU状态
nvidia-smi -i 6 -l 1

# 进程状态
ps aux | grep train_best_params | grep -v grep
```

## 训练参数

所有数据集使用相同的最佳参数：
- Learning Rate: 0.001
- Batch Size: 16
- Epochs: 25
- Lambda Align: 0.001
- Lambda Sparse: 0.01
- Weight Decay: 1e-5
- h_dim: 512, z_dim: 256

