# Optuna 30 Trials 部署报告

**部署时间**: 2026-01-09 05:42

## ✅ 修复的Bug

### 1. A3D Optuna IndexError修复
- **问题**: `index 478 is out of bounds for axis 0 with size 478`
- **原因**: `compute_ap`函数使用了固定的`n_frames=100`，但不同数据集帧数不同
- **修复**: 
  - 添加`n_frames`参数，根据数据集实际帧数设置
  - 添加数组长度一致性检查，确保`all_pred`、`all_labels`、`time_of_accidents`长度一致
  - 在调用`compute_ap`时传入正确的`fps`和`n_frames`参数

### 2. 参数搜索空间优化
基于已有实验结果优化了三个数据集的搜索空间：

#### DAD数据集
- **已有最佳结果**: AP=0.6511
- **最佳参数**: λ_align=3.50e-05, λ_sparse=3.45e-04, batch_size=8, lr=4.44e-04, h_dim=512, z_dim=128
- **优化策略**: 围绕最佳参数缩小搜索范围，聚焦最优区域
  - λ_align: (1e-5, 1e-4) - 围绕3.50e-05
  - λ_sparse: (1e-4, 1e-3) - 围绕3.45e-04
  - batch_size: [8, 16] - 重点关注小batch
  - learning_rate: (2e-4, 8e-4) - 围绕4.44e-04
  - z_dim: [128, 256] - 重点关注小维度

#### Crash数据集
- **当前最佳结果**: AP=0.6667
- **优化策略**: 围绕当前最佳结果扩展搜索空间
  - λ_align: (1e-4, 2e-3) - 围绕0.0008附近
  - λ_sparse: (1e-3, 3e-2) - 围绕0.019附近
  - learning_rate: (5e-5, 5e-4) - 围绕0.0002附近

#### A3D数据集
- **优化策略**: 扩大搜索范围，探索更多可能性
  - λ_align: (1e-5, 1e-3)
  - λ_sparse: (1e-4, 2e-2)
  - learning_rate: (1e-5, 5e-4)

## 🚀 部署状态

### 进程信息
- **DAD Optuna**: PID 1897406 (30 trials)
- **Crash Optuna**: PID 1897407 (30 trials)
- **A3D Optuna**: PID 1897408 (30 trials)

### 配置
- **每个数据集**: 30个trials
- **每个trial**: 15个epoch
- **GPU**: 6 (NVIDIA GeForce RTX 4090)
- **输出目录**: `optuna_studies/gpu6_30trials/`
- **日志目录**: `logs/optuna_30trials_gpu6/`

## 📊 预计完成时间

- **每个trial**: 约1-2小时（取决于数据集和参数）
- **每个数据集**: 约30-60小时（30个trials）
- **三个数据集并行**: 约30-60小时（取决于GPU资源分配）

## 📝 监控命令

### 查看进程状态
```bash
ps aux | grep -E 'train_optuna' | grep -v grep
```

### 查看GPU状态
```bash
nvidia-smi -i 6 -l 1
```

### 查看日志
```bash
# DAD
tail -f logs/optuna_30trials_gpu6/optuna_dad_20260109_054247.log

# Crash
tail -f logs/optuna_30trials_gpu6/optuna_crash_20260109_054247.log

# A3D
tail -f logs/optuna_30trials_gpu6/optuna_a3d_20260109_054247.log
```

## 🛑 停止实验

如果需要停止所有实验：
```bash
pkill -f "train_optuna_multi_dataset.py"
```

## ✅ 部署完成确认

- [x] 修复A3D Optuna IndexError
- [x] 优化参数搜索空间（基于已有结果）
- [x] 终止旧实验进程
- [x] 部署三个数据集Optuna搜参（每个30个trials）
- [x] 验证进程正常运行

**所有实验已成功部署并正常运行！** 🎉



