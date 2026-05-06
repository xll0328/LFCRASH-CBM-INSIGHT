# GPU 6 实验部署状态

## 部署时间
2026-01-08 13:38:24

## 实验配置

### 完整训练实验（使用各自最佳参数）
- **DAD数据集**: 25个epoch，使用最佳参数
- **Crash数据集**: 25个epoch，使用最佳参数  
- **A3D数据集**: 25个epoch，使用最佳参数

### Optuna搜参实验（并行运行）
- **DAD数据集**: 20个trials，每个trial 15个epoch
- **Crash数据集**: 20个trials，每个trial 15个epoch
- **A3D数据集**: 20个trials，每个trial 15个epoch

## 进程信息

### 完整训练进程
- DAD: PID 819731
- Crash: PID 819732
- A3D: PID 819733

### Optuna搜参进程
- DAD: PID 819734
- Crash: PID 819735
- A3D: PID 819736

## 输出目录

### 完整训练结果
- 输出目录: `output_gru/full_training_gpu6/`
- 每个数据集会生成：
  - `best_model.pth` - 最佳模型
  - `results.json` - 评估结果
  - `training_log.json` - 训练日志

### Optuna搜参结果
- 输出目录: `optuna_studies/gpu6/`
- 每个数据集会生成：
  - `optuna_gpu6_{dataset}_{timestamp}_results.json` - 搜索结果
  - `logs/optuna_gpu6_{dataset}_{timestamp}.log` - 详细日志

## 日志文件

### 完整训练日志
- DAD: `logs/full_training_and_optuna_gpu6/train_dad_20260108_133824.log`
- Crash: `logs/full_training_and_optuna_gpu6/train_crash_20260108_133824.log`
- A3D: `logs/full_training_and_optuna_gpu6/train_a3d_20260108_133824.log`

### Optuna搜参日志
- DAD: `logs/full_training_and_optuna_gpu6/optuna_dad_20260108_133824.log`
- Crash: `logs/full_training_and_optuna_gpu6/optuna_crash_20260108_133824.log`
- A3D: `logs/full_training_and_optuna_gpu6/optuna_a3d_20260108_133824.log`

## 监控命令

### 查看GPU使用情况
```bash
nvidia-smi -i 6 -l 1
```

### 查看进程状态
```bash
ps aux | grep -E 'train_best_params|train_optuna' | grep -v grep
```

### 查看完整训练日志
```bash
tail -f logs/full_training_and_optuna_gpu6/train_dad_20260108_133824.log
tail -f logs/full_training_and_optuna_gpu6/train_crash_20260108_133824.log
tail -f logs/full_training_and_optuna_gpu6/train_a3d_20260108_133824.log
```

### 查看Optuna搜参日志
```bash
tail -f logs/full_training_and_optuna_gpu6/optuna_dad_20260108_133824.log
tail -f logs/full_training_and_optuna_gpu6/optuna_crash_20260108_133824.log
tail -f logs/full_training_and_optuna_gpu6/optuna_a3d_20260108_133824.log
```

## 预计完成时间

### 完整训练
- DAD: 约2-3小时（25个epoch）
- Crash: 约2-3小时（25个epoch）
- A3D: 约2-3小时（25个epoch）

### Optuna搜参
- 每个数据集: 约4-6小时（20个trials × 15个epoch）
- 三个数据集并行: 约4-6小时（取决于GPU资源分配）

## 注意事项

1. **GPU资源分配**: 6个进程同时运行在GPU 6上，GPU会自动分配资源
2. **内存使用**: 24GB GPU内存应该足够支持6个并行实验
3. **进程监控**: 建议定期检查进程状态和GPU利用率
4. **结果保存**: 所有结果会自动保存到指定目录

## 停止实验

如果需要停止所有实验：
```bash
pkill -f "train_best_params.py"
pkill -f "train_optuna_multi_dataset.py"
```




