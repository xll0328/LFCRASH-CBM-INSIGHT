# 实验进度和状态报告

**生成时间**: 2026-01-09 05:30

## 📊 总体状态

### ✅ 已完成实验

#### 1. DAD数据集 - 完整训练
- **状态**: ✅ 已完成
- **结果文件**: `output_gru/full_training_gpu6/best_dad/results.json`
- **最佳Epoch**: 5
- **性能指标**:
  - AP: 0.3489 (34.89%)
  - mTTA: 5.0
  - TTA@R80: 5.0
  - P@R80: 0.3489
- **参数**: λ_align=0.001, λ_sparse=0.01, batch_size=16, lr=0.001, h_dim=512, z_dim=256

#### 2. DAD数据集 - Optuna搜参
- **状态**: ✅ 已完成（20个trials全部完成）
- **结果文件**: `optuna_studies/gpu6/optuna_gpu6_dad_*_results.json`
- **最佳性能**: AP = 0.6511 (65.11%) ⭐ **显著优于完整训练**
- **最佳参数**:
  - λ_align: 3.50e-05
  - λ_sparse: 3.45e-04
  - batch_size: 8
  - learning_rate: 4.44e-04
  - weight_decay: 4.90e-05
  - h_dim: 512
  - z_dim: 128
- **完成情况**: 20个trials完成，0个剪枝，0个失败

### 🔄 进行中的实验

#### 3. Crash数据集 - Optuna搜参
- **状态**: 🔄 运行中
- **当前进度**: Trial 12/20, Epoch 2/15
- **运行时间**: 约224小时（9.3天）
- **日志文件**: `logs/full_training_and_optuna_gpu6/optuna_crash_fixed_*.log`
- **预计完成时间**: 还需约8个trials，每个trial约15个epoch

### ❌ 遇到错误的实验

#### 4. Crash数据集 - 完整训练
- **状态**: ❌ 训练完成但测试时出错
- **错误**: `IndexError: index 0 is out of bounds for axis 0 with size 0`
- **错误位置**: `eval_tools.py` line 135, `P_R80 = new_Precision[a[0][0]]`
- **原因**: Recall未达到0.8，导致无法计算P@R80
- **说明**: 这是模型性能问题，不是代码bug（符合CRASH原始逻辑）

#### 5. A3D数据集 - 完整训练
- **状态**: ❌ 训练完成但测试时出错
- **错误**: `IndexError: index 0 is out of bounds for axis 0 with size 0`
- **错误位置**: `eval_tools.py` line 135, `P_R80 = new_Precision[a[0][0]]`
- **原因**: Recall未达到0.8，导致无法计算P@R80
- **说明**: 这是模型性能问题，不是代码bug（符合CRASH原始逻辑）

#### 6. A3D数据集 - Optuna搜参
- **状态**: ❌ Trial 0失败
- **错误**: `IndexError: index 478 is out of bounds for axis 0 with size 478`
- **错误位置**: `train_optuna_multi_dataset.py` line 568
- **原因**: 数组索引越界问题
- **日志文件**: `logs/full_training_and_optuna_gpu6/optuna_a3d_fixed_*.log`

## 📈 性能对比

| 数据集 | 完整训练AP | Optuna最佳AP | 提升 |
|--------|-----------|-------------|------|
| DAD    | 0.3489 (34.89%) | 0.6511 (65.11%) | +30.22% ⭐ |
| Crash  | 错误（recall<0.8） | 进行中 | - |
| A3D    | 错误（recall<0.8） | 错误 | - |

## 🔍 问题分析

### IndexError问题（P@R80）
- **现象**: Crash和A3D的完整训练在evaluation时出现IndexError
- **原因**: 模型性能较差，recall未达到0.8，无法计算P@R80
- **处理**: 这是CRASH原始代码的预期行为，表示模型性能不足
- **建议**: 
  1. 检查训练过程是否正常
  2. 考虑调整超参数或模型架构
  3. 或者接受这个结果（表示模型在该数据集上表现不佳）

### A3D Optuna IndexError
- **现象**: `index 478 is out of bounds for axis 0 with size 478`
- **原因**: 数组索引问题，可能是数据加载或处理时的bug
- **需要**: 检查`train_optuna_multi_dataset.py` line 568附近的代码

## 💻 GPU资源使用

- **GPU**: NVIDIA GeForce RTX 4090 (24GB)
- **当前内存**: 4.3GB / 24GB (17%)
- **利用率**: 0%（可能在数据加载阶段）

## 📝 建议行动

1. **继续监控Crash Optuna搜参**：预计还需要较长时间完成
2. **检查A3D Optuna错误**：需要修复索引越界问题
3. **分析Crash和A3D完整训练失败原因**：可能是超参数不适合，或者需要更多训练
4. **考虑重新运行失败的实验**：使用Optuna找到的最佳参数

## 📁 关键文件位置

- **完整训练结果**: `output_gru/full_training_gpu6/`
- **Optuna结果**: `optuna_studies/gpu6/`
- **日志文件**: `logs/full_training_and_optuna_gpu6/`
- **汇总结果**: `output_gru/full_training_gpu6/all_results.json`



