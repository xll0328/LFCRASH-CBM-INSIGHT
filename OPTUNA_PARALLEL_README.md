# 多数据集并行Optuna超参数搜索

## 概述

本方案为三个数据集（DAD、CCD/crash、A3D）设计了不同的超参数搜索空间，并在GPU 3上并行运行，每个数据集20个trials。

## 文件说明

- `train_optuna_multi_dataset.py`: Optuna搜索主脚本，支持数据集特定的搜索空间
- `run_optuna_parallel_gpu3.sh`: 并行运行脚本，在GPU 3上同时运行三个数据集的搜索

## 搜索空间设计

### DAD数据集
- **目标**: 从52.29%提升到接近65.3%（CRASH原文）
- **策略**: 基于之前Optuna找到的最佳参数（64.37%），在较小范围内精细搜索
- **搜索空间**:
  - `lambda_align`: [1e-5, 5e-4] (log scale)
  - `lambda_sparse`: [1e-5, 5e-4] (log scale)
  - `batch_size`: [8, 16, 32]
  - `learning_rate`: [1e-5, 5e-4] (log scale)
  - `h_dim`: [256, 512, 768]
  - `z_dim`: [128, 256, 512]

### CCD (crash) 数据集
- **目标**: 保持或提升99.77%的性能
- **策略**: 已经表现很好，使用保守的搜索空间
- **搜索空间**:
  - `lambda_align`: [5e-4, 5e-3] (log scale)
  - `lambda_sparse`: [5e-3, 5e-2] (log scale)
  - `batch_size`: [16, 32]
  - `learning_rate`: [1e-4, 1e-3] (log scale)
  - `h_dim`: [512, 768]
  - `z_dim`: [256, 512]

### A3D数据集
- **目标**: 从94.03%提升到接近96.0%（CRASH原文）
- **策略**: 中等范围的搜索空间
- **搜索空间**:
  - `lambda_align`: [5e-4, 2e-3] (log scale)
  - `lambda_sparse`: [5e-3, 2e-2] (log scale)
  - `batch_size`: [8, 16, 32]
  - `learning_rate`: [5e-5, 5e-4] (log scale)
  - `h_dim`: [256, 512, 768]
  - `z_dim`: [128, 256, 512]

## 使用方法

### 1. 检查依赖

确保已安装：
- PyTorch
- Optuna
- 其他项目依赖

```bash
pip install optuna
```

### 2. 运行并行搜索

```bash
cd /data/sony/LFCRASH/LFCRASH-CBM
bash run_optuna_parallel_gpu3.sh
```

### 3. 监控进度

脚本会：
- 在后台启动3个数据集的搜索任务
- 每分钟显示GPU使用情况
- 自动汇总最终结果

查看实时日志：
```bash
# 主日志
tail -f optuna_studies/gpu3/parallel_search_*.log

# 各数据集日志
tail -f optuna_studies/gpu3/logs/optuna_dad_*.log
tail -f optuna_studies/gpu3/logs/optuna_crash_*.log
tail -f optuna_studies/gpu3/logs/optuna_a3d_*.log
```

### 4. 单独运行某个数据集

```bash
python3 train_optuna_multi_dataset.py \
    --dataset dad \
    --gpu_id 3 \
    --n_trials 20 \
    --n_epochs 15 \
    --num_workers 2
```

## GPU内存管理

- **每个实验**: 使用`num_workers=2`，batch_size根据搜索空间调整（8-32）
- **并行运行**: 3个实验同时运行，预计总内存使用约18-22GB（24GB GPU可承受）
- **优化措施**:
  - 使用`pin_memory=True`和`non_blocking=True`加速数据传输
  - 每个trial结束后清理GPU缓存
  - 使用`persistent_workers`减少进程创建开销

## 输出文件

搜索完成后，结果保存在：
- `optuna_studies/gpu3/optuna_<dataset>_<timestamp>_results.json`: 每个数据集的搜索结果
- `optuna_studies/gpu3/logs/optuna_<dataset>_<timestamp>.log`: 每个数据集的详细日志
- `optuna_studies/gpu3/parallel_search_<timestamp>.log`: 并行运行的主日志

## 预期时间

- **每个trial**: 约15-30分钟（取决于batch_size和数据集大小）
- **每个数据集20个trials**: 约5-10小时
- **三个数据集并行**: 约5-10小时（取决于最慢的数据集）

## 注意事项

1. **数据路径**: 确保数据集路径正确（在`train_optuna_multi_dataset.py`中调整`data_root`）
2. **概念文件**: 确保`000_all_concept_set.txt`文件路径正确
3. **导入路径**: 如果导入失败，检查`src/models_gru.py`, `src/data_loader.py`, `src/eval_tools.py`是否存在
4. **GPU内存**: 如果出现OOM，可以减少`batch_size`的搜索范围或`num_workers`

## 结果分析

搜索完成后，使用以下命令查看最佳结果：

```bash
cd /data/sony/LFCRASH/LFCRASH-CBM
python3 -c "
import json
import glob
for f in glob.glob('optuna_studies/gpu3/optuna_*_results.json'):
    with open(f) as file:
        data = json.load(file)
        print(f\"{data['dataset']}: AP={data['best_value']:.4f}, 参数={data['best_params']}\")
"
```




