# LFCRASH-CBM Phase 1: 代码完善与稳定性

## 目标
确保 Concept-Gated CRASH 架构完全稳定、可复现，为后续实验奠定基础。

## 任务清单

### 1.1 核心架构优化 ✓
- [x] 创建 train_enhanced.py 脚本
  - 完整的日志记录系统
  - NaN/Inf 自动检测和恢复
  - 梯度范数监控
  - 学习率跟踪
  - Checkpoint 管理

### 1.2 数据加载与预处理
- [ ] 验证 TOA（Time of Accident）标注的正确性
  - 检查三个数据集的 TOA 分布
  - 确保 TOA 在有效范围内
  - 验证 TOA 与标签的一致性

- [ ] 实现数据增强
  - Temporal jittering: 随机时间偏移
  - Frame dropping: 随机帧丢弃
  - Noise injection: 高斯噪声注入

- [ ] 添加数据统计信息
  - 类别平衡分析
  - 特征分布统计
  - 样本长度分布

- [ ] 创建标准化的数据分割
  - 确保 train/val/test 分割一致
  - 记录分割信息到 JSON

### 1.3 评估指标完善
- [ ] 实现 AP (Average Precision) 计算
  - 支持不同的 IoU 阈值
  - 计算 mAP@IoU

- [ ] 实现 AUC-ROC, Precision, Recall, F1-Score
  - 逐帧评估
  - 整体评估

- [ ] 实现 mAP@IoU 指标（用于时序定位）
  - IoU 计算
  - 阈值扫描

- [ ] 添加置信度校准分析（ECE, MCE）
  - Expected Calibration Error
  - Maximum Calibration Error

## 执行步骤

### Step 1: 验证数据完整性
```bash
cd /data/sony/LFCRASH/LFCRASH-CBM
python -c "
from src.data_loader import DADDataset, CrashDataset, A3DDataset
from pathlib import Path

DATA_ROOT = Path('../CRASH/data')

for ds_name, DS in [('dad', DADDataset), ('crash', CrashDataset), ('a3d', A3DDataset)]:
    ds = DS(str(DATA_ROOT / ds_name), 'vgg16', phase='train', toTensor=False)
    print(f'{ds_name}: {len(ds)} samples')
    for i in range(min(3, len(ds))):
        x, y, toa = ds[i]
        print(f'  Sample {i}: x.shape={x.shape}, y={y}, toa={toa}')
"
```

### Step 2: 运行增强训练脚本（单个数据集）
```bash
# 测试 CRASH 数据集
python train_enhanced.py \
  --dataset crash \
  --epochs 5 \
  --batch_size 16 \
  --gpu 0 \
  --output_dir output/phase1_test \
  --tag crash_test_v1

# 检查日志
tail -f output/phase1_test/crash_test_v1/train.log
```

### Step 3: 验证日志输出
检查以下内容：
- ✓ 每个 epoch 的 loss 输出
- ✓ 梯度范数是否正常
- ✓ 学习率是否按计划变化
- ✓ 是否有 NaN/Inf 警告
- ✓ Checkpoint 是否正确保存

### Step 4: 运行三个数据集的基准训练
```bash
# 并行运行三个数据集
for ds in dad crash a3d; do
  python train_enhanced.py \
    --dataset $ds \
    --epochs 10 \
    --batch_size 16 \
    --gpu $((RANDOM % 8)) \
    --output_dir output/phase1_baseline \
    --tag ${ds}_baseline_v1 &
done
wait
```

### Step 5: 分析结果
```bash
# 收集所有结果
python -c "
import json
from pathlib import Path

results_dir = Path('output/phase1_baseline')
for result_file in results_dir.glob('*/results.json'):
    with open(result_file) as f:
        data = json.load(f)
    print(f'{result_file.parent.name}: AP={data[\"best_ap\"]:.4f}, epoch={data[\"best_epoch\"]}')
"
```

## 成功标准

- ✓ 代码无 NaN/Inf 问题（nan_count=0, inf_count=0）
- ✓ 三个数据集都能正常训练
- ✓ 日志清晰记录所有关键信息
- ✓ Checkpoint 正确保存和加载
- ✓ 评估指标正确计算
- ✓ 训练曲线平滑（无异常波动）

## 预期时间
- 数据验证: 1 小时
- 单数据集测试: 2 小时
- 三数据集基准训练: 12 小时
- 结果分析: 1 小时
- **总计: 16 小时**

## 下一步
完成 Phase 1 后，进入 Phase 2: 系统实验与基准测试
- 训练 Baseline CRASH（无概念层）
- 训练 Baseline GRU（无 CRASH 模块）
- 训练完整的 CG-CRASH（所有模块）
- 执行完整消融实验

