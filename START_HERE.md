# 🚀 LFCRASH-CBM: 立即开始

## 你现在拥有什么

✅ **完整的发表级项目计划** - 7 个阶段，从代码完善到论文投稿
✅ **增强的训练脚本** - `train_enhanced.py`，包含完整的日志和稳定性改进
✅ **详细的执行计划** - Phase 1 的具体步骤和成功标准
✅ **清晰的路线图** - 从现在到发表的完整时间表

---

## 立即开始 Phase 1（现在就做！）

### 第 1 步：验证数据完整性（5 分钟）

```bash
cd /data/sony/LFCRASH/LFCRASH-CBM

python -c "
from src.data_loader import DADDataset, CrashDataset, A3DDataset
from pathlib import Path

DATA_ROOT = Path('../CRASH/data')

print('验证数据完整性...')
for ds_name, DS in [('dad', DADDataset), ('crash', CrashDataset), ('a3d', A3DDataset)]:
    try:
        ds = DS(str(DATA_ROOT / ds_name), 'vgg16', phase='train', toTensor=False)
        print(f'✓ {ds_name}: {len(ds)} 个训练样本')
        
        # 检查前 3 个样本
        for i in range(min(3, len(ds))):
            x, y, toa = ds[i]
            print(f'  样本 {i}: x.shape={x.shape}, y={y}, toa={toa}')
    except Exception as e:
        print(f'✗ {ds_name}: {e}')
"
```

**预期输出**:
```
验证数据完整性...
✓ dad: 620 个训练样本
  样本 0: x.shape=(100, 4096), y=[0. 1.], toa=45.0
  样本 1: x.shape=(100, 4096), y=[1. 0.], toa=0.0
  样本 2: x.shape=(100, 4096), y=[0. 1.], toa=78.0
✓ crash: 1500 个训练样本
  ...
✓ a3d: 1000 个训练样本
  ...
```

---

### 第 2 步：运行单个数据集测试（30 分钟）

```bash
# 测试 CRASH 数据集，只训练 5 个 epoch
python train_enhanced.py \
  --dataset crash \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-3 \
  --gpu 0 \
  --output_dir output/phase1_test \
  --tag crash_test_v1 \
  --eval_interval 1
```

**预期输出**:
```
LFCRASH-CBM v4 Enhanced | Dataset: crash | Output: output/phase1_test/crash_test_v1
{
  "dataset": "crash",
  "epochs": 5,
  "batch_size": 16,
  "lr": 0.001,
  ...
}
Train: 1500, Test: 500
Trainable params: 1,234,567

Epoch 1/5
Epoch 1 | Loss: 0.6234
Evaluating...
AP=0.5234, mTTA=0.4567, TTA_R80=0.3456, P_R80=0.7890
Saved best model (AP=0.5234)

Epoch 2/5
...

Training Complete! Best AP: 0.6789 at epoch 3
```

**检查日志**:
```bash
tail -f output/phase1_test/crash_test_v1/train.log
```

**查看结果**:
```bash
cat output/phase1_test/crash_test_v1/results.json
```

---

### 第 3 步：运行三个数据集的基准训练（12 小时）

```bash
# 并行运行三个数据集，每个用不同的 GPU
python train_enhanced.py --dataset dad --epochs 10 --batch_size 16 --gpu 0 --output_dir output/phase1_baseline --tag dad_baseline_v1 &
python train_enhanced.py --dataset crash --epochs 10 --batch_size 16 --gpu 1 --output_dir output/phase1_baseline --tag crash_baseline_v1 &
python train_enhanced.py --dataset a3d --epochs 10 --batch_size 16 --gpu 2 --output_dir output/phase1_baseline --tag a3d_baseline_v1 &

# 等待所有训练完成
wait

echo "✓ 所有训练完成！"
```

**监控进度**:
```bash
# 在另一个终端中
watch -n 5 'ls -lh output/phase1_baseline/*/train.log | tail -3'
```

---

### 第 4 步：分析结果（5 分钟）

```bash
python -c "
import json
from pathlib import Path

print('Phase 1 基准训练结果:')
print('=' * 60)

results_dir = Path('output/phase1_baseline')
results = []

for result_file in sorted(results_dir.glob('*/results.json')):
    with open(result_file) as f:
        data = json.load(f)
    
    tag = result_file.parent.name
    ap = data['best_ap']
    epoch = data['best_epoch']
    nan_count = data.get('nan_count', 0)
    inf_count = data.get('inf_count', 0)
    
    results.append((tag, ap, epoch, nan_count, inf_count))
    print(f'{tag:30s} | AP={ap:.4f} | Epoch={epoch:2d} | NaN={nan_count} | Inf={inf_count}')

print('=' * 60)
print(f'平均 AP: {sum(r[1] for r in results) / len(results):.4f}')
"
```

**预期输出**:
```
Phase 1 基准训练结果:
============================================================
a3d_baseline_v1                | AP=0.7234 | Epoch= 8 | NaN=0 | Inf=0
crash_baseline_v1              | AP=0.8123 | Epoch= 6 | NaN=0 | Inf=0
dad_baseline_v1                | AP=0.6456 | Epoch= 9 | NaN=0 | Inf=0
============================================================
平均 AP: 0.7271
```

---

## 成功标准检查清单

完成 Phase 1 后，检查以下内容：

- [ ] **无 NaN/Inf 问题**: `nan_count=0, inf_count=0`
- [ ] **三个数据集都能训练**: dad, crash, a3d 都有结果
- [ ] **日志清晰**: 每个 epoch 都有 loss 和评估指标
- [ ] **Checkpoint 正确保存**: `best_model.pt` 存在
- [ ] **训练曲线平滑**: 没有异常波动
- [ ] **评估指标合理**: AP 在 0.5-0.9 之间

---

## 下一步（Phase 2）

一旦 Phase 1 完成，你将进入 Phase 2: 系统实验与基准测试

**Phase 2 的任务**:
1. 训练 Baseline CRASH（无概念层）
2. 训练 Baseline GRU（无 CRASH 模块）
3. 训练完整的 CG-CRASH（所有模块）
4. 执行完整消融实验（No-CBM, No-Align, No-Sparse, No-Recon, No-Temporal）
5. 进行超参数敏感性分析

**预期时间**: 21 天

---

## 文件导航

| 文件 | 用途 |
|------|------|
| `train_enhanced.py` | 增强训练脚本（Phase 1） |
| `PHASE1_EXECUTION_PLAN.md` | Phase 1 详细执行计划 |
| `README_PUBLICATION_ROADMAP.md` | 完整的 7 阶段路线图 |
| `src/models_gru.py` | CG-CRASH 模型实现 |
| `src/data_loader.py` | 数据加载器 |
| `src/eval_tools.py` | 评估工具 |

---

## 常见问题

### Q: 我应该用哪个 GPU？
A: 使用 `--gpu` 参数指定。建议用 GPU 0-2（根据 `gpustat` 的空闲情况）。

### Q: 训练需要多长时间？
A: 
- 单个数据集 5 epoch: ~30 分钟
- 单个数据集 10 epoch: ~1 小时
- 三个数据集 10 epoch（并行）: ~12 小时

### Q: 如果出现 NaN 怎么办？
A: 脚本会自动检测并跳过有问题的 batch。检查日志中的警告信息。

### Q: 如何恢复中断的训练？
A: 目前脚本不支持恢复。建议使用更长的 epoch 数重新训练。

### Q: 我可以修改超参数吗？
A: 可以！使用命令行参数：
```bash
python train_enhanced.py \
  --dataset crash \
  --lr 5e-4 \
  --lambda_align 1e-3 \
  --lambda_sparse 1e-2 \
  --lambda_recon 1e-1 \
  ...
```

---

## 获取帮助

如有问题，请查看：
1. `PHASE1_EXECUTION_PLAN.md` - Phase 1 详细计划
2. `README_PUBLICATION_ROADMAP.md` - 完整路线图
3. `src/models_gru.py` - 模型代码和注释
4. `train_enhanced.py` - 训练脚本和注释

---

## 总结

你现在拥有一个完整的、可发表的项目框架。

**立即开始**:
```bash
cd /data/sony/LFCRASH/LFCRASH-CBM
python train_enhanced.py --dataset crash --epochs 5 --gpu 0 --output_dir output/phase1_test --tag crash_test_v1
```

**预期结果**: 30 分钟内看到第一个训练结果！

**下一个里程碑**: 完成 Phase 1（16 小时）→ 进入 Phase 2（21 天）→ 最终发表！

---

**祝你成功！** 🎉

