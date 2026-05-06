# LFCRASH-CBM: 发表级项目完整路线图

## 项目概述

**目标**: 将 LFCRASH-CBM 打造成可发表在 **ICCV/ECCV/IJCAI** 的完整项目

**核心创新**: Concept-Gated CRASH (CG-CRASH) - 一个可解释的概念瓶颈架构，用于事故预测

**关键特性**:
- ✓ 可解释性：通过概念激活理解模型决策
- ✓ 高准确性：三个数据集上显著优于 Baseline
- ✓ 理论支持：信息论和泛化误差上界
- ✓ 完整实验：消融、超参数敏感性、跨数据集泛化

---

## 7 阶段执行计划

### 📋 Phase 1: 代码完善与稳定性（1-2 周）
**状态**: 🔄 进行中

**已完成**:
- ✅ 创建 `train_enhanced.py` 脚本
  - 完整的日志记录系统
  - NaN/Inf 自动检测和恢复
  - 梯度范数监控
  - Checkpoint 管理

**待完成**:
- [ ] 验证数据完整性（TOA 标注、分布、一致性）
- [ ] 实现数据增强（temporal jittering, frame dropping, noise injection）
- [ ] 添加数据统计信息（类别平衡、特征分布）
- [ ] 完善评估指标（AP, AUC-ROC, Precision, Recall, F1, ECE, MCE）

**执行命令**:
```bash
cd /data/sony/LFCRASH/LFCRASH-CBM

# Step 1: 验证数据
python -c "from src.data_loader import *; print('Data OK')"

# Step 2: 测试单个数据集
python train_enhanced.py --dataset crash --epochs 5 --gpu 0 --output_dir output/phase1_test

# Step 3: 运行三个数据集基准训练
for ds in dad crash a3d; do
  python train_enhanced.py --dataset $ds --epochs 10 --gpu $((RANDOM % 8)) --output_dir output/phase1_baseline --tag ${ds}_baseline_v1 &
done
wait

# Step 4: 分析结果
python -c "import json; from pathlib import Path; [print(f'{f.parent.name}: AP={json.load(open(f))[\"best_ap\"]:.4f}') for f in Path('output/phase1_baseline').glob('*/results.json')]"
```

**成功标准**:
- ✓ 无 NaN/Inf 问题（nan_count=0, inf_count=0）
- ✓ 三个数据集都能正常训练
- ✓ 日志清晰记录所有关键信息
- ✓ 训练曲线平滑

**预期时间**: 16 小时

---

### 🧪 Phase 2: 系统实验与基准测试（2-3 周）
**状态**: ⏳ 待开始

**任务**:
1. **基准模型训练**
   - Baseline CRASH（无概念层）
   - Baseline GRU（无 CRASH 模块）
   - 完整的 CG-CRASH（所有模块）

2. **完整消融实验**
   - No-CBM: 移除概念瓶颈
   - No-Align: 移除 CLIP 对齐损失
   - No-Sparse: 移除稀疏性约束
   - No-Recon: 移除重建损失
   - No-Temporal: 移除时序注意力

3. **超参数敏感性分析**
   - λ_align ∈ {0.01, 0.1, 1.0, 10.0}
   - λ_sparse ∈ {0.001, 0.01, 0.1, 1.0}
   - λ_recon ∈ {0.001, 0.01, 0.1, 1.0}
   - num_concepts ∈ {16, 32, 64, 128}
   - learning_rate ∈ {1e-4, 5e-4, 1e-3, 5e-3}

4. **跨数据集泛化性测试**
   - DAD → CRASH/A3D
   - CRASH → DAD/A3D
   - A3D → DAD/CRASH

**预期时间**: 21 天

---

### 🎨 Phase 3: 可解释性分析与可视化（2 周）
**状态**: ⏳ 待开始

**任务**:
1. **概念激活分析**
   - 提取概念激活向量
   - 计算稀疏性和多样性
   - 识别 Top-K 最重要的概念
   - 分析概念的时序动态

2. **概念可视化**
   - 概念激活热力图
   - 概念重要性排序（Bar chart）
   - 时序曲线（关键概念在事故前的激活模式）
   - t-SNE 投影（概念空间结构）
   - 注意力可视化

3. **案例研究（Case Study）**
   - 选择 5-10 个代表性样本
   - 生成原始视频帧序列
   - 绘制概念激活时序曲线
   - 可视化关键概念
   - 撰写详细分析文字

4. **对比分析**
   - 预测准确性（AP, AUC）
   - 预测的可解释性
   - 预测的稳定性
   - 计算效率

**预期时间**: 14 天

---

### 📐 Phase 4: 理论分析与证明（1-2 周）
**状态**: ⏳ 待开始

**任务**:
1. **理论框架建立**
   - 定义"可解释性"的形式化定义
   - 证明概念瓶颈的信息论性质
   - 分析概念激活的稀疏性与可解释性的关系
   - 推导泛化误差上界

2. **定理与引理**
   - **Theorem 1**: 概念瓶颈的信息压缩性质
   - **Theorem 2**: 概念对齐的有效性
   - **Lemma 1**: 时序注意力的收敛性

3. **实验验证理论**
   - 通过实验验证定理
   - 绘制理论预测 vs 实验结果的对比图
   - 讨论理论与实验的一致性

**预期时间**: 10 天

---

### 📝 Phase 5: 论文写作（2-3 周）
**状态**: ⏳ 待开始

**论文结构** (8-10 页):
1. **Introduction** (1.5 页)
   - 问题陈述：事故预测的可解释性需求
   - 现有方法的局限性
   - 本文的创新点

2. **Related Work** (1.5 页)
   - 事故预测方法
   - 可解释 AI / 概念瓶颈模型
   - 时序模型的可解释性

3. **Method** (2.5 页)
   - Concept-Gated CRASH 架构
   - 概念对齐与稀疏性约束
   - 时序注意力机制
   - 损失函数设计

4. **Experiments** (3 页)
   - 数据集与实验设置
   - 基准模型与消融实验
   - 定量结果（表格）
   - 定性分析（可视化）

5. **Results & Analysis** (2 页)
   - 性能对比
   - 可解释性分析
   - 案例研究
   - 理论验证

6. **Discussion** (1 页)
   - 主要发现
   - 局限性
   - 未来工作

7. **Conclusion** (0.5 页)

**关键图表与表格**:
- Table 1: 三个数据集上的性能对比（AP, AUC, F1）
- Table 2: 消融实验结果
- Table 3: 超参数敏感性分析
- Figure 1: CG-CRASH 架构图
- Figure 2: 概念激活热力图（3-5 个案例）
- Figure 3: Top-20 概念的重要性排序
- Figure 4: 时序曲线对比（预测 vs 真实）
- Figure 5: t-SNE 投影与概念空间结构
- Figure 6: 消融实验的性能变化

**预期时间**: 21 天

---

### 💻 Phase 6: 代码发布与文档（1 周）
**状态**: ⏳ 待开始

**任务**:
1. **代码整理**
   - 清理代码，移除调试代码
   - 添加详细的代码注释
   - 统一代码风格（PEP 8）
   - 创建 requirements.txt 和 setup.py

2. **文档编写**
   - README.md：项目概述、快速开始
   - INSTALL.md：安装指南
   - USAGE.md：使用教程
   - API.md：API 文档
   - RESULTS.md：复现结果的步骤

3. **开源发布**
   - 在 GitHub 上发布代码
   - 上传预训练模型
   - 提供数据集处理脚本
   - 提供可视化工具

**预期时间**: 7 天

---

### 🎯 Phase 7: 投稿与修改（持续）
**状态**: ⏳ 待开始

**任务**:
1. **目标会议/期刊**
   - 一级目标：ICCV, ECCV, IJCAI, AAAI
   - 二级目标：CVPR, NeurIPS, ICML
   - 三级目标：TPAMI, IJCV, IEEE TMM

2. **投稿准备**
   - 选择目标会议
   - 准备投稿版本（遵循格式要求）
   - 准备补充材料（代码、数据、视频）
   - 撰写投稿信

3. **审稿回复**
   - 收集审稿意见
   - 进行必要的实验和修改
   - 撰写详细的回复信
   - 提交修改版本

**预期时间**: 持续

---

## 时间表与里程碑

| 阶段 | 任务 | 预计时间 | 截止日期 |
|------|------|---------|---------|
| 1 | 代码完善 | 1-2 周 | 3 月 22 日 |
| 2 | 系统实验 | 2-3 周 | 4 月 5 日 |
| 3 | 可视化分析 | 2 周 | 4 月 19 日 |
| 4 | 理论分析 | 1-2 周 | 5 月 3 日 |
| 5 | 论文写作 | 2-3 周 | 5 月 24 日 |
| 6 | 代码发布 | 1 周 | 5 月 31 日 |
| 7 | 投稿 | 持续 | 6 月+ |

---

## 成功标准

- ✅ 代码完全稳定，无 NaN/Inf 问题
- ✅ 三个数据集上的性能显著优于 Baseline（>10% AP 提升）
- ✅ 消融实验清晰展示每个模块的贡献
- ✅ 可视化和案例研究充分展示可解释性
- ✅ 理论分析支持实验发现
- ✅ 论文清晰、完整、可发表
- ✅ 代码开源、可复现
- ✅ 投稿至顶级会议并获得接受

---

## 关键成功因素

1. **稳定性**: 确保所有实验可复现，无随机性问题
2. **完整性**: 覆盖所有重要的实验和分析
3. **清晰性**: 论文和代码都要清晰易懂
4. **创新性**: 突出可解释性的创新贡献
5. **严谨性**: 理论和实验都要严谨
6. **美观性**: 图表和可视化要专业美观

---

## 快速开始

### 立即开始 Phase 1:

```bash
cd /data/sony/LFCRASH/LFCRASH-CBM

# 1. 验证数据
python -c "from src.data_loader import DADDataset, CrashDataset, A3DDataset; print('✓ Data OK')"

# 2. 运行单个数据集测试
python train_enhanced.py --dataset crash --epochs 5 --gpu 0 --output_dir output/phase1_test --tag crash_test

# 3. 检查日志
tail -f output/phase1_test/crash_test/train.log

# 4. 查看结果
cat output/phase1_test/crash_test/results.json
```

---

## 文件结构

```
LFCRASH-CBM/
├── train_enhanced.py              # 增强训练脚本（Phase 1）
├── PHASE1_EXECUTION_PLAN.md       # Phase 1 详细执行计划
├── README_PUBLICATION_ROADMAP.md  # 本文件
├── src/
│   ├── models_gru.py              # CG-CRASH 模型
│   ├── data_loader.py             # 数据加载器
│   ├── eval_tools.py              # 评估工具
│   ├── concept_utils.py           # 概念可视化工具
│   └── ablation.py                # 消融实验框架
├── output/
│   ├── phase1_test/               # Phase 1 测试结果
│   ├── phase1_baseline/           # Phase 1 基准结果
│   ├── phase2_ablation/           # Phase 2 消融实验
│   └── ...
└── paper/
    ├── main.tex                   # 论文主文件
    ├── figures/                   # 论文图表
    └── tables/                    # 论文表格
```

---

## 联系与支持

如有问题，请查看：
- Phase 1 详细计划: `PHASE1_EXECUTION_PLAN.md`
- 模型代码: `src/models_gru.py`
- 训练脚本: `train_enhanced.py`

