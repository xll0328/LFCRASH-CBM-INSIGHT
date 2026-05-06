# CG-CRASH 过夜任务状态汇总

最后更新：2026-03-18 14:00 UTC

## 训练任务

| 任务 | GPU | 状态 | 进度 |
|------|-----|------|------|
| DAD 200ep (dad_longer) | 0 | 运行中 | Epoch ~40/200 |
| crash_frac25 | — | 已完成 | AP=98.6%, mTTA=3.72s ✓ |
| crash_frac50 | — | 已完成 | AP=99.2%, mTTA=4.27s ✓ |
| crash_frac75 | 5 | 运行中 | Epoch 3/80 (预计明早完成) |
| crash_frac100 | 5 | 排队中 | (frac75完成后自动启动) |
| dad_frac25 | — | 已完成 | AP=58.3%, mTTA=2.08s ✓ |
| dad_frac50 | — | **需要重跑** | AP=44.7% (训练中断，结果不可靠) |
| dad_frac75 | 6 | 运行中 | Epoch ~3/80 |
| dad_frac100 | 6 | 排队中 | (frac75完成后自动启动) |
| a3d_frac25 | — | 已完成 | AP=92.1%, mTTA=4.76s ✓ |
| a3d_frac50 | — | 已完成 | AP=93.8%, mTTA=4.47s ✓ |
| a3d_frac75 | — | 已完成 | AP=92.8%, mTTA=3.68s ✓ |
| a3d_frac100 | 5 | 排队中 | (crash_frac75/100完成后启动) |

## 可视化修复（今日完成）

- ✅ GIF concept 名重叠 → yticks [:22]
- ✅ best_case_study 空白/重叠 → figsize=(22,10), hspace=0.18, legend upper left
- ✅ multi_case 帧太小 → figsize=36, 帧420×260, 帧占宽73%
- ✅ paper_strip 概念名重复 → 只第一列显示
- ✅ timeline_concepts 空白 → figsize=(14,17), hspace=0.40
- ✅ HTML 布局重叠 → 完全重写为4行1列，无重叠
- ✅ HTML 概念名截断 → margin l=220 完整显示

## 数据效率实验当前结果

| Dataset | 25% | 50% | 75% | 100% |
|---------|-----|-----|-----|------|
| CCD | 98.6% | 99.2% | running | running |
| DAD | 58.3% | **需重跑** | running | running |
| A3D | 92.1% | 93.8% | 92.8% | running |

**注意**: dad_frac50 训练在 06:45 被 DataLoader 异常中断，AP=44.7% 不可信，需重跑。

## 明早待做

1. 检查数据效率实验是否完成：
   ```bash
   python3 /data/sony/LFCRASH/LFCRASH-CBM/output/summarize_data_efficiency.py
   ```

2. 重跑 dad_frac50（用 num_workers=0）：
   ```bash
   python3 /data/sony/LFCRASH/LFCRASH-CBM/train_enhanced.py \
     --dataset dad --gpu 5 --epochs 80 --batch_size 16 \
     --lr 0.0003 --weight_decay 3e-5 \
     --h_dim 256 --z_dim 128 \
     --lambda_align 0.0001 --lambda_sparse 0.0005 --lambda_recon 0.005 \
     --num_concepts 837 --num_workers 0 --eval_interval 5 \
     --train_fraction 0.50 \
     --output_dir output/data_efficiency --tag dad_frac50
   ```

3. 绘制数据效率曲线图（AP vs 训练数据比例）。
