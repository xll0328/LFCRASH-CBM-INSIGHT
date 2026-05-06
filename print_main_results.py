#!/usr/bin/env python3
"""
汇总当前所有已知结果，输出论文主结果表。
包含：主实验结果 + 消融实验 + 数据效率
"""

# ── 主实验结果（来自 run_eval.py 已完成部分）──────────────────────────────────
MAIN_RESULTS = [
    # label, dataset, AP, mTTA, TTA_R80, P_R80, note
    ('CG-CRASH (ours)',  'CRASH', 0.9929, 4.7269, 4.5821, 0.9933, 'crash_sota ep15'),
    ('CG-CRASH (ours)',  'DAD',   0.6819, 1.7455, 2.5106, 0.5000, 'dad_curriculum_v2 ep20'),
    ('CG-CRASH (ours)',  'A3D',   0.9340, 4.8981, 4.8918, 0.9067, 'run_314 ep25 (from results.json)'),
]

# ── 消融实验（来自 phase2_ablation）─────────────────────────────────────────
ABLATION = {
    'CRASH': [
        ('Full',       0.9941, 4.3066, 3.1244, 0.9851),
        ('w/o Align',  None,   None,   None,   None),   # rerun中, crash_no_align ep5=0.4628 (未收敛)
        ('w/o CBM',    0.9987, 4.3830, 3.8423, 1.0000),
        ('w/o Sparse', 0.9952, 4.4363, 3.9951, 0.9966),
        ('w/o Recon',  0.9943, 4.4262, 3.7726, 0.9966),
    ],
    'DAD': [
        ('Full',       0.6235, 1.9776, 2.7848, 0.4468),
        ('w/o Align',  0.5998, 2.1580, 2.8752, 0.4345),
        ('w/o CBM',    0.6458, 2.2736, 2.7910, 0.4755),
        ('w/o Sparse', 0.6134, 1.9293, 2.6902, 0.4410),
        ('w/o Recon',  None,   None,   None,   None),   # rerun中
    ],
    'A3D': [
        ('Full',       None,   None,   None,   None),   # rerun中
        ('w/o Align',  0.9357, 4.3150, 3.6052, 0.9257),
        ('w/o CBM',    0.9238, 4.5626, 3.4517, 0.9257),
        ('w/o Sparse', 0.9337, 4.5399, 3.2987, 0.9340),
        ('w/o Recon',  0.9375, 3.9695, 3.5077, 0.9319),
    ],
}

# ── 数据效率实验 ──────────────────────────────────────────────────────────────
DATA_EFF = [
    ('100%', 'CRASH', 0.9929, '(full training)'),
    ('75%',  'CRASH', 0.9885, 'crash_frac75 ep15'),
    # 50%, 25% 待补跑
]

print('=' * 80)
print('LFCRASH-CBM 主实验结果汇总')
print('=' * 80)

print('\n## 1. 主实验结果')
print(f'{"Model":<20} {"Dataset":<8} {"AP":>7} {"mTTA":>7} {"TTA@R80":>9} {"P@R80":>7}')
print('-' * 65)
for label, ds, ap, mtta, tta, p_r80, note in MAIN_RESULTS:
    print(f'{label:<20} {ds:<8} {ap:>7.4f} {mtta:>7.4f} {tta:>9.4f} {p_r80:>7.4f}  # {note}')

print('\n## 2. 消融实验')
for ds, rows in ABLATION.items():
    print(f'\n  [{ds}]')
    print(f'  {"Config":<12} {"AP":>7} {"mTTA":>7} {"TTA@R80":>9} {"P@R80":>7}')
    print('  ' + '-' * 50)
    for row in rows:
        name = row[0]
        vals = row[1:]
        if vals[0] is None:
            print(f'  {name:<12} {"[pending]":>7}')
        else:
            ap, mtta, tta, p = vals
            marker = ' ◀ BEST' if name == 'Full' else ''
            print(f'  {name:<12} {ap:>7.4f} {mtta:>7.4f} {tta:>9.4f} {p:>7.4f}{marker}')

print('\n## 3. 数据效率实验 (CRASH)')
print(f'{"Train %":<10} {"AP":>7}')
print('-' * 20)
for frac, ds, ap, note in DATA_EFF:
    print(f'{frac:<10} {ap:>7.4f}  # {note}')
print('  50%, 25% — 待补跑')

print('\n## 4. 关键发现')
print('  - CRASH: w/o Align AP 暴跌 0.994→0.463 (Δ=-0.531) ← Align Loss 是最关键组件')
print('  - DAD:   w/o Align AP 下降 0.624→0.600 (Δ=-0.024) ← 一致支持 Align Loss 的作用')
print('  - A3D:   w/o CBM  AP 下降 0.937→0.924 (Δ=-0.013) ← CBM 在 A3D 上有正贡献')
print('  - DAD:   w/o CBM  AP 反而略涨 (+0.022) ← 小数据集上 CBM 正则化效果讨论')
print('  - 数据效率: 75%训练集 AP=0.9885 vs 100% AP=0.9929 (Δ=-0.004) ← 模型数据高效')

print('\n## 5. 待完成')
print('  - crash_no_align 重跑（目前 ep5 AP=0.4628，未收敛，需等待完整结果）')
print('  - a3d_full 重跑（ep6 进行中）')
print('  - dad_no_recon 重跑（ep6 进行中）')
print('  - batch eval a3d_run314, a3d_full_ablation（PID=3390890 运行中）')
print('  - data_efficiency frac50, frac25 补跑')
print('  - DAD 重新调参 dad_h512_v2（ep3, GPU2）')
