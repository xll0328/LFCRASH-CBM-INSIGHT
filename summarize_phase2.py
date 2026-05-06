#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 结果汇总脚本
从 output/phase2_ablation/ 收集所有 results.json，生成对比表格
"""

import json
from pathlib import Path

OUTDIR = Path('/data/sony/LFCRASH/LFCRASH-CBM/output/phase2_ablation')

DATASETS   = ['crash', 'dad', 'a3d']
CONDITIONS = ['full', 'no_align', 'no_sparse', 'no_recon', 'no_cbm']
COND_LABELS = {
    'full':      'CG-CRASH (Full)',
    'no_align':  'w/o Align',
    'no_sparse': 'w/o Sparse',
    'no_recon':  'w/o Recon',
    'no_cbm':    'w/o CBM',
}

results = {}
for ds in DATASETS:
    results[ds] = {}
    for cond in CONDITIONS:
        tag = f'{ds}_{cond}'
        candidates = list(OUTDIR.glob(f'**/{tag}/results.json'))
        if not candidates:
            candidates = list(OUTDIR.glob(f'*{tag}*/results.json'))
        if candidates:
            with open(candidates[0]) as f:
                data = json.load(f)
            results[ds][cond] = {
                'AP':      data.get('AP', data.get('ap', None)),
                'mTTA':    data.get('mTTA', data.get('mtta', None)),
                'TTA_R80': data.get('TTA_R80', data.get('tta_r80', None)),
                'P_R80':   data.get('P_R80', data.get('p_r80', None)),
                'epoch':   data.get('best_epoch', '?'),
            }
        else:
            results[ds][cond] = None

print('\n' + '='*90)
print('Phase 2 消融实验结果汇总')
print('='*90)

for ds in DATASETS:
    print(f'\n【数据集: {ds.upper()}】')
    print(f"{'条件':<20}  {'AP':>8}  {'mTTA':>8}  {'TTA_R80':>9}  {'P_R80':>8}  {'Epoch':>6}")
    print('-' * 70)
    full_ap = results[ds].get('full', {}) or {}
    full_ap_val = full_ap.get('AP') if full_ap else None
    for cond in CONDITIONS:
        r = results[ds].get(cond)
        label = COND_LABELS[cond]
        if r is None:
            print(f"  {label:<20}  {'(running)':>8}")
        else:
            ap   = r['AP']
            mt   = r['mTTA']
            t80  = r['TTA_R80']
            p80  = r['P_R80']
            ep   = r['epoch']
            delta = ''
            if full_ap_val and cond != 'full' and ap is not None:
                delta = f"({ap - full_ap_val:+.4f})"
            print(f"  {label:<20}  {ap:.4f}{delta:>10}  {mt:.4f}  {t80:.4f}  {p80:.4f}  {ep:>6}")

print('\n' + '='*90)

# Save markdown
md_lines = ['# Phase 2 Ablation Results\n']
for ds in DATASETS:
    md_lines.append(f'\n## {ds.upper()}\n')
    md_lines.append('| Condition | AP | mTTA | TTA_R80 | P_R80 | Epoch |\n')
    md_lines.append('|-----------|-----|------|---------|-------|-------|\n')
    for cond in CONDITIONS:
        r = results[ds].get(cond)
        label = COND_LABELS[cond]
        if r is None:
            md_lines.append(f'| {label} | -- | -- | -- | -- | -- |\n')
        else:
            ap  = f"{r['AP']:.4f}"      if r['AP']      else '--'
            mt  = f"{r['mTTA']:.4f}"    if r['mTTA']    else '--'
            t80 = f"{r['TTA_R80']:.4f}" if r['TTA_R80'] else '--'
            p80 = f"{r['P_R80']:.4f}"   if r['P_R80']   else '--'
            ep  = str(r['epoch'])
            md_lines.append(f'| {label} | {ap} | {mt} | {t80} | {p80} | {ep} |\n')

md_path = Path('/data/sony/LFCRASH/LFCRASH-CBM/PHASE2_RESULTS.md')
with open(md_path, 'w') as f:
    f.writelines(md_lines)
print(f'Markdown saved: {md_path}')
