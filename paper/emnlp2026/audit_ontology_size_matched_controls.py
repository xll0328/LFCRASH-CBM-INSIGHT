#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
OUT = ROOT / 'output' / 'emnlp2026_support'
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ['dad', 'a3d']
CONDS = [
    ('historical_stratified_30', 30),
    ('risk_core_v1', 30),
    ('historical_stratified_80', 80),
    ('perfect_v1', 80),
]
SEEDS = [42, 123, 3407]

rows = []
status = {}
for ds in DATASETS:
    status[ds] = {}
    for cond, k in CONDS:
        done = 0
        ap_vals = []
        tta_vals = []
        missing = []
        for s in SEEDS:
            tag = f'{ds}_sizectrl_{cond}_s{s}'
            p = ROOT / 'output' / f'{ds}_ac' / tag / 'results.json'
            if not p.exists():
                missing.append(s)
                continue
            try:
                data = json.loads(p.read_text())
            except Exception:
                missing.append(s)
                continue
            done += 1
            if isinstance(data.get('AP'), (int, float)):
                ap_vals.append(float(data['AP']) * 100.0)
            if isinstance(data.get('mTTA'), (int, float)):
                tta_vals.append(float(data['mTTA']))

        ap_mean = sum(ap_vals) / len(ap_vals) if ap_vals else None
        tta_mean = sum(tta_vals) / len(tta_vals) if tta_vals else None

        status[ds][cond] = {
            'concept_count': k,
            'done': done,
            'total': len(SEEDS),
            'ap_mean_percent': ap_mean,
            'mtta_mean_sec': tta_mean,
            'missing_seeds': missing,
        }
        rows.append((ds, cond, k, done, len(SEEDS), ap_mean, tta_mean, missing))

json_path = OUT / 'ontology_size_matched_status.json'
md_path = OUT / 'ontology_size_matched_status.md'
json_path.write_text(json.dumps(status, indent=2), encoding='utf-8')

lines = [
    '# Ontology Size-Matched Control Status',
    '',
    '- Seeds tracked: `42, 123, 3407`',
    '',
    '| Dataset | Condition | #Concepts | Seeds done | AP mean | mTTA mean | Missing |',
    '|---|---|---:|---:|---:|---:|---|',
]

for ds, cond, k, done, total, ap, tta, missing in rows:
    ap_s = '--' if ap is None else f'{ap:.2f}%'
    tta_s = '--' if tta is None else f'{tta:.2f}s'
    miss_s = '--' if not missing else ', '.join(str(x) for x in missing)
    lines.append(f'| {ds} | {cond} | {k} | {done}/{total} | {ap_s} | {tta_s} | {miss_s} |')

md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'[wrote] {json_path}')
print(f'[wrote] {md_path}')
