#!/usr/bin/env python3
"""
watch_and_update.py
===================
自动监控 DAD 训练进度，当出现新 best AP 时:
1. 更新 results summary
2. 重新生成 Fig1 + Fig2
3. 打印最新论文数字

Usage:
  python watch_and_update.py  # runs every 5 min
"""
import time, json, subprocess
from pathlib import Path

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
LOGS = {
    'curriculum': ROOT / 'output' / 'dad_curriculum_v1.log',
    'finetune':   ROOT / 'output' / 'dad_finetune_z256.log',
}
BEST_RESULTS = ROOT / 'output' / 'best_dad_result.json'


def parse_best_ap(log_path):
    """Extract best AP from training log."""
    best_ap = 0.0
    try:
        for line in open(log_path):
            if 'EVAL' in line and 'AP=' in line:
                import re
                m = re.search(r'AP=(\d+\.\d+)', line)
                if m:
                    ap = float(m.group(1))
                    if ap > best_ap:
                        best_ap = ap
            if '*** New best' in line:
                import re
                m = re.search(r'AP=(\d+\.\d+)', line)
                if m:
                    best_ap = max(best_ap, float(m.group(1)))
    except: pass
    return best_ap


def get_current_best():
    try:
        return json.load(open(BEST_RESULTS)).get('AP', 0)
    except: return 0.0


def update_figures():
    print('  Regenerating figures...')
    subprocess.run(
        ['python3', 'visualize_paper_figures.py', '--skip_model'],
        cwd=ROOT, capture_output=True, timeout=120
    )
    print('  Figures updated.')


def print_summary():
    print('\n=== CURRENT BEST RESULTS ===')
    for ds, path in [
        ('CRASH', 'output/sota_push/crash_sota/results.json'),
        ('A3D',   'output/phase2_ablation/a3d_full/results.json'),
        ('DAD',   'output/dad_sota_push/dad_z512/results.json'),
    ]:
        try:
            d = json.load(open(ROOT/path))
            print(f'  {ds}: AP={d["AP"]*100:.2f}% mTTA={d["mTTA"]:.3f}s TTA@R80={d["TTA_R80"]:.3f}s')
        except:
            print(f'  {ds}: N/A')


def main():
    print('Watching DAD training logs...')
    known_best = get_current_best()
    print(f'Current best DAD AP: {known_best*100:.2f}%')

    while True:
        for tag, log in LOGS.items():
            if not log.exists(): continue
            ap = parse_best_ap(log)
            if ap > known_best + 0.001:  # 0.1% improvement threshold
                print(f'\n*** NEW BEST DAD AP: {ap*100:.2f}% (from {tag}) ***')
                known_best = ap
                # Save
                json.dump({'AP': ap, 'source': tag},
                          open(BEST_RESULTS, 'w'), indent=2)
                update_figures()
                print_summary()
        time.sleep(300)  # Check every 5 minutes


if __name__ == '__main__':
    main()
