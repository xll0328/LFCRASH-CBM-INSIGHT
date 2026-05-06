#!/usr/bin/env python3
"""
monitor.py — 实时监控所有训练实验的进度
Usage: python monitor.py
"""
import os, json, time, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPS = [
    ('dad_ac_v1',           ROOT / 'output/dad_ac/dad_ac_v1'),
    ('dad_ac_distill_v1',   ROOT / 'output/dad_ac_distill/dad_ac_distill_v1'),
    ('dad_ac_v3_fixed_lr',  ROOT / 'output/dad_ac/dad_ac_v3_fixed_lr'),
    ('dad_ac_v4_final',     ROOT / 'output/dad_ac/dad_ac_v4_final'),
    ('a3d_ac_v1',           ROOT / 'output/a3d_ac/a3d_ac_v1'),
]

def parse_log_tail(log_path, n=20):
    if not log_path.exists(): return []
    lines = log_path.read_text().splitlines()
    return lines[-n:]

def extract_metrics(lines):
    best_ap = None; best_epoch = None; cur_epoch = None; cur_loss = None
    for l in reversed(lines):
        if 'New best AP' in l and best_ap is None:
            try: best_ap = float(l.split('AP=')[1].split()[0])
            except: pass
        if 'New best AP' in l and best_epoch is None:
            try: best_epoch = int(l.split('epoch')[1].strip().rstrip('*').strip())
            except: pass
        if '[CBM' in l or '[WARMUP' in l:
            if cur_epoch is None:
                try: cur_epoch = int(l.split('Ep')[1].split('/')[0].strip())
                except: pass
            if cur_loss is None:
                try: cur_loss = float(l.split('loss=')[1].split()[0])
                except: pass
    return best_ap, best_epoch, cur_epoch, cur_loss

def load_results(out_dir):
    p = out_dir / 'results.json'
    if p.exists():
        with open(p) as f: return json.load(f)
    return None

def main():
    print('\n' + '='*70)
    print('CG-CRASH v4 Training Monitor')
    print('='*70)
    for name, out_dir in EXPS:
        log_path = out_dir / 'nohup.log'
        lines = parse_log_tail(log_path)
        best_ap, best_epoch, cur_epoch, cur_loss = extract_metrics(lines)
        results = load_results(out_dir)

        status = 'RUNNING' if log_path.exists() else 'NOT STARTED'
        if log_path.exists():
            mtime = os.path.getmtime(log_path)
            age = time.time() - mtime
            if age > 600: status = 'STALLED'

        print(f'\n[{name}]  Status: {status}')
        if cur_epoch:  print(f'  Current epoch : {cur_epoch}')
        if cur_loss:   print(f'  Current loss  : {cur_loss:.4f}')
        if best_ap:    print(f'  Best AP       : {best_ap:.4f} (ep {best_epoch})')
        if results:
            print(f'  Results.json  : AP={results["AP"]:.4f} '
                  f'mTTA={results["mTTA"]:.4f}')
        if lines:
            print(f'  Last log line : {lines[-1].strip()[-100:]}')

    print('\n' + '='*70)
    print('Baseline (CG-CRASH v3): AP=68.19%, mTTA=1.75s')
    print('Target  (NeurIPS 2026): AP>80%, mTTA>3.5s, WHY+WHEN interpretable')
    print('='*70 + '\n')

if __name__ == '__main__':
    main()
