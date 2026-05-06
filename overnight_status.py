#!/usr/bin/env python3
"""
 overnight_status.py
 每30分钟检查一次所有任务进度，记录到 output/overnight_report.log
"""
import subprocess, time, os, json
from datetime import datetime

LOG = '/data/sony/LFCRASH/LFCRASH-CBM/output/overnight_report.log'
BASE = '/data/sony/LFCRASH/LFCRASH-CBM'

TASKS = {
    'a3d_full':       f'{BASE}/output/phase2_ablation/a3d_full_rerun.log',
    'crash_no_align': f'{BASE}/output/phase2_ablation/crash_no_align_rerun.log',
    'dad_no_recon':   f'{BASE}/output/phase2_ablation/dad_no_recon_rerun.log',
    'dad_h512_v2':    f'{BASE}/output/dad_sota_push/dad_h512_v2.log',
    'run_eval':       f'{BASE}/output/main_results_eval.log',
    'crash_cgta':     f'{BASE}/output/cgta_crs_training/crash_cgta.log',
    'a3d_sota':       f'{BASE}/output/sota_push/a3d_sota.log',
    'dad_z512':       f'{BASE}/output/dad_sota_push/dad_z512.log',
}

PIDS = [3558075, 3569107, 3576140, 3579906, 3580924, 2119909, 3005962, 2998088]

def tail(path, n=3):
    try:
        lines = open(path).readlines()
        relevant = [l.strip() for l in lines if any(k in l for k in ['AP=','Epoch','loss=','crash_','dad_','a3d_','Error','Traceback'])]
        return relevant[-n:] if relevant else lines[-n:]
    except:
        return ['(no output yet)']

def check_pids():
    alive = []
    dead = []
    for pid in PIDS:
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except:
            dead.append(pid)
    return alive, dead

def check_results():
    results = {}
    for tag in ['a3d_full','crash_no_align','dad_no_recon']:
        rj = f'{BASE}/output/phase2_ablation/{tag}/results.json'
        try:
            d = json.load(open(rj))
            results[tag] = f"AP={d.get('AP','?'):.4f}"
        except:
            results[tag] = 'pending'
    # eval results
    try:
        lines = open(TASKS['run_eval']).readlines()
        eval_lines = [l.strip() for l in lines if any(ds in l for ds in ['crash','dad','a3d']) and 'AP' in l and '===' not in l and '[CG' not in l]
        results['eval'] = eval_lines[-5:] if eval_lines else ['pending']
    except:
        results['eval'] = ['pending']
    return results

with open(LOG, 'a') as f:
    f.write('\n' + '='*70 + '\n')
    f.write(f'overnight_status.py started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write('='*70 + '\n')

while True:
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    alive, dead = check_pids()
    results = check_results()

    report = []
    report.append(f'\n{"-"*60}')
    report.append(f'[{now}] STATUS REPORT')
    report.append(f'  Alive PIDs ({len(alive)}): {alive}')
    if dead:
        report.append(f'  Dead  PIDs ({len(dead)}): {dead}')

    report.append('\n  [Ablation Results]')
    for tag, val in results.items():
        if tag != 'eval':
            report.append(f'    {tag}: {val}')

    report.append('\n  [Eval Progress]')
    for line in results['eval']:
        report.append(f'    {line}')

    report.append('\n  [Recent Log Tails]')
    for name, path in TASKS.items():
        lines = tail(path, 2)
        report.append(f'    [{name}]')
        for l in lines:
            report.append(f'      {l}')

    text = '\n'.join(report) + '\n'
    with open(LOG, 'a') as f:
        f.write(text)
    print(text, flush=True)

    # 如果所有 ablation 都完成了，发出完成信号
    if all(v != 'pending' for k,v in results.items() if k != 'eval'):
        with open(LOG, 'a') as f:
            f.write(f'\n*** ALL ABLATION TASKS COMPLETE at {now} ***\n')

    time.sleep(1800)  # 每30分钟检查一次
