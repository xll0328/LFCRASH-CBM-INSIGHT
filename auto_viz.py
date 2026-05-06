#!/usr/bin/env python3
"""
auto_viz.py — 等待训练达到目标 AP 后自动生成可视化
Usage: python auto_viz.py --tag dad_ac_v1 --target_ap 0.70 --gpu 0
"""
import time, argparse, subprocess, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def check_best_ap(tag, exp_type='dad_ac'):
    results = ROOT / f'output/{exp_type}/{tag}/results.json'
    if results.exists():
        with open(results) as f:
            d = json.load(f)
        return d.get('AP', 0.0), d.get('mTTA', 0.0)
    return 0.0, 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag',        type=str, default='dad_ac_v1')
    p.add_argument('--exp_type',   type=str, default='dad_ac')
    p.add_argument('--target_ap',  type=float, default=0.70)
    p.add_argument('--gpu',        type=int, default=0)
    p.add_argument('--check_every',type=int, default=300,
                   help='Check every N seconds')
    args = p.parse_args()

    ckpt = ROOT / f'output/{args.exp_type}/{args.tag}/best_model.pt'
    print(f'Watching {args.tag} for AP >= {args.target_ap:.2%}...')
    print(f'Checking every {args.check_every}s')

    while True:
        ap, mtta = check_best_ap(args.tag, args.exp_type)
        print(f'[{time.strftime("%H:%M:%S")}] Current best AP={ap:.4f} mTTA={mtta:.4f}')

        if ap >= args.target_ap:
            print(f'Target AP {args.target_ap:.2%} reached! Launching visualization...')
            cmd = [
                'python3', 'visualize_v4.py',
                '--ckpt', str(ckpt),
                '--gpu', str(args.gpu),
                '--n_samples', '5',
                '--out_dir', str(ROOT / f'output/{args.exp_type}/{args.tag}/viz'),
            ]
            result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
            print(result.stdout)
            if result.returncode == 0:
                print('Visualization complete!')
            else:
                print(f'Viz error: {result.stderr[:500]}')
            break

        time.sleep(args.check_every)

if __name__ == '__main__':
    main()
