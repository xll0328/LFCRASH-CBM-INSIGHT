#!/usr/bin/env python3
"""
auto_monitor.py — 每5分钟扫描所有DAD训练日志，发现新best自动记录
Usage: python3 auto_monitor.py &
"""
import time, re, json
from pathlib import Path
from datetime import datetime

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')

LOGS = {
    'finetune_z256':    ROOT / 'output' / 'dad_finetune_z256.log',
    'curriculum_v1':    ROOT / 'output' / 'dad_curriculum_v1.log',
    'curriculum_v2':    ROOT / 'output' / 'dad_curriculum_v2.log',
    'no_sparse_long':   ROOT / 'output' / 'dad_no_sparse_long.log',
}

BEST_FILE = ROOT / 'output' / 'dad_training_best.json'


def parse_best_ap(log_path):
    """Parse best AP and epoch from log file."""
    if not log_path.exists():
        return None, None
    best_ap, best_ep = 0.0, 0
    for line in open(log_path):
        m = re.search(r'New best AP=([0-9.]+) at epoch (\d+)', line)
        if m:
            ap = float(m.group(1))
            ep = int(m.group(2))
            if ap > best_ap:
                best_ap, best_ep = ap, ep
    return best_ap if best_ap > 0 else None, best_ep


def parse_latest_epoch(log_path):
    """Parse latest epoch number and loss."""
    if not log_path.exists():
        return None, None
    ep, loss = None, None
    for line in open(log_path):
        m = re.search(r'Epoch\s+(\d+)/', line)
        if m:
            ep = int(m.group(1))
        m2 = re.search(r'Loss: ([0-9.]+)', line)
        if m2:
            loss = float(m2.group(1))
    return ep, loss


def load_known():
    try:
        return json.load(open(BEST_FILE))
    except:
        return {}


def save_known(d):
    json.dump(d, open(BEST_FILE, 'w'), indent=2)


def main():
    print(f'[auto_monitor] Started at {datetime.now().strftime("%H:%M:%S")}')
    known = load_known()
    overall_best_ap = max((v.get('best_ap', 0) for v in known.values()), default=0)

    while True:
        now = datetime.now().strftime('%H:%M:%S')
        updated = False
        lines = []

        for name, log_path in LOGS.items():
            best_ap, best_ep = parse_best_ap(log_path)
            cur_ep, cur_loss = parse_latest_epoch(log_path)

            prev_best = known.get(name, {}).get('best_ap', 0)

            if best_ap and best_ap > prev_best:
                delta = best_ap - prev_best
                lines.append(
                    f'  *** NEW BEST [{name}] AP={best_ap*100:.2f}% (+{delta*100:.2f}%) @ ep{best_ep} ***'
                )
                known[name] = {'best_ap': best_ap, 'best_ep': best_ep,
                               'cur_ep': cur_ep, 'cur_loss': cur_loss,
                               'updated_at': now}
                updated = True

                if best_ap > overall_best_ap:
                    overall_best_ap = best_ap
                    lines.append(
                        f'  !!!! OVERALL DAD BEST: AP={best_ap*100:.2f}% [{name}] !!!!'
                    )
            else:
                ep_str = f'ep{cur_ep}' if cur_ep else '?'
                loss_str = f'loss={cur_loss:.4f}' if cur_loss else '?'
                best_str = f'best={best_ap*100:.2f}%' if best_ap else 'no eval yet'
                lines.append(f'  [{name}] {ep_str} {loss_str} {best_str}')

        print(f'\n[{now}]')
        for l in lines:
            print(l)

        if updated:
            save_known(known)
            # Also regenerate dashboard
            import subprocess
            subprocess.run(['python3', str(ROOT/'dashboard.py')],
                          capture_output=True, timeout=30)

        time.sleep(300)  # 5 minutes


if __name__ == '__main__':
    main()
