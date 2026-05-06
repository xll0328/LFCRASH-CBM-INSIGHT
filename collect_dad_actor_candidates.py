#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'output'


def collect_candidates():
    candidates = []
    for results_path in OUTPUT.glob('dad_ac/**/results.json'):
        ckpt = results_path.parent / 'best_model.pt'
        if not ckpt.exists():
            ckpt = results_path.parent / 'best_model.pth'
        if not ckpt.exists():
            continue
        try:
            data = json.loads(results_path.read_text())
        except Exception:
            continue
        candidates.append({
            'tag': results_path.parent.name,
            'results_path': str(results_path),
            'checkpoint': str(ckpt),
            'AP': data.get('AP'),
            'mTTA': data.get('mTTA'),
            'TTA_R80': data.get('TTA_R80'),
            'P_R80': data.get('P_R80'),
            'epoch': data.get('epoch'),
        })
    return candidates


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--min_ap', type=float, default=0.60)
    args = parser.parse_args()

    rows = [r for r in collect_candidates() if (r['AP'] or 0.0) >= args.min_ap]
    rows.sort(key=lambda r: ((r['AP'] or 0.0), (r['mTTA'] or 0.0)), reverse=True)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'candidates': rows}, indent=2))
    print(json.dumps({'num_candidates': len(rows), 'top5': rows[:5]}, indent=2))


if __name__ == '__main__':
    main()
