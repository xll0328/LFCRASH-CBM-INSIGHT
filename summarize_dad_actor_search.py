#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEARCH_DIR = ROOT / 'output' / 'reviewer_recovery' / 'dad_actor_search'


def main():
    rows = []
    for p in SEARCH_DIR.glob('*.json'):
        data = json.loads(p.read_text())
        rows.append({
            'tag': p.stem,
            'classifier_AP': data['classifier_trigger']['AP'],
            'classifier_mTTA': data['classifier_trigger']['mTTA'],
            'actor_AP': data['actor_trigger']['AP'],
            'actor_mTTA': data['actor_trigger']['mTTA'],
            'actor_TTA_R80': data['actor_trigger']['TTA_R80'],
            'actor_P_R80': data['actor_trigger']['P_R80'],
            'actor_crossing_rate': data['actor_diagnostics']['crossing_rate_at_threshold'],
            'actor_peak_mean': data['actor_diagnostics']['peak_mean'],
        })
    rows.sort(key=lambda r: (r['actor_AP'], r['actor_mTTA']), reverse=True)
    out = ROOT / 'output' / 'reviewer_recovery' / 'dad_actor_search_summary.json'
    out.write_text(json.dumps({'rows': rows, 'best_by_actor_ap': rows[0] if rows else None}, indent=2))
    print(json.dumps({'num_rows': len(rows), 'best_by_actor_ap': rows[0] if rows else None}, indent=2))


if __name__ == '__main__':
    main()
