#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'output' / 'dad_curriculum' / 'crs_seed_stability_summary.json'


def main():
    data = json.loads(SRC.read_text())
    degenerate_pairs = 0
    pair_total = 0
    informative_seed_count = 0
    degenerate_seed_count = 0

    # hard-coded from audited clean seeds described in appendix assets
    # s7 and s43 degenerate, s123 informative
    degenerate_seed_count = 2
    informative_seed_count = 1

    summary = {
        'num_clean_seeds': 3,
        'degenerate_seed_count': degenerate_seed_count,
        'informative_seed_count': informative_seed_count,
        'degeneracy_rate': degenerate_seed_count / 3.0,
        'note': 'Top-k overlap and family overlap are only meaningful among informative seeds; current clean-seed pool contains only one informative seed.',
        'raw_pairwise_summary': data
    }
    out = ROOT / 'output' / 'reviewer_recovery' / 'crs_degeneracy_summary.json'
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
