#!/usr/bin/env python3
import json
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Export final concept set txt from clustered ontology json')
    p.add_argument('--input_json', required=True)
    p.add_argument('--output_txt', required=True)
    p.add_argument('--min_count', type=int, default=1)
    p.add_argument('--min_score', type=int, default=0)
    p.add_argument('--topn', type=int, default=None)
    args = p.parse_args()

    data = json.load(open(args.input_json))
    if 'standard_concepts' in data:
        concepts = [c.strip() for c in data.get('standard_concepts', []) if c and c.strip()]
    else:
        concepts = []
        for item in data.get('concepts', []):
            concept = item.get('concept', '').strip()
            if not concept:
                continue
            if item.get('count', 0) < args.min_count:
                continue
            if item.get('best_score', 0) < args.min_score:
                continue
            concepts.append(concept)

    if args.topn is not None:
        concepts = concepts[:args.topn]

    out_path = Path(args.output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for c in concepts:
            f.write(c + '\n')
    print(json.dumps({'output_txt': str(out_path), 'num_concepts': len(concepts)}, indent=2))


if __name__ == '__main__':
    main()
