#!/usr/bin/env python3
import json
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Promote adjudicated concept set to final candidate artifacts')
    p.add_argument('--adjudicated_json', required=True)
    p.add_argument('--output_txt', required=True)
    p.add_argument('--output_meta', required=True)
    args = p.parse_args()

    data = json.load(open(args.adjudicated_json))
    concepts = data.get('final_concepts', [])
    families = data.get('families', {})

    txt_path = Path(args.output_txt)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for c in concepts:
            f.write(c.strip() + '\n')

    meta = {
        'name': 'new_discovered_concept_set_final_candidate',
        'version': 'v2_candidate',
        'num_concepts': len(concepts),
        'source': 'qwen3.5-27b final adjudication over compact v1 set and refined candidate pool',
        'families': families,
        'adjudicated_json': str(Path(args.adjudicated_json)),
        'note': 'Promoted automatically from final adjudication stage.'
    }
    meta_path = Path(args.output_meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps({'output_txt': str(txt_path), 'output_meta': str(meta_path), 'num_concepts': len(concepts)}, indent=2))


if __name__ == '__main__':
    main()
