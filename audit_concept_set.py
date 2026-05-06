#!/usr/bin/env python3
import re
import json
import argparse
from pathlib import Path
from collections import Counter


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def family_of(s: str) -> str:
    rules = {
        'occlusion_visibility': ['visibility', 'occlusion', 'blind', 'glare', 'dark', 'night', 'fog'],
        'right_of_way_conflict': ['red light', 'intersection', 'crossing', 'right-of-way', 'u-turn', 'turn'],
        'relative_motion': ['tailgating', 'following', 'merging', 'overtaking', 'oncoming', 'lane', 'cutting'],
        'vulnerable_agents': ['pedestrian', 'cyclist', 'motorcyclist', 'scooter'],
        'road_surface_weather': ['wet', 'snow', 'icy', 'weather', 'road surface'],
        'obstacle_density': ['obstacle', 'blocked', 'stopped', 'debris', 'truck', 'congestion', 'construction'],
    }
    ls = s.lower()
    for fam, keys in rules.items():
        if any(k in ls for k in keys):
            return fam
    return 'other'


def main():
    p = argparse.ArgumentParser(description='Audit concept set quality and structure')
    p.add_argument('--concept_file', required=True)
    p.add_argument('--output_json', required=True)
    args = p.parse_args()

    lines = [l.strip() for l in open(args.concept_file, encoding='utf-8') if l.strip()]
    normalized = [normalize(x) for x in lines]
    counts = Counter(normalized)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    lengths = [len(x.split()) for x in lines]
    family_counts = Counter(family_of(x) for x in lines)

    out = {
        'concept_file': args.concept_file,
        'num_concepts': len(lines),
        'num_unique_normalized': len(counts),
        'num_duplicates': len(duplicates),
        'duplicate_examples': list(duplicates.items())[:50],
        'avg_num_words': sum(lengths) / max(len(lengths), 1),
        'max_num_words': max(lengths) if lengths else 0,
        'min_num_words': min(lengths) if lengths else 0,
        'family_counts': dict(family_counts),
        'sample_concepts': lines[:30],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({'output_json': str(out_path), 'num_concepts': len(lines)}, indent=2))


if __name__ == '__main__':
    main()
