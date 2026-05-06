#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

FAMILY_RULES = [
    ('vulnerable_road_users', ['pedestrian', 'cyclist', 'bicycle', 'motorcycle', 'motorcyclist', 'scooter', 'two-wheeler', 'rider', 'crosswalk']),
    ('surface_weather', ['wet', 'slippery', 'rain', 'fog', 'glare', 'hydroplan', 'traction', 'surface', 'nighttime', 'low-light', 'precipitation']),
    ('occlusion_visibility', ['visibility', 'occlusion', 'occluded', 'blind', 'glare', 'sight distance', 'peripheral', 'forward visibility']),
    ('right_of_way_conflict', ['right-of-way', 'intersection', 'cross-traffic', 'oncoming', 'yield', 'signal', 'stop-line']),
    ('relative_motion', ['proximity', 'deceleration', 'braking', 'closing', 'rear-end', 'following distance', 'adjacent vehicle', 'lead vehicle', 'preceding vehicle']),
    ('agent_behavior', ['lane change', 'encroachment', 'merge', 'turn', 'cut-in', 'brake light', 'crossing', 'entry conflict']),
    ('road_layout_constraint', ['lane boundary', 'lane marking', 'clearance', 'road curvature', 'shoulder', 'narrow', 'junction', 'roadside']),
    ('traffic_density_obstacle', ['traffic density', 'vehicle proximity', 'large vehicle', 'obstacle', 'congestion', 'parked vehicle']),
    ('imminent_crash_cue', ['collision risk', 'conflict zone', 'emergency', 'crash', 'imminent']),
]

DROP_PATTERNS = [
    r'^potential ',
    r'^possible ',
    r'^moderate ',
    r'^high ',
    r'^low ',
    r'^limited ',
    r'^reduced ',
    r'^adjacent ',
    r'^preceding ',
    r'^lead ',
]

NOISY_CANONICALS = {
    'traction', 'traction surface', 'visibility conditions', 'visibility range', 'ambient light', 'ambient illumination',
    'nighttime visibility', 'intersection entry', 'intersection proximity', 'lane merge', 'oncoming traffic', 'pedestrian presence',
    'pedestrian activity', 'vehicle proximity', 'lateral visibility', 'peripheral visibility'
}

REPLACEMENTS = [
    ('rear-end collision potential', 'rear-end collision risk'),
    ('potential rear-end collision', 'rear-end collision risk'),
    ('potential pedestrian crossing', 'pedestrian crossing risk'),
    ('pedestrian crossing potential', 'pedestrian crossing risk'),
    ('potential merge conflict', 'merge conflict'),
    ('potential merging conflict', 'merge conflict'),
    ('vehicle merging conflict', 'merge conflict'),
    ('merging vehicle conflict', 'merge conflict'),
    ('potential lane encroachment', 'lane encroachment'),
    ('potential lane merge conflict', 'lane merge conflict'),
    ('intersection conflict potential', 'intersection conflict'),
    ('potential hydroplaning risk', 'hydroplaning risk'),
    ('potential lane change maneuver', 'lane change maneuver'),
    ('potential lane change intent', 'lane change intent'),
    ('lane change conflict potential', 'lane change conflict'),
    ('reduced lateral clearance', 'lateral clearance'),
    ('limited lateral clearance', 'lateral clearance'),
    ('limited forward visibility', 'forward visibility limitation'),
    ('reduced forward visibility', 'forward visibility limitation'),
    ('reduced visibility', 'visibility reduction'),
    ('limited visibility', 'visibility reduction'),
    ('nighttime low visibility', 'nighttime visibility reduction'),
    ('nighttime reduced visibility', 'nighttime visibility reduction'),
    ('reduced nighttime visibility', 'nighttime visibility reduction'),
    ('adjacent lane vehicle proximity', 'adjacent lane proximity'),
    ('adjacent lane proximity', 'adjacent lane proximity'),
    ('adjacent vehicle proximity', 'vehicle proximity'),
    ('vehicle proximity', 'vehicle proximity'),
    ('lead vehicle proximity', 'lead vehicle proximity'),
    ('preceding vehicle proximity', 'lead vehicle proximity'),
    ('lead vehicle deceleration', 'lead vehicle deceleration'),
    ('preceding vehicle deceleration', 'lead vehicle deceleration'),
    ('right-of-way uncertainty', 'right-of-way ambiguity'),
    ('intersection right-of-way ambiguity', 'right-of-way ambiguity'),
    ('intersection right-of-way conflict', 'right-of-way conflict'),
    ('potential right-of-way conflict', 'right-of-way conflict'),
    ('pedestrian emergence risk', 'pedestrian emergence'),
    ('potential pedestrian emergence', 'pedestrian emergence'),
    ('potential pedestrian occlusion', 'pedestrian occlusion'),
    ('potential pedestrian presence', 'pedestrian presence'),
    ('red traffic signal', 'traffic signal state'),
    ('traffic signal visibility', 'traffic signal state'),
    ('oncoming vehicle presence', 'oncoming traffic'),
    ('oncoming vehicle proximity', 'oncoming vehicle proximity'),
]


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def canonicalize(concept: str) -> str:
    c = normalize(concept)
    for old, new in REPLACEMENTS:
        if c == old:
            return new
    for pat in DROP_PATTERNS:
        c = re.sub(pat, '', c)
    c = re.sub(r'\bvehicle merging\b', 'merge', c)
    c = re.sub(r'\bmerging vehicle\b', 'merge', c)
    c = re.sub(r'\bpedestrian crossing\b', 'pedestrian crossing risk', c)
    c = c.replace('_', ' ')
    c = re.sub(r'\s+', ' ', c).strip()
    if c in NOISY_CANONICALS:
        return ''
    return c


def assign_family(concept: str) -> str:
    for family, keywords in FAMILY_RULES:
        if any(k in concept for k in keywords):
            return family
    return 'agent_behavior'


def main():
    p = argparse.ArgumentParser(description='Deterministically finalize large-vocab ontology without LLM clustering')
    p.add_argument('--input_json', required=True)
    p.add_argument('--output_json', required=True)
    p.add_argument('--output_txt', required=True)
    p.add_argument('--min_count', type=int, default=5)
    p.add_argument('--min_score', type=int, default=3)
    p.add_argument('--topn', type=int, default=None)
    args = p.parse_args()

    data = json.load(open(args.input_json))
    grouped = defaultdict(lambda: {'count': 0, 'best_score': 0, 'variants': []})

    for item in data.get('concepts', []):
        if item.get('count', 0) < args.min_count:
            continue
        if item.get('best_score', 0) < args.min_score:
            continue
        raw = normalize(item['concept'])
        canon = canonicalize(raw)
        if not canon:
            continue
        grouped[canon]['count'] += item.get('count', 0)
        grouped[canon]['best_score'] = max(grouped[canon]['best_score'], item.get('best_score', 0))
        grouped[canon]['variants'].append(raw)

    concepts = []
    for canon, meta in grouped.items():
        variants = sorted(set(meta['variants']))
        concepts.append({
            'concept': canon,
            'count': meta['count'],
            'best_score': meta['best_score'],
            'family': assign_family(canon),
            'variants': variants,
        })

    concepts.sort(key=lambda x: (-x['count'], -x['best_score'], x['concept']))
    if args.topn is not None:
        concepts = concepts[:args.topn]

    families = defaultdict(list)
    merge_examples = {}
    for item in concepts:
        families[item['family']].append(item['concept'])
        if len(item['variants']) > 1:
            merge_examples[item['concept']] = item['variants']

    out = {
        'status': 'deterministic_finalized',
        'input_count': len(data.get('concepts', [])),
        'num_standard_concepts': len(concepts),
        'standard_concepts': [x['concept'] for x in concepts],
        'families': {k: v for k, v in families.items()},
        'merge_examples': merge_examples,
        'concept_details': concepts,
        'source': str(Path(args.input_json)),
        'method': 'rule_based_deduplication_and_family_assignment',
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    out_txt = Path(args.output_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, 'w', encoding='utf-8') as f:
        for c in out['standard_concepts']:
            f.write(c + '\n')

    print(json.dumps({'output_json': str(out_json), 'output_txt': str(out_txt), 'num_standard_concepts': len(out['standard_concepts'])}, indent=2))


if __name__ == '__main__':
    main()
