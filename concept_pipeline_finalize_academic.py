#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

FAMILY_RULES = [
    ('vulnerable_road_users', ['pedestrian', 'cyclist', 'bicycle', 'motorcycle', 'motorcyclist', 'scooter', 'two-wheeler', 'rider', 'crosswalk']),
    ('surface_weather', ['wet', 'slippery', 'rain', 'fog', 'glare', 'hydroplan', 'traction', 'surface', 'nighttime', 'low-light', 'precipitation', 'pavement']),
    ('occlusion_visibility', ['visibility', 'occlusion', 'occluded', 'blind', 'sightline', 'sight distance', 'glare', 'illumination', 'lighting']),
    ('right_of_way_conflict', ['right-of-way', 'intersection', 'cross-traffic', 'oncoming', 'yield', 'signal', 'stop-line', 'conflict zone']),
    ('relative_motion', ['proximity', 'deceleration', 'braking', 'following distance', 'rear-end', 'closing speed']),
    ('agent_behavior', ['lane change', 'encroachment', 'merge', 'turn', 'cut-in', 'brake light', 'crossing intent', 'entry conflict', 'filtering']),
    ('road_layout_constraint', ['lane boundary', 'lane marking', 'clearance', 'road curvature', 'shoulder', 'narrow', 'junction', 'roadside', 'carriageway', 'lane width']),
    ('traffic_density_obstacle', ['traffic density', 'congestion', 'parked vehicle', 'large vehicle', 'obstacle', 'mixed traffic', 'heavy vehicle']),
    ('imminent_crash_cue', ['collision risk', 'crash', 'emergency', 'sudden deceleration', 'conflict point']),
]

EXACT_REPLACEMENTS = {
    'pedestrian crossing potential': 'pedestrian crossing risk',
    'potential pedestrian crossing': 'pedestrian crossing risk',
    'rear-end collision potential': 'rear-end collision risk',
    'potential rear-end collision': 'rear-end collision risk',
    'potential hydroplaning risk': 'hydroplaning risk',
    'potential lane change maneuver': 'lane change maneuver',
    'potential lane change intent': 'lane change intent',
    'lane change conflict potential': 'lane change conflict',
    'potential lane merge conflict': 'lane merge conflict',
    'potential merge conflict': 'merge conflict',
    'potential merging conflict': 'merge conflict',
    'vehicle merging conflict': 'merge conflict',
    'merging vehicle conflict': 'merge conflict',
    'intersection conflict potential': 'intersection conflict',
    'potential right-of-way conflict': 'right-of-way conflict',
    'right-of-way uncertainty': 'right-of-way ambiguity',
    'potential pedestrian emergence': 'pedestrian emergence',
    'pedestrian emergence risk': 'pedestrian emergence',
    'potential pedestrian occlusion': 'pedestrian occlusion',
    'pedestrian crossing': 'pedestrian crossing risk',
    'pedestrian_crossing_intent': 'pedestrian crossing intent',
    'adjacent_vehicle_proximity': 'adjacent vehicle proximity',
    'lead_vehicle_braking': 'lead vehicle braking',
    'rear_end_collision_risk': 'rear-end collision risk',
    'vulnerable_road_user_proximity': 'vulnerable road user proximity',
    'intersection_approach': 'intersection approach',
    'low_light_visibility': 'low light visibility',
}

PREFIXES = ('potential ', 'possible ')


def normalize_text(text: str) -> str:
    text = text.strip().lower().replace('_', ' ')
    text = text.replace('–', '-').replace('—', '-')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*-\s*', '-', text)
    return text.strip(' .,:;')


def canonicalize(text: str) -> str:
    c = normalize_text(text)
    if c in EXACT_REPLACEMENTS:
        return EXACT_REPLACEMENTS[c]
    for prefix in PREFIXES:
        if c.startswith(prefix):
            stripped = c[len(prefix):].strip()
            if stripped in EXACT_REPLACEMENTS:
                return EXACT_REPLACEMENTS[stripped]
            c = stripped
            break
    c = re.sub(r'\breduced nighttime visibility\b', 'nighttime visibility reduction', c)
    c = re.sub(r'\bnighttime reduced visibility\b', 'nighttime visibility reduction', c)
    c = re.sub(r'\bnighttime low visibility\b', 'nighttime visibility reduction', c)
    c = re.sub(r'\blimited forward visibility\b', 'limited forward visibility', c)
    c = re.sub(r'\breduced forward visibility\b', 'reduced forward visibility', c)
    c = re.sub(r'\breduced lateral clearance\b', 'reduced lateral clearance', c)
    c = re.sub(r'\blimited lateral clearance\b', 'limited lateral clearance', c)
    c = re.sub(r'\s+', ' ', c).strip()
    return c


def assign_family(concept: str) -> str:
    for family, keywords in FAMILY_RULES:
        if any(k in concept for k in keywords):
            return family
    return 'agent_behavior'


def build_signature(concept: str) -> str:
    s = concept
    s = s.replace(' risk', '')
    s = s.replace(' potential', '')
    s = s.replace(' conditions', '')
    s = s.replace(' state', '')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def choose_representative(variants_with_meta):
    def score(v):
        concept, total_count, best_score = v
        specificity = len(concept.split())
        has_risk = 1 if 'risk' in concept or 'conflict' in concept or 'ambiguity' in concept else 0
        return (best_score, has_risk, total_count, specificity)
    return max(variants_with_meta, key=score)[0]


def main():
    p = argparse.ArgumentParser(description='Academic-grade deterministic ontology finalizer')
    p.add_argument('--input_json', required=True)
    p.add_argument('--output_json', required=True)
    p.add_argument('--output_txt', required=True)
    p.add_argument('--min_count', type=int, default=3)
    p.add_argument('--min_score', type=int, default=3)
    p.add_argument('--topn', type=int, default=None)
    args = p.parse_args()

    data = json.load(open(args.input_json))

    normalized_entries = []
    for item in data.get('concepts', []):
        if item.get('count', 0) < args.min_count:
            continue
        if item.get('best_score', 0) < args.min_score:
            continue
        canon = canonicalize(item['concept'])
        if not canon:
            continue
        normalized_entries.append({
            'concept': canon,
            'count': item['count'],
            'best_score': item['best_score'],
            'signature': build_signature(canon),
        })

    grouped = defaultdict(list)
    for item in normalized_entries:
        grouped[item['signature']].append(item)

    concept_details = []
    merge_examples = {}
    for signature, items in grouped.items():
        aggregate = defaultdict(lambda: {'count': 0, 'best_score': 0})
        for item in items:
            aggregate[item['concept']]['count'] += item['count']
            aggregate[item['concept']]['best_score'] = max(aggregate[item['concept']]['best_score'], item['best_score'])

        representatives = [(concept, meta['count'], meta['best_score']) for concept, meta in aggregate.items()]
        canonical = choose_representative(representatives)
        total_count = sum(meta['count'] for meta in aggregate.values())
        best_score = max(meta['best_score'] for meta in aggregate.values())
        variants = sorted(aggregate.keys())
        concept_details.append({
            'concept': canonical,
            'count': total_count,
            'best_score': best_score,
            'family': assign_family(canonical),
            'variants': variants,
            'signature': signature,
        })
        if len(variants) > 1:
            merge_examples[canonical] = variants

    concept_details.sort(key=lambda x: (-x['count'], -x['best_score'], x['concept']))
    if args.topn is not None:
        concept_details = concept_details[:args.topn]

    families = defaultdict(list)
    for item in concept_details:
        families[item['family']].append(item['concept'])

    out = {
        'status': 'academic_deterministic_finalized',
        'input_count': len(data.get('concepts', [])),
        'num_standard_concepts': len(concept_details),
        'standard_concepts': [x['concept'] for x in concept_details],
        'families': {k: v for k, v in families.items()},
        'merge_examples': merge_examples,
        'concept_details': concept_details,
        'source': str(Path(args.input_json)),
        'method': 'conservative_rule_based_finalization',
        'notes': [
            'Keeps fine-grained risk primitives whenever semantics differ.',
            'Only merges formatting variants, explicit paraphrases, and clearly symmetric duplicates.',
            'Avoids broad collapsing of reduced/limited/high/low descriptors unless explicitly mapped.'
        ]
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    out_txt = Path(args.output_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, 'w', encoding='utf-8') as f:
        for concept in out['standard_concepts']:
            f.write(concept + '\n')

    print(json.dumps({
        'output_json': str(out_json),
        'output_txt': str(out_txt),
        'num_standard_concepts': len(out['standard_concepts'])
    }, indent=2))


if __name__ == '__main__':
    main()
