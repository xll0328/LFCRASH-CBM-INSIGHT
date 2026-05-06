#!/usr/bin/env python3
import re
import json
import argparse
from pathlib import Path
from collections import Counter

RISK_KEYWORDS = [
    'conflict', 'risk', 'blind', 'occluded', 'obstructed', 'ambiguity', 'merge', 'turn',
    'cross-traffic', 'oncoming', 'proximity', 'cut-in', 'lane change', 'encroachment',
    'visibility', 'clearance', 'collision', 'skidding', 'traction', 'pedestrian', 'cyclist',
    'motorcyclist', 'scooter', 'right-of-way', 'wrong-way', 'tailgating', 'stopped vehicle'
]
NOISE_KEYWORDS = [
    'sky', 'building', 'power lines', 'sign gantry', 'sedan in lane', 'clouds', 'urban background',
    'pole with signal', 'crosswalk markings', 'concrete barrier'
]


def normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'^[\-\*\d\.\)\s]+', '', s)
    s = re.sub(r'\ba photo of\b', '', s)
    s = re.sub(r'\bthe\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip(' .,:;')
    return s


def score_phrase(phrase: str, source: str) -> int:
    score = 3 if source == 'risk_factors' else 1
    if any(k in phrase for k in RISK_KEYWORDS):
        score += 2
    if any(k in phrase for k in NOISE_KEYWORDS):
        score -= 2
    if len(phrase.split()) > 10:
        score -= 1
    return score


def parse_content(content: str):
    cleaned = content.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
        parsed = []
        for source in ['risk_factors', 'raw_concepts']:
            phrases = data.get(source, [])
            if isinstance(phrases, str):
                phrases = [phrases]
            for p in phrases:
                norm = normalize_phrase(p)
                if norm:
                    parsed.append({'phrase': norm, 'source': source, 'score': score_phrase(norm, source)})
        parsed.sort(key=lambda x: (-x['score'], x['phrase']))
        return parsed
    except Exception:
        lines = []
        for x in re.split(r'[\n,;]+', cleaned):
            norm = normalize_phrase(x)
            if norm:
                lines.append({'phrase': norm, 'source': 'fallback', 'score': score_phrase(norm, 'fallback')})
        lines.sort(key=lambda x: (-x['score'], x['phrase']))
        return lines


def main():
    p = argparse.ArgumentParser(description='Normalize and aggregate raw concepts with risk-priority scoring')
    p.add_argument('--input_jsonl', required=True)
    p.add_argument('--output_json', required=True)
    p.add_argument('--min_count', type=int, default=2)
    p.add_argument('--min_score', type=int, default=2)
    args = p.parse_args()

    counter = Counter()
    best_meta = {}
    per_item = []
    with open(args.input_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            parsed = parse_content(item.get('content', ''))
            kept = []
            for obj in parsed:
                phrase = obj['phrase']
                if obj['score'] < args.min_score:
                    continue
                counter[phrase] += 1
                if phrase not in best_meta or obj['score'] > best_meta[phrase]['score']:
                    best_meta[phrase] = {'score': obj['score'], 'source': obj['source']}
                kept.append(obj)
            per_item.append({
                'frame_path': item.get('frame_path'),
                'dataset': item.get('dataset'),
                'concepts': kept,
            })

    concepts = [
        {
            'concept': concept,
            'count': count,
            'best_score': best_meta[concept]['score'],
            'preferred_source': best_meta[concept]['source'],
        }
        for concept, count in counter.most_common()
        if count >= args.min_count
    ]

    out = {
        'num_unique_before_filter': len(counter),
        'num_concepts_after_filter': len(concepts),
        'concepts': concepts,
        'per_item': per_item,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({'output_json': str(out_path), 'num_concepts_after_filter': len(concepts)}, indent=2))


if __name__ == '__main__':
    main()
