#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path

import requests

def build_system_prompt(mode: str) -> str:
    if mode == 'large_vocab':
        return (
            'You are building a NeurIPS-quality large-scale risk concept atlas for accident anticipation. '
            'Keep semantically distinct risk concepts separate unless they are true duplicates. '
            'Prefer transferable, intervention-friendly risk primitives over scene captions, but preserve useful granularity. '
            'Do not over-compress the ontology into a tiny list. '
            'Return JSON only.'
        )
    return (
        'You are building a NeurIPS-quality risk concept ontology for accident anticipation. '
        'Group similar concepts, remove redundant wording, and output standardized risk concepts. '
        'Prefer transferable risk primitives over frame-specific descriptions. Merge near-synonyms into one canonical concept. '
        'Keep concepts short, action- or hazard-centered, and suitable for intervention analysis. '
        'Return JSON only.'
    )


def build_user_prompt(concept_blob: str, mode: str, min_concepts: int, max_concepts: int) -> str:
    if mode == 'large_vocab':
        return (
            'Given the following mined concepts, return strict JSON with keys: '
            'standard_concepts (list of concise canonical concepts), '
            'families (object mapping family_name -> list of concepts), '
            'merge_examples (object mapping canonical concept -> list of merged variants). '
            'Use only these families: agent_behavior, relative_motion, right_of_way_conflict, occlusion_visibility, road_layout_constraint, surface_weather, vulnerable_road_users, traffic_density_obstacle, imminent_crash_cue. '
            f'Keep between {min_concepts} and {max_concepts} canonical concepts unless the input quality is clearly insufficient. '
            'Do not collapse semantically distinct hazards into one phrase just to be concise. '
            'Preserve fine-grained but reusable accident-risk distinctions, remove only true duplicates or obvious noise, '
            'and do not include prose before or after the JSON. '
            f'Concepts:\n{concept_blob}'
        )
    return (
        'Given the following mined concepts, return strict JSON with keys: '
        'standard_concepts (list of concise canonical concepts), '
        'families (object mapping family_name -> list of concepts), '
        'merge_examples (object mapping canonical concept -> list of merged variants). '
        'Use only these families: agent_behavior, relative_motion, right_of_way_conflict, occlusion_visibility, road_layout_constraint, surface_weather, vulnerable_road_users, traffic_density_obstacle, imminent_crash_cue. '
        'Do not include prose before or after the JSON. '
        f'Concepts:\n{concept_blob}'
    )



def extract_json_block(text: str):
    text = text.strip()
    if not text:
        raise ValueError('Empty response content')
    try:
        return json.loads(text)
    except Exception:
        pass

    fence_match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', text)
    if fence_match:
        return json.loads(fence_match.group(1))

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError('Could not extract JSON object from model response')


def main():
    p = argparse.ArgumentParser(description='Use an LLM to cluster and summarize mined concepts')
    p.add_argument('--input_json', required=True)
    p.add_argument('--output_json', required=True)
    p.add_argument('--api_url', default=os.environ.get('LLM_API_URL', ''))
    p.add_argument('--api_key', default=os.environ.get('LLM_API_KEY', ''))
    p.add_argument('--model', default=os.environ.get('LLM_MODEL', 'gpt-4.1'))
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--topn', type=int, default=300)
    p.add_argument('--mode', choices=['compact', 'large_vocab'], default='compact')
    p.add_argument('--min_concepts', type=int, default=120)
    p.add_argument('--max_concepts', type=int, default=400)
    args = p.parse_args()

    data = json.load(open(args.input_json))
    concepts = [x['concept'] for x in data.get('concepts', [])[:args.topn]]
    concept_blob = '\n'.join(f'- {c}' for c in concepts)
    result = {'input_count': len(concepts), 'mode': args.mode}

    if not args.api_url:
        result['status'] = 'stubbed'
        result['standard_concepts'] = concepts
        result['families'] = {'unclustered': concepts}
    else:
        headers = {'Authorization': f'Bearer {args.api_key}', 'Content-Type': 'application/json'} if args.api_key else {'Content-Type': 'application/json'}
        payload = {
            'model': args.model,
            'messages': [
                {'role': 'system', 'content': build_system_prompt(args.mode)},
                {'role': 'user', 'content': build_user_prompt(concept_blob, args.mode, args.min_concepts, args.max_concepts)}
            ],
            'temperature': 0.2,
        }
        r = requests.post(args.api_url, headers=headers, json=payload, timeout=args.timeout)
        r.raise_for_status()
        content = r.json()['choices'][0]['message']['content']
        if isinstance(content, list):
            content = ''.join(part.get('text', '') for part in content)
        parsed = extract_json_block(content)
        result['status'] = 'ok'
        result['raw_content'] = content
        result.update(parsed)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({'output_json': str(out_path), 'status': result['status']}, indent=2))


if __name__ == '__main__':
    main()
