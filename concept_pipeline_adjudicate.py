#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path

import requests

SYSTEM_PROMPT = (
    'You are serving as a final ontology adjudicator for a NeurIPS paper on accident anticipation. '
    'You are given a compact discovered concept set plus candidate mined concepts. '
    'Your job is to improve concept quality, not increase noise. '
    'Prefer transferable risk primitives, balanced family coverage, concise naming, and concepts suitable for CBM intervention analysis. '
    'Avoid scene-caption trivia, family label leakage, and near-duplicate wording. '
    'Return JSON only.'
)

USER_TEMPLATE = (
    'Return strict JSON with keys: '\
    'final_concepts (list of 10-18 canonical concepts), '\
    'families (object mapping family_name -> list of concepts), '\
    'drop_concepts (list of concepts to remove), '\
    'merge_examples (object mapping final concept -> list of merged variants), '\
    'rationale (short list of 3-6 bullets as strings). '\
    'Allowed families: agent_behavior, relative_motion, right_of_way_conflict, occlusion_visibility, road_layout_constraint, surface_weather, vulnerable_road_users, traffic_density_obstacle, imminent_crash_cue. '\
    'Current compact set:\n{current_blob}\n\nCandidate mined concepts:\n{candidate_blob}'
)


def extract_json_block(text: str):
    text = text.strip()
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
    raise ValueError('Could not extract JSON from adjudicator response')


def main():
    p = argparse.ArgumentParser(description='Final adjudication of discovered risk concepts using a stronger model')
    p.add_argument('--current_json', required=True)
    p.add_argument('--candidate_json', required=True)
    p.add_argument('--output_json', required=True)
    p.add_argument('--api_url', default=os.environ.get('LLM_API_URL', ''))
    p.add_argument('--api_key', default=os.environ.get('LLM_API_KEY', ''))
    p.add_argument('--model', default=os.environ.get('LLM_MODEL', 'qwen3.5-27b'))
    p.add_argument('--timeout', type=int, default=240)
    args = p.parse_args()

    current = json.load(open(args.current_json))
    candidate = json.load(open(args.candidate_json))
    current_blob = '\n'.join(f'- {c}' for c in current.get('standard_concepts', []))
    cand_list = candidate.get('concepts', [])
    candidate_blob = '\n'.join(f'- {c}' for c in cand_list)

    headers = {'Authorization': f'Bearer {args.api_key}', 'Content-Type': 'application/json'} if args.api_key else {'Content-Type': 'application/json'}
    payload = {
        'model': args.model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_TEMPLATE.format(current_blob=current_blob, candidate_blob=candidate_blob)}
        ],
        'temperature': 0.2,
    }
    r = requests.post(args.api_url, headers=headers, json=payload, timeout=args.timeout)
    r.raise_for_status()
    content = r.json()['choices'][0]['message']['content']
    if isinstance(content, list):
        content = ''.join(part.get('text', '') for part in content)
    parsed = extract_json_block(content)
    parsed['raw_content'] = content

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    print(json.dumps({'output_json': str(out_path), 'num_final': len(parsed.get('final_concepts', []))}, indent=2))


if __name__ == '__main__':
    main()
