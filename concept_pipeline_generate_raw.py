#!/usr/bin/env python3
import os
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests

DEFAULT_OPENAI_COMPAT_URL = 'https://aihubmix.com/v1/chat/completions'

SYSTEM_PROMPT = (
    'You are a traffic safety analyst building a paper-quality concept ontology for accident anticipation. '
    'Extract concise risk primitives from a single driving frame. Do not narrate the whole scene. '
    'Focus on transferable causes of accidents: agent behavior, relative motion conflict, right-of-way conflict, '
    'visibility/occlusion problems, road-layout constraints, road-surface/weather hazards, vulnerable road users, '
    'traffic density/obstacles, and imminent-crash cues. Prefer short canonical noun phrases that could transfer across datasets. '
    'Avoid colors, vehicle identity details, and frame-specific trivia unless they directly change risk. Return JSON only.'
)

USER_PROMPT = (
    'Analyze this frame for accident anticipation. Output strict JSON with keys: '\
    'raw_concepts (8-16 short canonical phrases), '\
    'risk_factors (4-8 highest-priority risk primitives), '\
    'family_tags (list chosen only from: agent_behavior, relative_motion, right_of_way_conflict, occlusion_visibility, road_layout_constraint, surface_weather, vulnerable_road_users, traffic_density_obstacle, imminent_crash_cue), '\
    'scene_summary (one short sentence). '\
    'Requirements: prefer reusable risk concepts, no long sentences in concept lists, no duplicates, no object-color trivia.'
)


def encode_image(path: Path) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def resolve_api_url(cli_value: str) -> str:
    return cli_value or os.environ.get('AIHUBMIX_BASE_URL', '') or os.environ.get('VLM_API_URL', '') or DEFAULT_OPENAI_COMPAT_URL


def resolve_api_key(cli_value: str) -> str:
    return cli_value or os.environ.get('AIHUBMIX_API_KEY', '') or os.environ.get('VLM_API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')


def call_openai_like_api(image_path: Path, api_url: str, api_key: str, model: str, timeout: int) -> Dict[str, Any]:
    image_b64 = encode_image(image_path)
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'} if api_key else {'Content-Type': 'application/json'}
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': USER_PROMPT},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}}
            ]}
        ],
        'temperature': 0.2,
    }
    r = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data['choices'][0]['message']['content']
    if isinstance(content, list):
        content = ''.join(part.get('text', '') for part in content)
    return {'raw_response': data, 'content': content}


def call_ollama(image_path: Path, api_url: str, model: str, timeout: int) -> Dict[str, Any]:
    image_b64 = encode_image(image_path)
    prompt = SYSTEM_PROMPT + '\n\n' + USER_PROMPT
    payload = {
        'model': model,
        'prompt': prompt,
        'images': [image_b64],
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0.2},
    }
    r = requests.post(api_url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data.get('response', '')
    return {'raw_response': data, 'content': content}


def process_item(item: Dict[str, Any], backend: str, api_url: str, api_key: str, model: str, timeout: int) -> Dict[str, Any]:
    result = {
        'dataset': item['dataset'],
        'video_path': item['video_path'],
        'frame_index': item['frame_index'],
        'frame_path': item['frame_path'],
        'backend': backend,
        'model': model,
    }
    if backend == 'stub':
        result['status'] = 'stubbed'
        result['content'] = json.dumps({
            'raw_concepts': [],
            'risk_factors': [],
            'family_tags': [],
            'scene_summary': 'No VLM backend configured yet.'
        })
        return result

    try:
        if backend == 'api':
            vlm_out = call_openai_like_api(Path(item['frame_path']), api_url, api_key, model, timeout)
        else:
            vlm_out = call_ollama(Path(item['frame_path']), api_url, model, timeout)
        result['status'] = 'ok'
        result['content'] = vlm_out['content']
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    return result


def main():
    p = argparse.ArgumentParser(description='Generate raw concepts from sampled driving frames via VLM API or local backend')
    p.add_argument('--frame_manifest', required=True)
    p.add_argument('--output_jsonl', required=True)
    p.add_argument('--backend', choices=['stub', 'api', 'ollama'], default=os.environ.get('VLM_BACKEND', 'stub'))
    p.add_argument('--api_url', default='')
    p.add_argument('--api_key', default='')
    p.add_argument('--model', default=os.environ.get('VLM_MODEL', 'qwen3.5-122b-a10b'))
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--sleep', type=float, default=0.2)
    p.add_argument('--max_items', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--resume', action='store_true', default=True)
    args = p.parse_args()

    manifest = json.load(open(args.frame_manifest))
    if args.max_items is not None:
        manifest = manifest[:args.max_items]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results = []
    completed_keys = set()
    if args.resume and out_path.exists():
        with open(out_path) as rf:
            for line in rf:
                if not line.strip():
                    continue
                obj = json.loads(line)
                existing_results.append(obj)
                completed_keys.add((obj.get('frame_path'), obj.get('frame_index')))
        manifest = [
            item for item in manifest
            if (item['frame_path'], item['frame_index']) not in completed_keys
        ]

    backend = args.backend
    args.api_url = resolve_api_url(args.api_url)
    args.api_key = resolve_api_key(args.api_key)
    if backend == 'api' and (not args.api_url or not args.api_key):
        backend = 'stub'
    if backend == 'ollama' and not args.api_url:
        args.api_url = 'http://127.0.0.1:11434/api/generate'

    mode = 'a' if existing_results else 'w'
    write_lock = Lock()

    with open(out_path, mode) as wf:
        if backend == 'stub':
            for item in manifest:
                result = process_item(item, backend, args.api_url, args.api_key, args.model, args.timeout)
                wf.write(json.dumps(result, ensure_ascii=False) + '\n')
        else:
            with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
                futures = [
                    ex.submit(process_item, item, backend, args.api_url, args.api_key, args.model, args.timeout)
                    for item in manifest
                ]
                for fut in as_completed(futures):
                    result = fut.result()
                    with write_lock:
                        wf.write(json.dumps(result, ensure_ascii=False) + '\n')
                        wf.flush()
                    if args.sleep > 0:
                        time.sleep(args.sleep)

    total_written = len(existing_results) + len(manifest)
    print(json.dumps({'output_jsonl': str(out_path), 'num_items': total_written, 'backend': backend}, indent=2))


if __name__ == '__main__':
    main()
