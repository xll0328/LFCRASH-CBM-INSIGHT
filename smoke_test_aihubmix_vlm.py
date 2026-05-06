#!/usr/bin/env python3
import os
import json
from pathlib import Path

from concept_pipeline_generate_raw import call_openai_like_api, resolve_api_key, resolve_api_url

FRAME = Path('/data/sony/LFCRASH/LFCRASH-CBM/output/concept_remake_large_vocab_v1/frames/dad__000015__f000016.jpg')
api_key = resolve_api_key('')
api_url = resolve_api_url('')
model = os.environ.get('VLM_MODEL', 'qwen3.5-122b-a10b')

print(json.dumps({
    'frame_exists': FRAME.exists(),
    'api_url': api_url,
    'api_key_present': bool(api_key),
    'model': model,
}, ensure_ascii=False, indent=2))

if not FRAME.exists():
    raise SystemExit('Sample frame not found')
if not api_key:
    raise SystemExit('No AIHUBMIX_API_KEY / VLM_API_KEY / OPENAI_API_KEY visible in this process')

out = call_openai_like_api(FRAME, api_url, api_key, model, timeout=180)
print(out['content'])
