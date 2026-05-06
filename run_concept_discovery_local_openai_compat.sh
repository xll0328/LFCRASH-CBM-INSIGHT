#!/bin/bash
set -e
ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
OUT="$ROOT/output/concept_remake_local_openai_v1"

python3 "$ROOT/concept_pipeline_sample_frames.py" \
  --datasets dad crash \
  --max_videos_per_dataset 120 \
  --frames_per_video 4 \
  --output_dir "$OUT"

# Use this script when you have a LOCAL OpenAI-compatible multimodal server running,
# e.g. vLLM / SGLang / LM Studio / local gateway.
python3 "$ROOT/concept_pipeline_generate_raw.py" \
  --backend api \
  --api_url "$LOCAL_VLM_API_URL" \
  --api_key "$LOCAL_VLM_API_KEY" \
  --model "$LOCAL_VLM_MODEL" \
  --frame_manifest "$OUT/frame_manifest.json" \
  --output_jsonl "$OUT/raw_concepts.jsonl"

python3 "$ROOT/concept_pipeline_refine.py" \
  --input_jsonl "$OUT/raw_concepts.jsonl" \
  --output_json "$OUT/refined_concepts.json"
