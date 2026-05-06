#!/bin/bash
set -e
ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
OUT="$ROOT/output/concept_remake_api_v1"

python3 "$ROOT/concept_pipeline_sample_frames.py" \
  --datasets dad crash \
  --max_videos_per_dataset 120 \
  --frames_per_video 4 \
  --output_dir "$OUT"

python3 "$ROOT/concept_pipeline_generate_raw.py" \
  --backend api \
  --api_url "$VLM_API_URL" \
  --api_key "$VLM_API_KEY" \
  --model "$VLM_MODEL" \
  --frame_manifest "$OUT/frame_manifest.json" \
  --output_jsonl "$OUT/raw_concepts.jsonl"

python3 "$ROOT/concept_pipeline_refine.py" \
  --input_jsonl "$OUT/raw_concepts.jsonl" \
  --output_json "$OUT/refined_concepts.json"

python3 "$ROOT/concept_pipeline_cluster.py" \
  --input_json "$OUT/refined_concepts.json" \
  --output_json "$OUT/clustered_concepts.json" \
  --api_url "$LLM_API_URL" \
  --api_key "$LLM_API_KEY" \
  --model "$LLM_MODEL"

python3 "$ROOT/concept_pipeline_export.py" \
  --input_json "$OUT/clustered_concepts.json" \
  --output_txt "$OUT/all_concepts_discovered.txt"
