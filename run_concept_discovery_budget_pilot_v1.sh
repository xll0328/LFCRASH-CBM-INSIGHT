#!/bin/bash
set -e
ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
OUT="$ROOT/output/concept_remake_budget_pilot_v1"

python3 "$ROOT/concept_pipeline_sample_frames.py" \
  --datasets dad crash \
  --max_videos_per_dataset 12 \
  --frames_per_video 2 \
  --output_dir "$OUT"

python3 "$ROOT/concept_pipeline_generate_raw.py" \
  --backend api \
  --api_url "$VLM_API_URL" \
  --api_key "$VLM_API_KEY" \
  --model "qwen3-vl-flash-2026-01-22" \
  --frame_manifest "$OUT/frame_manifest.json" \
  --output_jsonl "$OUT/raw_concepts.jsonl" \
  --sleep 0.4

python3 "$ROOT/concept_pipeline_refine.py" \
  --input_jsonl "$OUT/raw_concepts.jsonl" \
  --output_json "$OUT/refined_concepts.json" \
  --min_count 2 \
  --min_score 2
