#!/bin/bash
set -e
ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
OUT="$ROOT/output/concept_remake_local_vlm"

python3 "$ROOT/concept_pipeline_sample_frames.py" \
  --datasets dad crash \
  --max_videos_per_dataset 50 \
  --frames_per_video 4 \
  --output_dir "$OUT"

# Example local VLM models via Ollama:
#   ollama pull llava:13b
#   ollama pull qwen2.5vl:7b
# Then run one of them locally before executing below.
python3 "$ROOT/concept_pipeline_generate_raw.py" \
  --backend ollama \
  --api_url "http://127.0.0.1:11434/api/generate" \
  --model "qwen2.5vl:7b" \
  --frame_manifest "$OUT/frame_manifest.json" \
  --output_jsonl "$OUT/raw_concepts.jsonl"

python3 "$ROOT/concept_pipeline_refine.py" \
  --input_jsonl "$OUT/raw_concepts.jsonl" \
  --output_json "$OUT/refined_concepts.json"

# Optional: if you also have a local or remote text LLM endpoint for clustering.
# python3 "$ROOT/concept_pipeline_cluster.py" \
#   --input_json "$OUT/refined_concepts.json" \
#   --output_json "$OUT/clustered_concepts.json" \
#   --api_url "http://127.0.0.1:11434/v1/chat/completions" \
#   --model "qwen2.5:14b"
