#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_FRAMES="${MAX_FRAMES:-120}"
GPU_TOPM="${GPU_TOPM:-7}"
GPU_VERB="${GPU_VERB:-6}"

BASE_MANIFEST="${BASE_MANIFEST:-$ROOT/output/concept_remake_large_vocab_v2/frame_manifest.json}"
CONCEPT_META="${CONCEPT_META:-$ROOT/output/concept_sets/perfect_concept_set_v1.meta.json}"
SUPPORT_DIR="${SUPPORT_DIR:-$ROOT/output/emnlp2026_support}"

DAD_MANIFEST="$SUPPORT_DIR/dad${MAX_FRAMES}_frame_manifest.json"
TOPM_JSON="$SUPPORT_DIR/topm_pseudolabel_sensitivity_dad${MAX_FRAMES}.json"
VERB_JSON="$SUPPORT_DIR/concept_verbalization_sensitivity_dad${MAX_FRAMES}.json"

mkdir -p "$SUPPORT_DIR"

"$PYTHON_BIN" - <<'PY' "$BASE_MANIFEST" "$DAD_MANIFEST" "$MAX_FRAMES"
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
max_frames = int(sys.argv[3])

data = json.loads(src.read_text())
dad = [row for row in data if row.get("dataset") == "dad"][:max_frames]
dst.write_text(json.dumps(dad, indent=2))
print(f"[support] wrote {len(dad)} DAD frames to {dst}")
PY

echo "[support] running top-m pseudo-label sensitivity on GPU ${GPU_TOPM}"
"$PYTHON_BIN" "$ROOT/analyze_topm_pseudolabel_sensitivity.py" \
  --frame_manifest "$DAD_MANIFEST" \
  --concept_meta "$CONCEPT_META" \
  --topms 1 3 5 10 \
  --max_frames "$MAX_FRAMES" \
  --gpu "$GPU_TOPM" \
  --output_json "$TOPM_JSON"

echo "[support] running concept verbalization sensitivity on GPU ${GPU_VERB}"
"$PYTHON_BIN" "$ROOT/analyze_concept_verbalization_sensitivity.py" \
  --frame_manifest "$DAD_MANIFEST" \
  --gpu "$GPU_VERB" \
  --max_frames "$MAX_FRAMES" \
  --output_json "$VERB_JSON"

"$PYTHON_BIN" - <<'PY' "$TOPM_JSON" "$VERB_JSON"
import json
import sys
from pathlib import Path

topm = json.loads(Path(sys.argv[1]).read_text())
verb = json.loads(Path(sys.argv[2]).read_text())

topm_summary = topm["topm_summary"]
brief = {
    "topm_avg_family_diversity": {
        k: round(v["avg_family_diversity_per_frame"], 3)
        for k, v in topm_summary.items()
    },
    "topm_avg_relative_score_mass_vs_top20": {
        k: round(v["avg_relative_score_mass_vs_top20"], 3)
        for k, v in topm_summary.items()
    },
    "verbalization": {
        k: round(v, 4) for k, v in verb["aggregate"].items()
    },
}
print("[support] summary")
print(json.dumps(brief, indent=2))
PY

echo "[support] outputs:"
echo "  - $DAD_MANIFEST"
echo "  - $TOPM_JSON"
echo "  - $VERB_JSON"
