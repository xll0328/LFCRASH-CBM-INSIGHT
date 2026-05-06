#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"

python3 "$ROOT/paper/emnlp2026/audit_dad_ontology_seed_extension.py"
python3 "$ROOT/paper/emnlp2026/audit_ontology_size_matched_controls.py"
python3 "$ROOT/paper/emnlp2026/summarize_ontology_size_matched_effects.py"

echo "[refresh] wrote:"
echo "  output/emnlp2026_support/dad_ontology_seed_extension_status.md"
echo "  output/emnlp2026_support/ontology_size_matched_status.md"
echo "  output/emnlp2026_support/ontology_size_matched_effects.md"
