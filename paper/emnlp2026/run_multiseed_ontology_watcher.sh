#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PY="/data/sony/anaconda3/bin/python"
SCRIPT="$ROOT/paper/emnlp2026/watch_multiseed_ontology_status.py"
LOG="$ROOT/output/emnlp2026_support/watch_multiseed_ontology_status.nohup.log"

setsid -f "$PY" "$SCRIPT" >"$LOG" 2>&1 < /dev/null
