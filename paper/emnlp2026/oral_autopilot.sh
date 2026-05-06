#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SUPPORT_DIR="$ROOT/output/emnlp2026_support"
LOG="$SUPPORT_DIR/oral_autopilot.log"
PIDFILE="$SUPPORT_DIR/oral_autopilot.pid"

WAIT_TAGS=(
  "a3d_shared_perfect_v1_s123"
  "a3d_shared_perfect_v1_s3407"
)

MAX_FRAMES="${MAX_FRAMES:-500}"
GPU_TOPM="${GPU_TOPM:-2}"
GPU_VERB="${GPU_VERB:-3}"
POLL_SEC="${POLL_SEC:-180}"
FORCE_SUPPORT="${FORCE_SUPPORT:-0}"

DAD_MANIFEST="$SUPPORT_DIR/dad${MAX_FRAMES}_frame_manifest.json"
TOPM_JSON="$SUPPORT_DIR/topm_pseudolabel_sensitivity_dad${MAX_FRAMES}.json"
VERB_JSON="$SUPPORT_DIR/concept_verbalization_sensitivity_dad${MAX_FRAMES}.json"
HUMAN_AUDIT_MD="$SUPPORT_DIR/human_ontology_audit_summary.md"

mkdir -p "$SUPPORT_DIR"

log() {
  local msg="[$(date '+%F %T')] $*"
  echo "$msg" | tee -a "$LOG"
}

cleanup() {
  local rc=$?
  rm -f "$PIDFILE"
  log "oral autopilot exit rc=$rc"
}

trap cleanup EXIT
echo "$$" > "$PIDFILE"

tags_running() {
  local tag
  for tag in "${WAIT_TAGS[@]}"; do
    if ps -eo cmd | grep -F "$tag" | grep -v grep >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

log "oral autopilot start"
log "waiting on tags: ${WAIT_TAGS[*]}"

while tags_running; do
  log "still waiting for active A3D perfect_v1 multiseed runs to finish"
  sleep "$POLL_SEC"
done

log "A3D perfect_v1 multiseed runs finished; refreshing EMNLP status"
"$PYTHON_BIN" "$ROOT/paper/emnlp2026/refresh_emnlp_status.py" | tee -a "$LOG"

if [[ "$FORCE_SUPPORT" == "1" || ! -f "$DAD_MANIFEST" || ! -f "$TOPM_JSON" || ! -f "$VERB_JSON" ]]; then
  log "running expanded EMNLP support analyses with MAX_FRAMES=$MAX_FRAMES on GPUs $GPU_TOPM/$GPU_VERB"
  MAX_FRAMES="$MAX_FRAMES" GPU_TOPM="$GPU_TOPM" GPU_VERB="$GPU_VERB" \
    "$ROOT/paper/emnlp2026/run_emnlp_support_analyses.sh" | tee -a "$LOG"
else
  log "expanded EMNLP support analyses already present for MAX_FRAMES=$MAX_FRAMES; skipping rerun"
fi

if [[ "$FORCE_SUPPORT" == "1" || ! -f "$HUMAN_AUDIT_MD" ]]; then
  log "summarizing human ontology audit"
  "$PYTHON_BIN" "$ROOT/paper/emnlp2026/summarize_human_ontology_audit.py" | tee -a "$LOG"
else
  log "human ontology audit summary already present; skipping rerun"
fi

log "refreshing EMNLP status after expanded audits"
"$PYTHON_BIN" "$ROOT/paper/emnlp2026/refresh_emnlp_status.py" | tee -a "$LOG"

log "oral autopilot done"
