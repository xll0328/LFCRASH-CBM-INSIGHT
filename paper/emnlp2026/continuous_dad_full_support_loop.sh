#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
OUT="$ROOT/output/dad_full_support_block"
ARCHIVE_ROOT="$ROOT/output/dad_full_support_block_archive"
SUPPORT_DIR="$ROOT/output/emnlp2026_support"
PYTHON_BIN="${PYTHON_BIN:-/data/sony/anaconda3/bin/python}"
GPUS="${GPUS:-0,3,4}"
STALE_AFTER_MIN="${STALE_AFTER_MIN:-35}"
POLL_SEC="${POLL_SEC:-300}"
MAX_RESTARTS_PER_TAG="${MAX_RESTARTS_PER_TAG:-4}"
LOCK_DIR="$SUPPORT_DIR/.continuous_dad_full_support_loop.lock"
LOCK_PID_FILE="$LOCK_DIR/pid"
STATUS_LOG="$SUPPORT_DIR/continuous_dad_full_support_loop.log"

mkdir -p "$OUT" "$ARCHIVE_ROOT" "$SUPPORT_DIR"

cleanup_lock() {
  rm -f "$LOCK_PID_FILE"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}

if [[ -d "$LOCK_DIR" ]]; then
  existing_pid=""
  if [[ -f "$LOCK_PID_FILE" ]]; then
    existing_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null || true)"
  fi
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "[loop] another watchdog instance appears to be running: $LOCK_DIR (pid=$existing_pid)"
    exit 1
  fi
  rm -f "$LOCK_PID_FILE"
  if ! rmdir "$LOCK_DIR" 2>/dev/null; then
    echo "[loop] stale lock exists but could not be removed: $LOCK_DIR"
    exit 1
  fi
fi
mkdir "$LOCK_DIR"
echo "$$" > "$LOCK_PID_FILE"
trap cleanup_lock EXIT

timestamp() {
  date -u +"%Y%m%dT%H%M%SZ"
}

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%d %H:%M:%S UTC')" "$*" | tee -a "$STATUS_LOG"
}

count_completed() {
  find "$OUT" -maxdepth 2 -path "$OUT/insight_journal_dad_full_r*/results.json" | wc -l
}

restart_count_for_tag() {
  local tag="$1"
  find "$ARCHIVE_ROOT" -maxdepth 1 -type d -name "${tag}_*" | wc -l
}

tag_live_pids() {
  local tag="$1"
  ps -eo pid=,comm=,args= | awk -v root="$ROOT/train.py" -v tag_pat="--tag ${tag}" '
    ($2 == "python" || $2 == "python3") && index($0, root) && index($0, tag_pat) { print $1 }
  '
}

archive_stale_runs() {
  shopt -s nullglob
  for run_dir in "$OUT"/insight_journal_dad_full_r*; do
    [[ -d "$run_dir" ]] || continue
    local tag log_path age_min restarts dest live_pids live_csv
    tag="$(basename "$run_dir")"
    log_path="$run_dir/train.log"

    if [[ -f "$run_dir/results.json" ]]; then
      continue
    fi
    if [[ ! -f "$log_path" ]]; then
      continue
    fi

    age_min="$((( $(date +%s) - $(stat -c %Y "$log_path") ) / 60))"
    if (( age_min < STALE_AFTER_MIN )); then
      continue
    fi

    live_pids="$(tag_live_pids "$tag")"
    if [[ -n "$live_pids" ]]; then
      live_csv="$(echo "$live_pids" | paste -sd, -)"
      log "stale-by-log but live process exists; leaving in place: tag=$tag age_min=$age_min pids=$live_csv"
      continue
    fi

    restarts="$(restart_count_for_tag "$tag")"
    if (( restarts >= MAX_RESTARTS_PER_TAG )); then
      log "stale run reached restart cap; leaving in place: tag=$tag age_min=$age_min cap=$MAX_RESTARTS_PER_TAG"
      continue
    fi

    dest="$ARCHIVE_ROOT/${tag}_stale_${age_min}min_$(timestamp)"
    mv "$run_dir" "$dest"
    log "archived stale run: tag=$tag age_min=$age_min dest=$dest"
  done
  shopt -u nullglob
}

refresh_status() {
  if "$PYTHON_BIN" "$ROOT/paper/emnlp2026/refresh_emnlp_status.py" >>"$STATUS_LOG" 2>&1; then
    log "refreshed EMNLP status artifacts"
  else
    log "status refresh returned non-zero; see $STATUS_LOG"
  fi
}

launch_missing_runs() {
  log "launching missing runs on gpus=$GPUS stale_after_min=$STALE_AFTER_MIN"
  bash "$ROOT/paper/emnlp2026/run_dad_full_support_block.sh" \
    --execute \
    --gpus "$GPUS" \
    --stale-after-min "$STALE_AFTER_MIN" >>"$STATUS_LOG" 2>&1
}

summarize_live_state() {
  if [[ -f "$SUPPORT_DIR/dad_hardening_status.json" ]]; then
    "$PYTHON_BIN" - <<'PY' >>"$STATUS_LOG" 2>&1
import json
from pathlib import Path
p = Path("/data/sony/LFCRASH/LFCRASH-CBM/output/emnlp2026_support/dad_hardening_status.json")
obj = json.loads(p.read_text())
block = obj.get("matched_full_support_block", {})
rows = block.get("in_progress_rows", [])
agg = block.get("in_progress_latest_eval_aggregate")
print("[live-summary] completed=", block.get("num_completed"), "in_progress=", len(rows))
if agg:
    ap = agg["AP"]["mean"] * 100.0
    ap_std = agg["AP"]["std"] * 100.0
    mtta = agg["mTTA"]["mean"]
    epoch_mode = agg.get("epoch_mode")
    print(f"[live-summary] epoch={epoch_mode} ap={ap:.2f}%±{ap_std:.2f} mTTA={mtta:.2f}s")
for row in rows:
    latest = row.get("latest_eval")
    if latest:
        print(f"[live-summary] {row['tag']} epoch={latest['epoch']} AP={latest['AP']*100:.2f}% mTTA={latest['mTTA']:.2f}s")
    else:
        print(f"[live-summary] {row['tag']} no_eval_yet")
PY
  fi
}

log "watchdog start: gpus=$GPUS stale_after_min=$STALE_AFTER_MIN poll_sec=$POLL_SEC"
refresh_status

while true; do
  completed="$(count_completed)"
  if (( completed >= 3 )); then
    log "matched full support block completed: results_json_count=$completed"
    refresh_status
    summarize_live_state
    break
  fi

  archive_stale_runs
  launch_missing_runs
  refresh_status
  summarize_live_state

  completed="$(count_completed)"
  if (( completed >= 3 )); then
    log "matched full support block completed after refresh: results_json_count=$completed"
    break
  fi

  log "sleeping for ${POLL_SEC}s before next poll"
  sleep "$POLL_SEC"
done

log "watchdog exit"
