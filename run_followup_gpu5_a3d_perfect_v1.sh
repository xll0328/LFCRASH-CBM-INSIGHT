#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONCEPT_FILE="$ROOT/output/concept_sets/perfect_concept_set_v1.txt"
WAIT_LOG="$ROOT/output/a3d_ac/followup_gpu5_a3d_perfect_v1.log"
TARGET_QUEUE_LOG="$ROOT/output/a3d_ac/queue_gpu5_a3d_modules_perfect_v1.log"
QUEUE_DONE_MARKER="Queue done GPU5 A3D modules perfect_v1"
QUEUE_STALE_SEC="${QUEUE_STALE_SEC:-21600}"

mkdir -p "$ROOT/output/a3d_ac"

echo "[$(date '+%F %T')] Follow-up watcher start for GPU5 A3D perfect_v1" > "$WAIT_LOG"
while ! grep -q "$QUEUE_DONE_MARKER" "$TARGET_QUEUE_LOG" 2>/dev/null; do
  if [[ -f "$TARGET_QUEUE_LOG" ]]; then
    queue_age_sec=$(( $(date +%s) - $(stat -c %Y "$TARGET_QUEUE_LOG") ))
    if (( queue_age_sec >= QUEUE_STALE_SEC )); then
      last_started_tag="$(grep 'START ' "$TARGET_QUEUE_LOG" | tail -n 1 | sed 's/.*START //')"
      echo "[$(date '+%F %T')] Queue log stale for ${queue_age_sec}s without done marker; last_started=${last_started_tag:-unknown}. Exiting watcher." >> "$WAIT_LOG"
      exit 0
    fi
  fi
  echo "[$(date '+%F %T')] Waiting for GPU5 base queue to finish..." >> "$WAIT_LOG"
  sleep 300
done

echo "[$(date '+%F %T')] GPU5 base queue finished. Launching focused follow-up." | tee -a "$WAIT_LOG"

run_job() {
  local tag="$1"
  shift
  local out_dir="$ROOT/output/a3d_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] START $tag" | tee -a "$WAIT_LOG"
  "$PYTHON_BIN" "$ROOT/train_multi.py" "$@" > "$out_dir/nohup.log" 2>&1
  local ec=$?
  echo "[$(date '+%F %T')] END $tag exit=$ec" | tee -a "$WAIT_LOG"
}

run_job "a3d_ac_perfect_v1_minuscgta_lowac_followup_q1" \
  --dataset a3d --gpu 5 --tag a3d_ac_perfect_v1_minuscgta_lowac_followup_q1 \
  --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
  --batch_size 16 --lr 3e-4 --h_dim 256 --disable_cgta \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.05 --lambda_ac_value 0.1 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" \
  --eval_every 5 --num_workers 4

run_job "a3d_ac_perfect_v1_minuscgta_h384_followup_q2" \
  --dataset a3d --gpu 5 --tag a3d_ac_perfect_v1_minuscgta_h384_followup_q2 \
  --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
  --batch_size 16 --lr 3e-4 --h_dim 384 --disable_cgta \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" \
  --eval_every 5 --num_workers 4

run_job "a3d_ac_perfect_v1_minuscgta_nosparse_followup_q3" \
  --dataset a3d --gpu 5 --tag a3d_ac_perfect_v1_minuscgta_nosparse_followup_q3 \
  --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
  --batch_size 16 --lr 3e-4 --h_dim 256 --disable_cgta \
  --lambda_align 0 --lambda_sparse 0 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" \
  --eval_every 5 --num_workers 4

echo "[$(date '+%F %T')] GPU5 A3D focused follow-up done" | tee -a "$WAIT_LOG"
