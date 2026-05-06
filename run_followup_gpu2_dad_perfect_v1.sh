#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONCEPT_FILE="$ROOT/output/concept_sets/perfect_concept_set_v1.txt"
WAIT_LOG="$ROOT/output/dad_ac/followup_gpu2_dad_perfect_v1.log"
TARGET_QUEUE_LOG="$ROOT/output/dad_ac/queue_gpu2_dad_multi_perfect_v1.log"

mkdir -p "$ROOT/output/dad_ac"

echo "[$(date '+%F %T')] Follow-up watcher start for GPU2 DAD perfect_v1" > "$WAIT_LOG"
while ! grep -q "Queue done GPU2 DAD multi perfect_v1" "$TARGET_QUEUE_LOG" 2>/dev/null; do
  echo "[$(date '+%F %T')] Waiting for GPU2 base queue to finish..." >> "$WAIT_LOG"
  sleep 300
done

echo "[$(date '+%F %T')] GPU2 base queue finished. Launching focused follow-up." | tee -a "$WAIT_LOG"

run_job() {
  local tag="$1"
  shift
  local out_dir="$ROOT/output/dad_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] START $tag" | tee -a "$WAIT_LOG"
  "$PYTHON_BIN" "$ROOT/train_multi.py" "$@" > "$out_dir/nohup.log" 2>&1
  local ec=$?
  echo "[$(date '+%F %T')] END $tag exit=$ec" | tee -a "$WAIT_LOG"
}

run_job "dad_multi_perfect_v1_lowac_followup_q1" \
  --dataset dad --gpu 2 --tag dad_multi_perfect_v1_lowac_followup_q1 \
  --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
  --batch_size 16 --lr 3e-4 --h_dim 256 \
  --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.1 --lambda_ac_value 0.1 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" \
  --eval_every 5 --num_workers 4

run_job "dad_multi_perfect_v1_noalign_followup_q2" \
  --dataset dad --gpu 2 --tag dad_multi_perfect_v1_noalign_followup_q2 \
  --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
  --batch_size 16 --lr 3e-4 --h_dim 256 \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" \
  --eval_every 5 --num_workers 4

run_job "dad_multi_perfect_v1_h384_followup_q3" \
  --dataset dad --gpu 2 --tag dad_multi_perfect_v1_h384_followup_q3 \
  --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
  --batch_size 16 --lr 3e-4 --h_dim 384 \
  --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" \
  --eval_every 5 --num_workers 4

echo "[$(date '+%F %T')] GPU2 DAD focused follow-up done" | tee -a "$WAIT_LOG"
