#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONCEPT_FILE="$ROOT/output/concept_sets/perfect_concept_set_v1.txt"
QLOG="$ROOT/output/dad_ac/queue_gpu3_dad_advanced_perfect_v1.log"

run_job() {
  local tag="$1"
  shift
  local out_dir="$ROOT/output/dad_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] START $tag" | tee -a "$QLOG"
  "$PYTHON_BIN" "$ROOT/train_dad_ac.py" "$@" > "$out_dir/nohup.log" 2>&1
  local ec=$?
  echo "[$(date '+%F %T')] END $tag exit=$ec" | tee -a "$QLOG"
  return $ec
}

echo "[$(date '+%F %T')] Queue restart GPU3 DAD advanced perfect_v1" >> "$QLOG"

run_job "dad_ac_perfect_v1_h384_q1" \
  --gpu 3 --tag dad_ac_perfect_v1_h384_q1 \
  --epochs 200 --warmup_epochs 20 --cbm_ramp_epochs 30 \
  --batch_size 16 --lr 3e-4 --h_dim 384 --z_dim 384 \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 --ac_gamma 0.95 --ac_entropy 0.01 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" \
  --eval_every 5 --num_workers 4

run_job "dad_ac_perfect_v1_h384_q2" \
  --gpu 3 --tag dad_ac_perfect_v1_h384_q2 \
  --epochs 200 --warmup_epochs 20 --cbm_ramp_epochs 30 \
  --batch_size 16 --lr 3e-4 --h_dim 384 --z_dim 384 \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 --ac_gamma 0.95 --ac_entropy 0.01 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" \
  --eval_every 5 --num_workers 4

run_job "dad_ac_perfect_v1_aligned_q3" \
  --gpu 3 --tag dad_ac_perfect_v1_aligned_q3 \
  --epochs 200 --warmup_epochs 20 --cbm_ramp_epochs 30 \
  --batch_size 16 --lr 3e-4 --h_dim 256 --z_dim 256 \
  --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 --ac_gamma 0.95 --ac_entropy 0.01 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" \
  --eval_every 5 --num_workers 4

echo "[$(date '+%F %T')] Queue done GPU3 DAD advanced perfect_v1" | tee -a "$QLOG"
