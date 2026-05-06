#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONCEPT_FILE="$ROOT/output/concept_sets/perfect_concept_set_v1.txt"
QUEUE_LOG="$ROOT/output/a3d_ac/queue_a3d_perfect_v1_24h.log"

run_job() {
  local tag="$1"
  shift
  local out_dir="$ROOT/output/a3d_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] START $tag" | tee -a "$QUEUE_LOG"
  "$PYTHON_BIN" "$ROOT/train_multi.py" "$@" > "$out_dir/nohup.log" 2>&1
  echo "[$(date '+%F %T')] END $tag exit=$?" | tee -a "$QUEUE_LOG"
}

mkdir -p "$ROOT/output/a3d_ac"
echo "[$(date '+%F %T')] Queue start: A3D perfect_v1 24h plan" > "$QUEUE_LOG"

run_job "a3d_ac_perfect_v1_noalign_tunednw4_q1" \
  --dataset a3d \
  --gpu 1 \
  --tag a3d_ac_perfect_v1_noalign_tunednw4_q1 \
  --epochs 150 \
  --warmup_epochs 15 \
  --cbm_ramp_epochs 20 \
  --batch_size 16 \
  --lr 3e-4 \
  --h_dim 256 \
  --lambda_align 0 \
  --lambda_sparse 5e-5 \
  --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 \
  --lambda_ac_value 0.3 \
  --num_concepts 80 \
  --concept_file "$CONCEPT_FILE" \
  --eval_every 5 \
  --num_workers 4

run_job "a3d_ac_perfect_v1_lowac_q2" \
  --dataset a3d \
  --gpu 1 \
  --tag a3d_ac_perfect_v1_lowac_q2 \
  --epochs 150 \
  --warmup_epochs 15 \
  --cbm_ramp_epochs 20 \
  --batch_size 16 \
  --lr 3e-4 \
  --h_dim 256 \
  --lambda_align 0 \
  --lambda_sparse 5e-5 \
  --lambda_recon 1e-4 \
  --lambda_ac_policy 0.05 \
  --lambda_ac_value 0.1 \
  --num_concepts 80 \
  --concept_file "$CONCEPT_FILE" \
  --eval_every 5 \
  --num_workers 4

run_job "a3d_ac_perfect_v1_plainac_q3" \
  --dataset a3d \
  --gpu 1 \
  --tag a3d_ac_perfect_v1_plainac_q3 \
  --epochs 150 \
  --warmup_epochs 15 \
  --cbm_ramp_epochs 20 \
  --batch_size 16 \
  --lr 3e-4 \
  --h_dim 256 \
  --disable_cgta \
  --disable_crs \
  --lambda_align 0 \
  --lambda_sparse 5e-5 \
  --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 \
  --lambda_ac_value 0.3 \
  --num_concepts 80 \
  --concept_file "$CONCEPT_FILE" \
  --eval_every 5 \
  --num_workers 4

echo "[$(date '+%F %T')] Queue done: A3D perfect_v1 24h plan" | tee -a "$QUEUE_LOG"
