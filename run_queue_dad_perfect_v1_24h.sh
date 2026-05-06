#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONCEPT_FILE="$ROOT/output/concept_sets/perfect_concept_set_v1.txt"
QUEUE_LOG="$ROOT/output/dad_ac/queue_dad_perfect_v1_24h.log"

run_job() {
  local tag="$1"
  shift
  local out_dir="$ROOT/output/dad_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] START $tag" | tee -a "$QUEUE_LOG"
  "$PYTHON_BIN" "$ROOT/train_dad_ac.py" "$@" > "$out_dir/nohup.log" 2>&1
  echo "[$(date '+%F %T')] END $tag exit=$?" | tee -a "$QUEUE_LOG"
}

mkdir -p "$ROOT/output/dad_ac"
echo "[$(date '+%F %T')] Queue start: DAD perfect_v1 24h plan" > "$QUEUE_LOG"

run_job "dad_ac_perfect_v1_noalign_tunednw4_q1" \
  --gpu 0 \
  --tag dad_ac_perfect_v1_noalign_tunednw4_q1 \
  --epochs 200 \
  --warmup_epochs 20 \
  --cbm_ramp_epochs 30 \
  --batch_size 16 \
  --lr 3e-4 \
  --h_dim 256 \
  --z_dim 256 \
  --lambda_align 0 \
  --lambda_sparse 5e-5 \
  --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 \
  --lambda_ac_value 0.3 \
  --ac_gamma 0.95 \
  --ac_entropy 0.01 \
  --num_concepts 80 \
  --concept_file "$CONCEPT_FILE" \
  --eval_every 5 \
  --num_workers 4

run_job "dad_ac_perfect_v1_v4final_q2" \
  --gpu 0 \
  --tag dad_ac_perfect_v1_v4final_q2 \
  --epochs 200 \
  --warmup_epochs 20 \
  --cbm_ramp_epochs 30 \
  --batch_size 16 \
  --lr 3e-4 \
  --h_dim 256 \
  --lambda_ac_policy 0.3 \
  --lambda_ac_value 0.3 \
  --ac_gamma 0.95 \
  --ac_entropy 0.01 \
  --num_concepts 80 \
  --concept_file "$CONCEPT_FILE" \
  --eval_every 5 \
  --num_workers 4

run_job "dad_ac_perfect_v1_v5early_q3" \
  --gpu 0 \
  --tag dad_ac_perfect_v1_v5early_q3 \
  --epochs 200 \
  --warmup_epochs 20 \
  --cbm_ramp_epochs 30 \
  --batch_size 16 \
  --lr 3e-4 \
  --h_dim 256 \
  --lambda_ac_policy 0.1 \
  --lambda_ac_value 0.2 \
  --ac_gamma 0.95 \
  --ac_entropy 0.01 \
  --num_concepts 80 \
  --concept_file "$CONCEPT_FILE" \
  --eval_every 5 \
  --num_workers 4

echo "[$(date '+%F %T')] Queue done: DAD perfect_v1 24h plan" | tee -a "$QUEUE_LOG"
