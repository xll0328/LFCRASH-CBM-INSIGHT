#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONCEPT_FILE="$ROOT/output/concept_sets/perfect_concept_set_v1.txt"
MASTER_LOG="$ROOT/output/perfect_v1_short_horizon_master.log"
mkdir -p "$ROOT/output/dad_ac" "$ROOT/output/a3d_ac"

ts() { date '+%F %T'; }

run_dad() {
  local tag="$1"
  shift
  local out_dir="$ROOT/output/dad_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(ts)] START $tag" | tee -a "$MASTER_LOG"
  "$PYTHON_BIN" "$ROOT/train_dad_ac.py" "$@" > "$out_dir/nohup.log" 2>&1
  echo "[$(ts)] END $tag exit=$?" | tee -a "$MASTER_LOG"
}

run_a3d() {
  local tag="$1"
  shift
  local out_dir="$ROOT/output/a3d_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(ts)] START $tag" | tee -a "$MASTER_LOG"
  "$PYTHON_BIN" "$ROOT/train_multi.py" "$@" > "$out_dir/nohup.log" 2>&1
  echo "[$(ts)] END $tag exit=$?" | tee -a "$MASTER_LOG"
}

echo "[$(ts)] Launching short-horizon focused perfect_v1 search" > "$MASTER_LOG"

(run_dad "dad_ac_perfect_v1_h384_short_q1" \
  --gpu 0 --tag dad_ac_perfect_v1_h384_short_q1 \
  --epochs 80 --warmup_epochs 15 --cbm_ramp_epochs 20 \
  --batch_size 16 --lr 3e-4 --h_dim 384 --z_dim 384 \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 --ac_gamma 0.95 --ac_entropy 0.01 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_dad "dad_ac_perfect_v1_h384_lowac_short_q2" \
  --gpu 2 --tag dad_ac_perfect_v1_h384_lowac_short_q2 \
  --epochs 80 --warmup_epochs 15 --cbm_ramp_epochs 20 \
  --batch_size 16 --lr 3e-4 --h_dim 384 --z_dim 384 \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.1 --lambda_ac_value 0.1 --ac_gamma 0.95 --ac_entropy 0.01 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_a3d "dad_multi_perfect_v1_h384_short_q3" \
  --dataset dad --gpu 3 --tag dad_multi_perfect_v1_h384_short_q3 \
  --epochs 60 --warmup_epochs 10 --cbm_ramp_epochs 15 \
  --batch_size 16 --lr 3e-4 --h_dim 384 \
  --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_a3d "a3d_ac_perfect_v1_minuscgta_short_q1" \
  --dataset a3d --gpu 1 --tag a3d_ac_perfect_v1_minuscgta_short_q1 \
  --epochs 60 --warmup_epochs 10 --cbm_ramp_epochs 15 \
  --batch_size 16 --lr 3e-4 --h_dim 256 --disable_cgta \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_a3d "a3d_ac_perfect_v1_minuscgta_h384_short_q2" \
  --dataset a3d --gpu 5 --tag a3d_ac_perfect_v1_minuscgta_h384_short_q2 \
  --epochs 60 --warmup_epochs 10 --cbm_ramp_epochs 15 \
  --batch_size 16 --lr 3e-4 --h_dim 384 --disable_cgta \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_a3d "a3d_ac_perfect_v1_minuscgta_nosparse_short_q3" \
  --dataset a3d --gpu 6 --tag a3d_ac_perfect_v1_minuscgta_nosparse_short_q3 \
  --epochs 60 --warmup_epochs 10 --cbm_ramp_epochs 15 \
  --batch_size 16 --lr 3e-4 --h_dim 256 --disable_cgta \
  --lambda_align 0 --lambda_sparse 0 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_a3d "a3d_ac_perfect_v1_minuscgta_lowac_short_q4" \
  --dataset a3d --gpu 7 --tag a3d_ac_perfect_v1_minuscgta_lowac_short_q4 \
  --epochs 60 --warmup_epochs 10 --cbm_ramp_epochs 15 \
  --batch_size 16 --lr 3e-4 --h_dim 256 --disable_cgta \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.05 --lambda_ac_value 0.1 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

wait
echo "[$(ts)] Short-horizon focused perfect_v1 search done" | tee -a "$MASTER_LOG"
