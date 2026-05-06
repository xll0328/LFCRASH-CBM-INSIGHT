#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONCEPT_FILE="$ROOT/output/concept_sets/perfect_concept_set_v1.txt"
MASTER_LOG="$ROOT/output/perfect_v1_8gpu_queue_master.log"

mkdir -p "$ROOT/output/dad_ac" "$ROOT/output/a3d_ac"

ts() { date '+%F %T'; }

run_dad_ac_job() {
  local queue_log="$1"
  local tag="$2"
  shift 2
  local out_dir="$ROOT/output/dad_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(ts)] START $tag" | tee -a "$queue_log"
  "$PYTHON_BIN" "$ROOT/train_dad_ac.py" "$@" > "$out_dir/nohup.log" 2>&1
  local ec=$?
  echo "[$(ts)] END $tag exit=$ec" | tee -a "$queue_log"
  return $ec
}

run_multi_job() {
  local queue_log="$1"
  local dataset_dir="$2"
  local tag="$3"
  shift 3
  local out_dir="$ROOT/output/${dataset_dir}_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(ts)] START $tag" | tee -a "$queue_log"
  "$PYTHON_BIN" "$ROOT/train_multi.py" "$@" > "$out_dir/nohup.log" 2>&1
  local ec=$?
  echo "[$(ts)] END $tag exit=$ec" | tee -a "$queue_log"
  return $ec
}

queue_gpu2_dad_multi() {
  local qlog="$ROOT/output/dad_ac/queue_gpu2_dad_multi_perfect_v1.log"
  echo "[$(ts)] Queue start GPU2 DAD multi perfect_v1" > "$qlog"
  run_multi_job "$qlog" dad "dad_multi_perfect_v1_shared_q1" \
    --dataset dad --gpu 2 --tag dad_multi_perfect_v1_shared_q1 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 \
    --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_multi_job "$qlog" dad "dad_multi_perfect_v1_lowac_q2" \
    --dataset dad --gpu 2 --tag dad_multi_perfect_v1_lowac_q2 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 \
    --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.1 --lambda_ac_value 0.1 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_multi_job "$qlog" dad "dad_multi_perfect_v1_plainac_q3" \
    --dataset dad --gpu 2 --tag dad_multi_perfect_v1_plainac_q3 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 \
    --disable_cgta --disable_crs \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  echo "[$(ts)] Queue done GPU2 DAD multi perfect_v1" | tee -a "$qlog"
}

queue_gpu3_dad_advanced() {
  local qlog="$ROOT/output/dad_ac/queue_gpu3_dad_advanced_perfect_v1.log"
  echo "[$(ts)] Queue start GPU3 DAD advanced perfect_v1" > "$qlog"
  run_dad_ac_job "$qlog" "dad_ac_perfect_v1_h384_q1" \
    --gpu 3 --tag dad_ac_perfect_v1_h384_q1 \
    --epochs 200 --warmup_epochs 20 --cbm_ramp_epochs 30 \
    --batch_size 16 --lr 3e-4 --h_dim 384 --z_dim 384 \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 --ac_gamma 0.95 --ac_entropy 0.01 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_dad_ac_job "$qlog" "dad_ac_perfect_v1_h384_q2" \
    --gpu 3 --tag dad_ac_perfect_v1_h384_q2 \
    --epochs 200 --warmup_epochs 20 --cbm_ramp_epochs 30 \
    --batch_size 16 --lr 3e-4 --h_dim 384 --z_dim 384 \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 --ac_gamma 0.95 --ac_entropy 0.01 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_dad_ac_job "$qlog" "dad_ac_perfect_v1_aligned_q3" \
    --gpu 3 --tag dad_ac_perfect_v1_aligned_q3 \
    --epochs 200 --warmup_epochs 20 --cbm_ramp_epochs 30 \
    --batch_size 16 --lr 3e-4 --h_dim 256 --z_dim 256 \
    --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 --ac_gamma 0.95 --ac_entropy 0.01 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  echo "[$(ts)] Queue done GPU3 DAD advanced perfect_v1" | tee -a "$qlog"
}

queue_gpu4_dad_policy() {
  local qlog="$ROOT/output/dad_ac/queue_gpu4_dad_policy_perfect_v1.log"
  echo "[$(ts)] Queue start GPU4 DAD policy perfect_v1" > "$qlog"
  run_dad_ac_job "$qlog" "dad_ac_perfect_v1_lowentropy_q1" \
    --gpu 4 --tag dad_ac_perfect_v1_lowentropy_q1 \
    --epochs 200 --warmup_epochs 20 --cbm_ramp_epochs 30 \
    --batch_size 16 --lr 3e-4 --h_dim 256 --z_dim 256 \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 --ac_gamma 0.95 --ac_entropy 0.005 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_dad_ac_job "$qlog" "dad_ac_perfect_v1_highpolicy_q2" \
    --gpu 4 --tag dad_ac_perfect_v1_highpolicy_q2 \
    --epochs 200 --warmup_epochs 20 --cbm_ramp_epochs 30 \
    --batch_size 16 --lr 3e-4 --h_dim 256 --z_dim 256 \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.5 --lambda_ac_value 0.3 --ac_gamma 0.95 --ac_entropy 0.01 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_dad_ac_job "$qlog" "dad_ac_perfect_v1_balancedac_q3" \
    --gpu 4 --tag dad_ac_perfect_v1_balancedac_q3 \
    --epochs 200 --warmup_epochs 20 --cbm_ramp_epochs 30 \
    --batch_size 16 --lr 3e-4 --h_dim 256 --z_dim 256 \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.2 --lambda_ac_value 0.2 --ac_gamma 0.95 --ac_entropy 0.01 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  echo "[$(ts)] Queue done GPU4 DAD policy perfect_v1" | tee -a "$qlog"
}

queue_gpu5_a3d_modules() {
  local qlog="$ROOT/output/a3d_ac/queue_gpu5_a3d_modules_perfect_v1.log"
  echo "[$(ts)] Queue start GPU5 A3D modules perfect_v1" > "$qlog"
  run_multi_job "$qlog" a3d "a3d_ac_perfect_v1_minuscgta_q1" \
    --dataset a3d --gpu 5 --tag a3d_ac_perfect_v1_minuscgta_q1 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 --disable_cgta \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_multi_job "$qlog" a3d "a3d_ac_perfect_v1_minuscrs_q2" \
    --dataset a3d --gpu 5 --tag a3d_ac_perfect_v1_minuscrs_q2 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 --disable_crs \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_multi_job "$qlog" a3d "a3d_ac_perfect_v1_aligned_q3" \
    --dataset a3d --gpu 5 --tag a3d_ac_perfect_v1_aligned_q3 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 \
    --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  echo "[$(ts)] Queue done GPU5 A3D modules perfect_v1" | tee -a "$qlog"
}

queue_gpu6_a3d_capacity() {
  local qlog="$ROOT/output/a3d_ac/queue_gpu6_a3d_capacity_perfect_v1.log"
  echo "[$(ts)] Queue start GPU6 A3D capacity perfect_v1" > "$qlog"
  run_multi_job "$qlog" a3d "a3d_ac_perfect_v1_h384_q1" \
    --dataset a3d --gpu 6 --tag a3d_ac_perfect_v1_h384_q1 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 384 \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_multi_job "$qlog" a3d "a3d_ac_perfect_v1_noconcepts_q2" \
    --dataset a3d --gpu 6 --tag a3d_ac_perfect_v1_noconcepts_q2 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 --ac_no_concepts \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_multi_job "$qlog" a3d "a3d_ac_perfect_v1_noac_q3" \
    --dataset a3d --gpu 6 --tag a3d_ac_perfect_v1_noac_q3 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 --disable_ac \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  echo "[$(ts)] Queue done GPU6 A3D capacity perfect_v1" | tee -a "$qlog"
}

queue_gpu7_a3d_regularization() {
  local qlog="$ROOT/output/a3d_ac/queue_gpu7_a3d_regularization_perfect_v1.log"
  echo "[$(ts)] Queue start GPU7 A3D regularization perfect_v1" > "$qlog"
  run_multi_job "$qlog" a3d "a3d_ac_perfect_v1_nosparse_q1" \
    --dataset a3d --gpu 7 --tag a3d_ac_perfect_v1_nosparse_q1 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 \
    --lambda_align 0 --lambda_sparse 0 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_multi_job "$qlog" a3d "a3d_ac_perfect_v1_norecon_q2" \
    --dataset a3d --gpu 7 --tag a3d_ac_perfect_v1_norecon_q2 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 \
    --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 0 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  run_multi_job "$qlog" a3d "a3d_ac_perfect_v1_align1e5_q3" \
    --dataset a3d --gpu 7 --tag a3d_ac_perfect_v1_align1e5_q3 \
    --epochs 150 --warmup_epochs 15 --cbm_ramp_epochs 20 \
    --batch_size 16 --lr 3e-4 --h_dim 256 \
    --lambda_align 1e-5 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
    --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
    --num_concepts 80 --concept_file "$CONCEPT_FILE" \
    --eval_every 5 --num_workers 4
  echo "[$(ts)] Queue done GPU7 A3D regularization perfect_v1" | tee -a "$qlog"
}

echo "[$(ts)] Launching perfect_v1 8-GPU expansion queues (GPU2-7 here; GPU0-1 already reserved)" | tee "$MASTER_LOG"
(queue_gpu2_dad_multi) &
(queue_gpu3_dad_advanced) &
(queue_gpu4_dad_policy) &
(queue_gpu5_a3d_modules) &
(queue_gpu6_a3d_capacity) &
(queue_gpu7_a3d_regularization) &
wait

echo "[$(ts)] All GPU2-7 perfect_v1 queues completed" | tee -a "$MASTER_LOG"
