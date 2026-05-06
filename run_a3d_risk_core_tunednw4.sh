#!/bin/bash
set -e
ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
cd "$ROOT"
nohup python3 "$ROOT/train_multi.py" \
  --dataset a3d \
  --gpu 5 \
  --tag a3d_ac_risk_core_noalign_v1_tunednw4 \
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
  --num_concepts 30 \
  --concept_file "$ROOT/output/concept_sets/risk_core_concept_set_v1.txt" \
  --eval_every 5 \
  --num_workers 4 \
  > "$ROOT/output/a3d_ac/a3d_ac_risk_core_noalign_v1_tunednw4/nohup.log" 2>&1 &
