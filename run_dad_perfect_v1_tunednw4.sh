#!/bin/bash
set -e
ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
cd "$ROOT"
mkdir -p "$ROOT/output/dad_ac/dad_ac_perfect_v1_noalign_tunednw4"
nohup python3 "$ROOT/train_dad_ac.py" \
  --gpu 0 \
  --tag dad_ac_perfect_v1_noalign_tunednw4 \
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
  --concept_file "$ROOT/output/concept_sets/perfect_concept_set_v1.txt" \
  --eval_every 5 \
  --num_workers 4 \
  > "$ROOT/output/dad_ac/dad_ac_perfect_v1_noalign_tunednw4/nohup.log" 2>&1 &
