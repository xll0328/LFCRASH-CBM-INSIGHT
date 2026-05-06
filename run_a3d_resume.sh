#!/bin/bash
# run_a3d_resume.sh — Resume A3D from best checkpoint (AP=97.36%)
mkdir -p /data/sony/LFCRASH/LFCRASH-CBM/output/a3d_ac/a3d_ac_v1_resume
exec python3 /data/sony/LFCRASH/LFCRASH-CBM/train_multi.py \
  --dataset a3d \
  --gpu 5 \
  --tag a3d_ac_v1_resume \
  --epochs 150 \
  --warmup_epochs 15 \
  --cbm_ramp_epochs 20 \
  --batch_size 16 \
  --lr 3e-4 \
  --lambda_ac_policy 0.05 \
  --lambda_ac_value 0.1 \
  --eval_every 5
