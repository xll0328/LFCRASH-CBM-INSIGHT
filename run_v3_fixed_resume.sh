#!/bin/bash
# run_v3_fixed_resume.sh — Resume v3_fixed_lr from best checkpoint with fixed AC loss
mkdir -p /data/sony/LFCRASH/LFCRASH-CBM/output/dad_ac/dad_ac_v3_fixed_resume
exec python3 /data/sony/LFCRASH/LFCRASH-CBM/train_dad_ac.py \
  --gpu 1 \
  --tag dad_ac_v3_fixed_resume \
  --epochs 200 \
  --warmup_epochs 20 \
  --cbm_ramp_epochs 30 \
  --batch_size 16 \
  --lr 3e-4 \
  --h_dim 256 \
  --lambda_ac_policy 0.05 \
  --lambda_ac_value 0.1 \
  --ac_gamma 0.95 \
  --ac_entropy 0.005 \
  --eval_every 5 \
  --resume /data/sony/LFCRASH/LFCRASH-CBM/output/dad_ac/dad_ac_v3_fixed_lr/best_model.pt
