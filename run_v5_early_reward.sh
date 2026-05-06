#!/bin/bash
# run_v5_early_reward.sh — CG-CRASH v5 with fixed early-reward design
# Key fix: reward ∝ time_remaining (earlier = better), not exp(-time)
mkdir -p /data/sony/LFCRASH/LFCRASH-CBM/output/dad_ac/dad_ac_v5_early_reward
exec python3 /data/sony/LFCRASH/LFCRASH-CBM/train_dad_ac.py \
  --gpu 6 \
  --tag dad_ac_v5_early_reward \
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
  --eval_every 5
