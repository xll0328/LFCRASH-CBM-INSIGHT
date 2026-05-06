#!/bin/bash
# run_rwkv.sh — Launch CG-CRASH v4 with RWKV temporal module
mkdir -p /data/sony/LFCRASH/LFCRASH-CBM/output/dad_rwkv/dad_rwkv_v1
exec python3 /data/sony/LFCRASH/LFCRASH-CBM/train_dad_ac.py \
  --gpu 3 --tag dad_rwkv_v1 --epochs 200 \
  --warmup_epochs 20 --cbm_ramp_epochs 30 \
  --batch_size 16 --lr 3e-4 --h_dim 256 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
  --ac_gamma 0.95 --ac_entropy 0.01 --eval_every 5 \
  --use_rwkv
