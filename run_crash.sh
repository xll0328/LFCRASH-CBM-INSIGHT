#!/bin/bash
# run_crash.sh — Launch CCD crash training
mkdir -p /data/sony/LFCRASH/LFCRASH-CBM/output/crash_ac/crash_ac_v1
cd /data/sony/LFCRASH/LFCRASH-CBM
exec python3 train_multi.py \
  --dataset crash \
  --gpu 4 \
  --tag crash_ac_v1 \
  --epochs 100 \
  --warmup_epochs 10 \
  --cbm_ramp_epochs 15 \
  --batch_size 16 \
  --lr 3e-4 \
  --eval_every 5
