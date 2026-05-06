#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
cd /data/sony/LFCRASH/LFCRASH-CBM
exec /data/sony/anaconda3/bin/python train.py \
  --dataset crash --epochs 80 --batch_size 8 \
  --h_dim 256 --z_dim 512 --lr 2e-4 --weight_decay 9.8e-5 \
  --lambda_align 1e-4 --lambda_sparse 1e-3 --lambda_recon 1e-2 \
  --num_concepts 837 --num_workers 0 --eval_every 5 \
  --output_dir output/v2_20260314 \
  > output/v2_20260314/crash_cgcrash.log 2>&1
