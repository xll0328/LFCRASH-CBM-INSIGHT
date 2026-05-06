#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
cd /data/sony/LFCRASH/LFCRASH-CBM
exec /data/sony/anaconda3/bin/python train.py \
  --dataset a3d --epochs 80 --batch_size 32 \
  --h_dim 768 --z_dim 128 --lr 2.5e-6 --weight_decay 1.2e-6 \
  --lambda_align 1e-4 --lambda_sparse 1e-3 --lambda_recon 1e-2 \
  --num_concepts 837 --num_workers 0 --eval_every 5 \
  --output_dir output/v2_20260314 \
  > output/v2_20260314/a3d_cgcrash.log 2>&1
