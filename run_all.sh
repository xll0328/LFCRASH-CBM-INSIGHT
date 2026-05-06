#!/bin/bash
# run_all.sh – Launch all 3 datasets in parallel on GPUs 5, 6, 1
# Usage: bash run_all.sh

cd /data/sony/LFCRASH/LFCRASH-CBM
PY=/data/sony/anaconda3/bin/python
OUT=output/run_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUT

echo "Launching DAD    on GPU 5..."
CUDA_VISIBLE_DEVICES=5 $PY train.py \
  --dataset dad \
  --gpu 5 \
  --epochs 100 \
  --batch_size 8 \
  --h_dim 256 \
  --z_dim 128 \
  --lr 4e-4 \
  --weight_decay 2.8e-5 \
  --lambda_align 2.4e-5 \
  --lambda_sparse 2.6e-4 \
  --num_concepts 837 \
  --num_workers 4 \
  --eval_every 5 \
  --output_dir $OUT \
  > $OUT/dad.log 2>&1 &
echo "  DAD  PID=$!"

echo "Launching Crash  on GPU 6..."
CUDA_VISIBLE_DEVICES=6 $PY train.py \
  --dataset crash \
  --gpu 6 \
  --epochs 100 \
  --batch_size 8 \
  --h_dim 256 \
  --z_dim 512 \
  --lr 2e-4 \
  --weight_decay 9.8e-5 \
  --lambda_align 1.1e-5 \
  --lambda_sparse 1.1e-3 \
  --num_concepts 837 \
  --num_workers 4 \
  --eval_every 5 \
  --output_dir $OUT \
  > $OUT/crash.log 2>&1 &
echo "  Crash PID=$!"

echo "Launching A3D    on GPU 3..."
CUDA_VISIBLE_DEVICES=3 $PY train.py \
  --dataset a3d \
  --gpu 3 \
  --epochs 100 \
  --batch_size 32 \
  --h_dim 768 \
  --z_dim 128 \
  --lr 2.5e-6 \
  --weight_decay 1.2e-6 \
  --lambda_align 6.6e-4 \
  --lambda_sparse 4.8e-3 \
  --num_concepts 837 \
  --num_workers 4 \
  --eval_every 5 \
  --output_dir $OUT \
  > $OUT/a3d.log 2>&1 &
echo "  A3D  PID=$!"

echo ""
echo "All 3 jobs launched. Output dir: $OUT"
echo "Monitor: tail -f $OUT/dad.log"
echo "Monitor: tail -f $OUT/crash.log"
echo "Monitor: tail -f $OUT/a3d.log"
