#!/bin/bash
# ============================================================================
# 补跑 CRASH 缺失的两个消融实验: no_sparse + no_recon
# GPU 5 顺序执行
# ============================================================================
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

OUTDIR="output/v3_final"
EPOCHS=80
EVAL_EVERY=2
WORKERS=4
GPU=5
DS=crash
COMMON="--dataset ${DS} --gpu ${GPU} --epochs ${EPOCHS} --eval_every ${EVAL_EVERY} --num_workers ${WORKERS} --output_dir ${OUTDIR}"
HP="--h_dim 256 --z_dim 512 --lr 2e-4 --weight_decay 9.8e-5 --batch_size 16"
CBM="--lambda_align 1e-4 --lambda_sparse 1e-3 --lambda_recon 1e-3"

echo "================================================================="
echo " CRASH Missing Ablations (GPU${GPU})"
echo " Start: $(date)"
echo "================================================================="

echo "[GPU${GPU}] === CRASH: No-Sparse ==="
python train.py ${COMMON} ${HP} ${CBM} --no_sparse --tag ${DS}_no_sparse
echo "[GPU${GPU}] No-Sparse DONE at $(date)"

echo "[GPU${GPU}] === CRASH: No-Recon ==="
python train.py ${COMMON} ${HP} ${CBM} --no_recon --tag ${DS}_no_recon
echo "[GPU${GPU}] No-Recon DONE at $(date)"

echo "[GPU${GPU}] === CRASH: ALL MISSING ABLATIONS DONE ==="
echo "Completed at: $(date)"
