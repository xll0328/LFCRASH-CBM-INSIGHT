#!/bin/bash
# DAD v4 re-training: fixes log_sigma_sq clamping + early stopping + tuned lambdas
# Key changes vs v3_final/dad_full:
#   - log_sigma_sq clamped to [-2, 2]  → no more negative total loss / CE explosion
#   - patience=10 evals (20 epochs)    → stops at real optimum instead of ep10
#   - lambda_sparse=0.0001             → less sparsity pressure lets concepts develop
#   - lambda_recon=0.002               → lighter recon regularization
#   - lr=2e-4, wd=3e-5                 → slightly higher lr for faster convergence
#   - epochs=60                        → no point running 80 if ES triggers at ~20-30
# GPU1: 8223 MiB used, 24564 total => ~16 GB free — perfect for DAD

set -e
cd /data/sony/LFCRASH/LFCRASH-CBM

GPU=1
OUTDIR=output/v4_dad
mkdir -p "$OUTDIR"

echo "=== DAD v4: Full (with CBM) ==="
python3 train.py \
  --dataset dad \
  --gpu $GPU \
  --epochs 60 \
  --batch_size 16 \
  --lr 2e-4 \
  --weight_decay 3e-5 \
  --h_dim 256 \
  --z_dim 128 \
  --lambda_align 1e-4 \
  --lambda_sparse 1e-4 \
  --lambda_recon 2e-3 \
  --num_concepts 837 \
  --num_workers 4 \
  --eval_every 2 \
  --patience 10 \
  --output_dir "$OUTDIR" \
  --tag dad_full_v4 \
  2>&1 | tee "$OUTDIR/dad_full_v4.log" &

PID_FULL=$!
echo "  PID=$PID_FULL"

wait $PID_FULL
echo "=== Full training done. Launching ablations... ==="

# Now run ablations sequentially (GPU1 has enough VRAM for one at a time)
for COND in no_cbm no_align no_sparse no_recon; do
  FLAG="--${COND}"
  echo "=== DAD v4: $COND ==="
  python3 train.py \
    --dataset dad \
    --gpu $GPU \
    --epochs 60 \
    --batch_size 16 \
    --lr 2e-4 \
    --weight_decay 3e-5 \
    --h_dim 256 \
    --z_dim 128 \
    --lambda_align 1e-4 \
    --lambda_sparse 1e-4 \
    --lambda_recon 2e-3 \
    --num_concepts 837 \
    --num_workers 4 \
    --eval_every 2 \
    --patience 10 \
    --output_dir "$OUTDIR" \
    --tag "dad_${COND}_v4" \
    $FLAG \
    2>&1 | tee "$OUTDIR/dad_${COND}_v4.log"
done

echo "=== ALL DAD v4 DONE ==="
