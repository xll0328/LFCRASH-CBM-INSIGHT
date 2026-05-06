#!/bin/bash
# run_journal_transfer_learning.sh
# Transfer learning: DAD pre-train → A3D fine-tune

set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PY="/data/sony/anaconda3/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/output/insight_journal_transfer_$STAMP"
LOG="$RUN_DIR/transfer.log"

mkdir -p "$RUN_DIR"

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "====================================================="
log "INSIGHT Journal: Transfer Learning Campaign"
log "====================================================="
log "Strategy: DAD pre-train (20 epochs) → A3D fine-tune (15 epochs)"
log "Seeds: 42, 123, 314"
log "====================================================="

for seed in 42 123 314; do
  log "[PHASE 1] DAD pre-training seed=$seed"
  
  "$PY" "$ROOT/train.py" \
    --dataset dad \
    --gpu 2 \
    --epochs 20 \
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
    --output_dir "$RUN_DIR/pretrain" \
    --tag "transfer_dad_pretrain_s${seed}" \
    --seed "$seed" \
    2>&1 | tee -a "$LOG"
  
  # Find best checkpoint from pre-training
  BEST_CKPT=$(find "$RUN_DIR/pretrain" -name "best_model.pth" -path "*transfer_dad_pretrain_s${seed}*" | head -1)
  
  if [ -z "$BEST_CKPT" ]; then
    log "[ERROR] No checkpoint found for seed=$seed, skipping fine-tune"
    continue
  fi
  
  log "[PHASE 2] A3D fine-tuning seed=$seed from $BEST_CKPT"
  
  # Fine-tune on A3D (lower lr, fewer epochs)
  "$PY" "$ROOT/train.py" \
    --dataset a3d \
    --gpu 2 \
    --epochs 15 \
    --batch_size 16 \
    --lr 5e-5 \
    --weight_decay 3e-5 \
    --h_dim 256 \
    --z_dim 128 \
    --lambda_align 1e-4 \
    --lambda_sparse 1e-4 \
    --lambda_recon 2e-3 \
    --num_concepts 837 \
    --num_workers 4 \
    --eval_every 2 \
    --output_dir "$RUN_DIR/finetune" \
    --tag "transfer_a3d_finetune_s${seed}" \
    --seed "$seed" \
    --pretrained_checkpoint "$BEST_CKPT" \
    2>&1 | tee -a "$LOG"
  
  log "[DONE] Transfer learning seed=$seed complete"
done

log "====================================================="
log "Transfer learning campaign complete!"
log "Results: $RUN_DIR"
log "====================================================="
