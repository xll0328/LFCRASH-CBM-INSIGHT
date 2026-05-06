#!/bin/bash
# INSIGHT journal campaign: >=20h automatic experiments
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PY="/data/sony/anaconda3/bin/python"
GPU_PRIMARY="4"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/output/insight_journal_campaign_$STAMP"
LOG="$RUN_DIR/campaign.log"

mkdir -p "$RUN_DIR"

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_curriculum_seed(){
  local seed="$1"
  local tag="insight_journal_s${seed}_curriculum"
  log "[START] DAD curriculum seed=$seed tag=$tag"
  "$PY" "$ROOT/train_dad_curriculum.py" \
    --gpu "$GPU_PRIMARY" \
    --tag "$tag" \
    --seed "$seed" \
    --epochs 150 \
    --warmup_epochs 20 \
    --eval_every 5 \
    --batch_size 16 \
    --h_dim 256 \
    --z_dim 256 \
    --lambda_align 1e-6 \
    --lambda_sparse 5e-5 \
    --lambda_recon 1e-4 \
    2>&1 | tee -a "$LOG"
  log "[DONE] DAD curriculum seed=$seed"
}

run_dad_ablation(){
  local cond="$1"
  local extra="$2"
  local tag="insight_journal_dad_${cond}"
  log "[START] DAD ablation cond=$cond"
  "$PY" "$ROOT/train.py" \
    --dataset dad \
    --gpu "$GPU_PRIMARY" \
    --epochs 40 \
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
    --output_dir "$RUN_DIR" \
    --tag "$tag" \
    $extra \
    2>&1 | tee -a "$LOG"
  log "[DONE] DAD ablation cond=$cond"
}

log "====================================================="
log "INSIGHT journal campaign started"
log "ROOT=$ROOT"
log "GPU_PRIMARY=$GPU_PRIMARY"
log "RUN_DIR=$RUN_DIR"
log "====================================================="

# Phase A: three-seed long curriculum runs (main robustness evidence)
for seed in 314 2718 3407; do
  run_curriculum_seed "$seed"
done

# Phase B: focused ablations for journal evidence density
run_dad_ablation "no_cbm" "--no_cbm"
run_dad_ablation "no_align" "--no_align"
run_dad_ablation "no_sparse" "--no_sparse"
run_dad_ablation "no_recon" "--no_recon"

log "====================================================="
log "INSIGHT journal campaign completed"
log "All logs and checkpoints under: $RUN_DIR"
log "====================================================="
