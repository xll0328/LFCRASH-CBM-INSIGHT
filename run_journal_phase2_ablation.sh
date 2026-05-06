#!/bin/bash
# run_journal_phase2_ablation.sh
# Journal Phase 2: 36 ablation tasks (4 variants × 3 datasets × 3 seeds)
# Estimated runtime: 40-60 hours total

set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PY="/data/sony/anaconda3/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/output/insight_journal_phase2_ablation_$STAMP"
LOG="$RUN_DIR/phase2.log"

mkdir -p "$RUN_DIR"

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# GPU allocation strategy: distribute across available GPUs
# Adjust GPUS array based on your system
GPUS=(0 1 2 3 4 5 6)
GPU_IDX=0

run_ablation(){
  local ds="$1"
  local variant="$2"
  local run_idx="$3"
  local gpu="${GPUS[$((GPU_IDX % ${#GPUS[@]}))]}"
  GPU_IDX=$((GPU_IDX + 1))
  
  local tag="insight_journal_${ds}_${variant}_r${run_idx}"
  local extra_flag=""
  
  case "$variant" in
    no_cbm)    extra_flag="--no_cbm" ;;
    no_align)  extra_flag="--no_align" ;;
    no_sparse) extra_flag="--no_sparse" ;;
    no_recon)  extra_flag="--no_recon" ;;
    *)         extra_flag="" ;;
  esac
  
  log "[START] $ds / $variant / run=$run_idx / GPU=$gpu"
  
  "$PY" "$ROOT/train.py" \
    --dataset "$ds" \
    --gpu "$gpu" \
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
    $extra_flag \
    2>&1 | tee -a "$LOG"
  
  log "[DONE] $ds / $variant / run=$run_idx"
}

log "====================================================="
log "INSIGHT Journal Phase 2: Ablation Campaign"
log "====================================================="
log "RUN_DIR=$RUN_DIR"
log "Total tasks: 36 (4 variants × 3 datasets × 3 seeds)"
log "Estimated time: 40-60 hours"
log "====================================================="

# Phase 2: Run all 36 ablation tasks
# Datasets: dad, a3d, crash
# Variants: no_cbm, no_align, no_sparse, no_recon
# Seeds: 42, 123, 314

for ds in dad a3d crash; do
  for variant in no_cbm no_align no_sparse no_recon; do
    for run_idx in 1 2 3; do
      run_ablation "$ds" "$variant" "$run_idx"
    done
  done
done

log "====================================================="
log "Phase 2 ablation campaign complete!"
log "Results directory: $RUN_DIR"
log "====================================================="
