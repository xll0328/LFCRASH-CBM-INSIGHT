#!/bin/bash
# run_journal_crash_baseline.sh
# CRASH dataset baseline: full model + 4 ablation variants

set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PY="/data/sony/anaconda3/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/output/insight_journal_crash_baseline_$STAMP"
LOG="$RUN_DIR/crash_baseline.log"

mkdir -p "$RUN_DIR"

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_crash(){
  local variant="$1"
  local run_idx="$2"
  local gpu="$3"
  
  local tag="insight_journal_crash_${variant}_r${run_idx}"
  local extra_flag=""
  
  case "$variant" in
    full)      extra_flag="" ;;
    no_cbm)    extra_flag="--no_cbm" ;;
    no_align)  extra_flag="--no_align" ;;
    no_sparse) extra_flag="--no_sparse" ;;
    no_recon)  extra_flag="--no_recon" ;;
  esac
  
  log "[START] CRASH / $variant / run=$run_idx / GPU=$gpu"
  
  "$PY" "$ROOT/train.py" \
    --dataset crash \
    --gpu "$gpu" \
    --epochs 30 \
    --batch_size 16 \
    --lr 2e-4 \
    --weight_decay 3e-5 \
    --h_dim 256 \
    --z_dim 512 \
    --lambda_align 1e-4 \
    --lambda_sparse 1e-3 \
    --lambda_recon 1e-2 \
    --num_concepts 837 \
    --num_workers 4 \
    --eval_every 2 \
    --patience 8 \
    --output_dir "$RUN_DIR" \
    --tag "$tag" \
    $extra_flag \
    2>&1 | tee -a "$LOG"
  
  log "[DONE] CRASH / $variant / run=$run_idx"
}

log "====================================================="
log "INSIGHT Journal: CRASH Baseline Campaign"
log "====================================================="
log "RUN_DIR=$RUN_DIR"
log "Tasks: 1 full + 4 ablations × 3 seeds = 15 total"
log "====================================================="

# Full model: 3 runs
for run_idx in 1 2 3; do
  run_crash "full" "$run_idx" "0"
done

# Ablations: 4 variants × 3 runs
for variant in no_cbm no_align no_sparse no_recon; do
  for run_idx in 1 2 3; do
    run_crash "$variant" "$run_idx" "1"
  done
done

log "====================================================="
log "CRASH baseline campaign complete!"
log "Results: $RUN_DIR"
log "====================================================="
