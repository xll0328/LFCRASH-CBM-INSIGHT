#!/bin/bash
# ============================================================================
# LFCRASH-CBM v3 Complete Experiment Suite
# ============================================================================
# GPU 4: CRASH dataset (all conditions)
# GPU 5: A3D   dataset (all conditions)
# GPU 6: DAD   dataset (all conditions)
#
# For each dataset: Full + 4 ablations = 5 experiments run sequentially per GPU
# All 3 GPUs run in parallel -> total ~15 experiments
# ============================================================================

set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

OUTDIR="output/v3_final"
EPOCHS=80
EVAL_EVERY=2
WORKERS=4

echo "================================================================="
echo " LFCRASH-CBM v3 Experiment Suite"
echo " Start: $(date)"
echo " Output: ${OUTDIR}"
echo "================================================================="

# ── GPU 4: CRASH ──────────────────────────────────────────────────────────
run_crash() {
    local GPU=4
    local DS=crash
    local COMMON="--dataset ${DS} --gpu ${GPU} --epochs ${EPOCHS} --eval_every ${EVAL_EVERY} --num_workers ${WORKERS} --output_dir ${OUTDIR}"
    local HP="--h_dim 256 --z_dim 512 --lr 2e-4 --weight_decay 9.8e-5 --batch_size 16"
    local CBM="--lambda_align 1e-4 --lambda_sparse 1e-3 --lambda_recon 1e-3"

    echo "[GPU${GPU}] === CRASH: Full CG-CRASH ==="
    python train.py ${COMMON} ${HP} ${CBM} --tag ${DS}_full

    echo "[GPU${GPU}] === CRASH: No-CBM (baseline) ==="
    python train.py ${COMMON} ${HP} --no_cbm --tag ${DS}_no_cbm

    echo "[GPU${GPU}] === CRASH: No-Align ==="
    python train.py ${COMMON} ${HP} ${CBM} --no_align --tag ${DS}_no_align

    echo "[GPU${GPU}] === CRASH: No-Sparse ==="
    python train.py ${COMMON} ${HP} ${CBM} --no_sparse --tag ${DS}_no_sparse

    echo "[GPU${GPU}] === CRASH: No-Recon ==="
    python train.py ${COMMON} ${HP} ${CBM} --no_recon --tag ${DS}_no_recon

    echo "[GPU${GPU}] === CRASH: ALL DONE ==="
}

# ── GPU 5: A3D ────────────────────────────────────────────────────────────
run_a3d() {
    local GPU=5
    local DS=a3d
    local COMMON="--dataset ${DS} --gpu ${GPU} --epochs ${EPOCHS} --eval_every ${EVAL_EVERY} --num_workers ${WORKERS} --output_dir ${OUTDIR}"
    local HP="--h_dim 256 --z_dim 256 --lr 1e-4 --weight_decay 1e-5 --batch_size 32"
    local CBM="--lambda_align 5e-4 --lambda_sparse 3e-3 --lambda_recon 1e-2"

    echo "[GPU${GPU}] === A3D: Full CG-CRASH ==="
    python train.py ${COMMON} ${HP} ${CBM} --tag ${DS}_full

    echo "[GPU${GPU}] === A3D: No-CBM (baseline) ==="
    python train.py ${COMMON} ${HP} --no_cbm --tag ${DS}_no_cbm

    echo "[GPU${GPU}] === A3D: No-Align ==="
    python train.py ${COMMON} ${HP} ${CBM} --no_align --tag ${DS}_no_align

    echo "[GPU${GPU}] === A3D: No-Sparse ==="
    python train.py ${COMMON} ${HP} ${CBM} --no_sparse --tag ${DS}_no_sparse

    echo "[GPU${GPU}] === A3D: No-Recon ==="
    python train.py ${COMMON} ${HP} ${CBM} --no_recon --tag ${DS}_no_recon

    echo "[GPU${GPU}] === A3D: ALL DONE ==="
}

# ── GPU 6: DAD ────────────────────────────────────────────────────────────
run_dad() {
    local GPU=6
    local DS=dad
    local COMMON="--dataset ${DS} --gpu ${GPU} --epochs ${EPOCHS} --eval_every ${EVAL_EVERY} --num_workers ${WORKERS} --output_dir ${OUTDIR}"
    local HP="--h_dim 256 --z_dim 128 --lr 3e-4 --weight_decay 3e-5 --batch_size 16"
    local CBM="--lambda_align 1e-4 --lambda_sparse 5e-4 --lambda_recon 5e-3"

    echo "[GPU${GPU}] === DAD: Full CG-CRASH ==="
    python train.py ${COMMON} ${HP} ${CBM} --tag ${DS}_full

    echo "[GPU${GPU}] === DAD: No-CBM (baseline) ==="
    python train.py ${COMMON} ${HP} --no_cbm --tag ${DS}_no_cbm

    echo "[GPU${GPU}] === DAD: No-Align ==="
    python train.py ${COMMON} ${HP} ${CBM} --no_align --tag ${DS}_no_align

    echo "[GPU${GPU}] === DAD: No-Sparse ==="
    python train.py ${COMMON} ${HP} ${CBM} --no_sparse --tag ${DS}_no_sparse

    echo "[GPU${GPU}] === DAD: No-Recon ==="
    python train.py ${COMMON} ${HP} ${CBM} --no_recon --tag ${DS}_no_recon

    echo "[GPU${GPU}] === DAD: ALL DONE ==="
}

# Launch all three GPUs in parallel
run_crash > "${OUTDIR}/crash_experiments.log" 2>&1 &
PID_CRASH=$!

run_a3d   > "${OUTDIR}/a3d_experiments.log"   2>&1 &
PID_A3D=$!

run_dad   > "${OUTDIR}/dad_experiments.log"   2>&1 &
PID_DAD=$!

echo "Launched experiments:"
echo "  GPU4 (CRASH): PID=${PID_CRASH}"
echo "  GPU5 (A3D):   PID=${PID_A3D}"
echo "  GPU6 (DAD):   PID=${PID_DAD}"
echo ""
echo "Monitor with:"
echo "  tail -f ${OUTDIR}/crash_experiments.log"
echo "  tail -f ${OUTDIR}/a3d_experiments.log"
echo "  tail -f ${OUTDIR}/dad_experiments.log"
echo ""

# Wait for all to complete
wait $PID_CRASH
EC_CRASH=$?
echo "[$(date)] CRASH experiments finished (exit=$EC_CRASH)"

wait $PID_A3D
EC_A3D=$?
echo "[$(date)] A3D experiments finished (exit=$EC_A3D)"

wait $PID_DAD
EC_DAD=$?
echo "[$(date)] DAD experiments finished (exit=$EC_DAD)"

echo ""
echo "================================================================="
echo " ALL EXPERIMENTS COMPLETE: $(date)"
echo "================================================================="

# Collect results
python -c "
import json, glob, os
results = {}
for f in sorted(glob.glob('${OUTDIR}/*/results.json')):
    tag = os.path.basename(os.path.dirname(f))
    with open(f) as fh:
        r = json.load(fh)
    results[tag] = {
        'AP': round(r['AP']*100, 2),
        'mTTA': round(r['mTTA'], 3),
        'TTA_R80': round(r['TTA_R80'], 3),
        'P_R80': round(r['P_R80']*100, 2),
        'best_epoch': r['best_epoch'],
    }
print('\n' + '='*80)
print('FINAL RESULTS SUMMARY')
print('='*80)
print(f'{\"Tag\":<25s} {\"AP%\":>8s} {\"mTTA\":>8s} {\"TTA@R80\":>8s} {\"P@R80%\":>8s} {\"BestEp\":>7s}')
print('-'*80)
for tag, r in sorted(results.items()):
    print(f'{tag:<25s} {r[\"AP\"]:8.2f} {r[\"mTTA\"]:8.3f} {r[\"TTA_R80\"]:8.3f} {r[\"P_R80\"]:8.2f} {r[\"best_epoch\"]:7d}')
print('='*80)
with open('${OUTDIR}/final_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Summary saved to ${OUTDIR}/final_summary.json')
"
