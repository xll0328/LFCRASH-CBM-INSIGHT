#!/bin/bash
# Phase 2: 完整消融实验 + 三数据集基准测试
set -e
cd /data/sony/LFCRASH/LFCRASH-CBM

OUTDIR=output/phase2_ablation
LOG=output/phase2_ablation_launch.log
mkdir -p $OUTDIR

echo "[$(date)] Phase 2 消融实验启动" | tee -a $LOG

# --- CRASH 参数 ---
CRASH_LR=2e-4
CRASH_BS=16
CRASH_EPOCHS=20
CRASH_H=256
CRASH_Z=512
CRASH_LA=1e-4
CRASH_LS=1e-3
CRASH_LRE=1e-2

# --- DAD 参数 ---
DAD_LR=4e-4
DAD_BS=16
DAD_EPOCHS=30
DAD_H=256
DAD_Z=128
DAD_LA=2.4e-5
DAD_LS=2.6e-4
DAD_LRE=1e-2

# --- A3D 参数 ---
A3D_LR=2e-4
A3D_BS=16
A3D_EPOCHS=20
A3D_H=256
A3D_Z=512
A3D_LA=1e-4
A3D_LS=1e-3
A3D_LRE=1e-2

run_exp() {
    DS=$1
    TAG=$2
    GPU=$3
    EPOCHS=$4
    BS=$5
    LR=$6
    H=$7
    Z=$8
    LA=$9
    LS=${10}
    LRE=${11}
    shift 11
    EXTRA=$@

    echo "[$(date)] START ${DS}-${TAG} GPU=${GPU}" | tee -a $LOG
    python train.py \
        --dataset $DS \
        --gpu $GPU \
        --epochs $EPOCHS \
        --batch_size $BS \
        --lr $LR \
        --h_dim $H \
        --z_dim $Z \
        --lambda_align $LA \
        --lambda_sparse $LS \
        --lambda_recon $LRE \
        --output_dir $OUTDIR \
        --tag ${DS}_${TAG} \
        $EXTRA \
        2>&1 | tee -a $OUTDIR/${DS}_${TAG}.log
    echo "[$(date)] DONE  ${DS}-${TAG}" | tee -a $LOG
}

# ============================================================
# A3D on GPU3 (background)
# ============================================================
(
    run_exp a3d full      3 $A3D_EPOCHS $A3D_BS $A3D_LR $A3D_H $A3D_Z $A3D_LA $A3D_LS $A3D_LRE
    run_exp a3d no_align  3 $A3D_EPOCHS $A3D_BS $A3D_LR $A3D_H $A3D_Z $A3D_LA $A3D_LS $A3D_LRE --no_align
    run_exp a3d no_sparse 3 $A3D_EPOCHS $A3D_BS $A3D_LR $A3D_H $A3D_Z $A3D_LA $A3D_LS $A3D_LRE --no_sparse
    run_exp a3d no_recon  3 $A3D_EPOCHS $A3D_BS $A3D_LR $A3D_H $A3D_Z $A3D_LA $A3D_LS $A3D_LRE --no_recon
    run_exp a3d no_cbm    3 $A3D_EPOCHS $A3D_BS $A3D_LR $A3D_H $A3D_Z $A3D_LA $A3D_LS $A3D_LRE --no_cbm
    echo "[$(date)] A3D all done" | tee -a $LOG
) &
A3D_PID=$!

# ============================================================
# CRASH on GPU2 (foreground serial)
# ============================================================
run_exp crash full      2 $CRASH_EPOCHS $CRASH_BS $CRASH_LR $CRASH_H $CRASH_Z $CRASH_LA $CRASH_LS $CRASH_LRE
run_exp crash no_align  2 $CRASH_EPOCHS $CRASH_BS $CRASH_LR $CRASH_H $CRASH_Z $CRASH_LA $CRASH_LS $CRASH_LRE --no_align
run_exp crash no_sparse 2 $CRASH_EPOCHS $CRASH_BS $CRASH_LR $CRASH_H $CRASH_Z $CRASH_LA $CRASH_LS $CRASH_LRE --no_sparse
run_exp crash no_recon  2 $CRASH_EPOCHS $CRASH_BS $CRASH_LR $CRASH_H $CRASH_Z $CRASH_LA $CRASH_LS $CRASH_LRE --no_recon
run_exp crash no_cbm    2 $CRASH_EPOCHS $CRASH_BS $CRASH_LR $CRASH_H $CRASH_Z $CRASH_LA $CRASH_LS $CRASH_LRE --no_cbm
echo "[$(date)] CRASH all done" | tee -a $LOG

# ============================================================
# DAD on GPU2
# ============================================================
run_exp dad full      2 $DAD_EPOCHS $DAD_BS $DAD_LR $DAD_H $DAD_Z $DAD_LA $DAD_LS $DAD_LRE
run_exp dad no_align  2 $DAD_EPOCHS $DAD_BS $DAD_LR $DAD_H $DAD_Z $DAD_LA $DAD_LS $DAD_LRE --no_align
run_exp dad no_sparse 2 $DAD_EPOCHS $DAD_BS $DAD_LR $DAD_H $DAD_Z $DAD_LA $DAD_LS $DAD_LRE --no_sparse
run_exp dad no_recon  2 $DAD_EPOCHS $DAD_BS $DAD_LR $DAD_H $DAD_Z $DAD_LA $DAD_LS $DAD_LRE --no_recon
run_exp dad no_cbm    2 $DAD_EPOCHS $DAD_BS $DAD_LR $DAD_H $DAD_Z $DAD_LA $DAD_LS $DAD_LRE --no_cbm
echo "[$(date)] DAD all done" | tee -a $LOG

wait $A3D_PID
echo "[$(date)] ====== Phase 2 ALL COMPLETE ======" | tee -a $LOG
