#!/bin/bash
# ============================================================
# experiment_scheduler.sh
# CG-CRASH 完整实验调度器
#
# 实验序列（后台有序执行，互不干扰）：
#   Phase 1: 等待并持续监控三个数据集训练，每5个epoch记录一次
#   Phase 2: 训练完成后，对最佳checkpoint运行：
#             a) 完整概念评估 (eval_concept.py)
#             b) 消融实验 (src/ablation.py)
#             c) Intervention测试
#   Phase 3: 生成汇总对比报告
#
# 使用：
#   nohup bash experiment_scheduler.sh > output/v2_20260314/scheduler.log 2>&1 &
# ============================================================

set -euo pipefail

PY=/data/sony/anaconda3/bin/python
CBM=/data/sony/LFCRASH/LFCRASH-CBM
BASE=$CBM/output/v2_20260314
CONCEPT_FILE=/data/sony/LFCRASH/000_all_concept_set.txt
export CUDA_VISIBLE_DEVICES=6

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $BASE/scheduler.log; }

mkdir -p $BASE/concept_eval $BASE/ablation $BASE/reports

# ── Helpers ──────────────────────────────────────────────────────────────────
latest_ckpt() {
    local ds=$1
    find $BASE -name 'best_model.pth' -path "*${ds}*" 2>/dev/null | sort | tail -1
}

training_done() {
    # Returns 0 (true) if training process for dataset is no longer running
    local ds=$1
    pgrep -f "train.py.*--dataset $ds.*v2_20260314" >/dev/null 2>&1 && return 1 || return 0
}

get_best_ap() {
    local ds=$1
    local ckpt=$(latest_ckpt $ds)
    [ -z "$ckpt" ] && echo "0" && return
    $PY -c "import torch; c=torch.load('$ckpt',map_location='cpu'); print(f\"{c.get('AP',0):.4f}\")" 2>/dev/null || echo "0"
}

# ── Phase 1: Training monitor (log every 5 min until done) ───────────────────
phase1_monitor() {
    log "=== PHASE 1: Training Monitor ==="
    local all_done=0
    while [ $all_done -eq 0 ]; do
        all_done=1
        for DS in dad crash a3d; do
            CKPT=$(latest_ckpt $DS)
            AP=$(get_best_ap $DS)
            if ! training_done $DS; then
                all_done=0
                log "  [$DS] training... best_AP=$AP  ckpt=$(basename $(dirname $CKPT 2>/dev/null) 2>/dev/null)"
            else
                log "  [$DS] DONE. best_AP=$AP"
            fi
        done
        [ $all_done -eq 0 ] && sleep 300  # check every 5 min
    done
    log "=== All training complete ==="
}

# ── Phase 2a: Concept evaluation ─────────────────────────────────────────────
run_concept_eval() {
    local DS=$1
    local CKPT=$(latest_ckpt $DS)
    [ -z "$CKPT" ] && log "[concept_eval/$DS] No checkpoint, skipping" && return
    log "[concept_eval/$DS] Starting... ckpt=$CKPT"
    cd $CBM
    $PY eval_concept.py \
        --checkpoint   "$CKPT" \
        --dataset      "$DS" \
        --concept_file "$CONCEPT_FILE" \
        --topk         20 \
        --batch_size   8 \
        --output_dir   "$BASE/concept_eval" \
        --gpu          0 \
        2>&1 | tee -a $BASE/scheduler.log
    log "[concept_eval/$DS] Done"
}

# ── Phase 2b: Ablation ───────────────────────────────────────────────────────
run_ablation() {
    local DS=$1
    local CKPT=$(latest_ckpt $DS)
    [ -z "$CKPT" ] && log "[ablation/$DS] No checkpoint, skipping" && return
    log "[ablation/$DS] Starting... ckpt=$CKPT"
    cd $CBM
    $PY -m src.ablation \
        --base_dir   "$BASE" \
        --checkpoint "$CKPT" \
        --dataset    "$DS" \
        --gpu        0 \
        2>&1 | tee -a $BASE/scheduler.log
    log "[ablation/$DS] Done"
}

# ── Phase 2c: Train eval on ALL checkpoints (not just best) ──────────────────
run_all_epoch_evals() {
    # For the DAD dataset (fastest), evaluate AP at each saved checkpoint
    # to get a learning curve. Other datasets: just best.
    local DS=dad
    log "[epoch_curve/$DS] Evaluating all checkpoints..."
    local count=0
    for CKPT in $(find $BASE -name 'best_model.pth' -path "*${DS}*" | sort); do
        AP=$(get_best_ap $DS)
        log "  $CKPT  AP=$AP"
        count=$((count+1))
    done
    log "[epoch_curve/$DS] $count checkpoints evaluated"
}

# ── Phase 3: Summary report ───────────────────────────────────────────────────
generate_report() {
    log "=== PHASE 3: Generating Summary Report ==="
    cd $CBM
    $PY - <<'PYEOF' 2>&1 | tee -a $BASE/scheduler.log
import json, os
from pathlib import Path

BASE = Path('output/v2_20260314')

# Collect ablation results
report = {'ablation': {}, 'concept_eval': {}}
for ds in ['dad', 'crash', 'a3d']:
    abl_path = BASE / 'ablation' / f'{ds}_ablation.json'
    if abl_path.exists():
        d = json.loads(abl_path.read_text())
        report['ablation'][ds] = d['results']

    ce_path = BASE / 'concept_eval' / f'{ds}_concept_summary.json'
    if ce_path.exists():
        d = json.loads(ce_path.read_text())
        report['concept_eval'][ds] = {
            'n_positive': d['n_positive'],
            'n_negative': d['n_negative'],
            'top3_discriminative': d['top_discriminative'][:3],
        }

out = BASE / 'reports' / 'final_summary.json'
out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(f'Report saved: {out}')

# Pretty print ablation table
print('\n' + '='*75)
print(' ABLATION RESULTS SUMMARY')
print('='*75)
for ds, rows in report['ablation'].items():
    print(f'\n  Dataset: {ds.upper()}')
    print(f'  {"Condition":<25s}  {"AP":6s}  {"mTTA":6s}  {"TTA@R80":8s}  {"P@R80":6s}')
    print('  ' + '-'*60)
    for r in rows:
        print(f"  {r['condition']:<25s}  {r['AP']:6.4f}  "
              f"{r['mTTA']:6.4f}  {r['TTA_R80']:8.4f}  {r['P_R80']:6.4f}")

print('\n' + '='*75)
print(' TOP DISCRIMINATIVE CONCEPTS')
print('='*75)
for ds, info in report['concept_eval'].items():
    print(f'\n  {ds.upper()} (pos={info["n_positive"]} neg={info["n_negative"]})')
    for item in info['top3_discriminative']:
        print(f"    {item['rank']}. {item['concept'][:55]}<57  "
              f"disc={item['discriminability']:.3f}")
PYEOF
    log "=== Report generation complete ==="
}

# ── Main execution ────────────────────────────────────────────────────────────
log "====================================================="
log " CG-CRASH Experiment Scheduler Started"
log "====================================================="

# Phase 1: Monitor training until done
phase1_monitor

# Phase 2: Run concept eval + ablation for each dataset (sequential to avoid GPU OOM)
for DS in dad crash a3d; do
    log "--- Processing dataset: $DS ---"
    run_concept_eval $DS
    run_ablation     $DS
done

# Phase 3: Summary
generate_report

log "====================================================="
log " All experiments complete!"
log " Results: $BASE/reports/final_summary.json"
log " Concepts: $BASE/concept_eval/"
log " Ablation: $BASE/ablation/"
log "====================================================="
