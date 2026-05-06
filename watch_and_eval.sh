#!/bin/bash
# watch_and_eval.sh
# Watches for best_model.pth to appear in v2_20260314 output dirs,
# then immediately runs concept eval + ablation for that dataset.
# Run in background: nohup bash watch_and_eval.sh > output/v2_20260314/watch_eval.log 2>&1 &

PY=/data/sony/anaconda3/bin/python
BASE=/data/sony/LFCRASH/LFCRASH-CBM/output/v2_20260314
CBM_DIR=/data/sony/LFCRASH/LFCRASH-CBM
CONCEPT_FILE=/data/sony/LFCRASH/000_all_concept_set.txt
LOG=$BASE/watch_eval.log

export CUDA_VISIBLE_DEVICES=6

echo "[$(date '+%H:%M:%S')] watch_and_eval.sh started" | tee -a $LOG
echo "[$(date '+%H:%M:%S')] Watching: $BASE" | tee -a $LOG

# Track which datasets have already been evaluated
declare -A DONE
DONE[dad]=0
DONE[crash]=0
DONE[a3d]=0

while true; do
    for DS in dad crash a3d; do
        if [ "${DONE[$DS]}" = "1" ]; then
            continue
        fi

        # Look for best_model.pth in any subdir matching the dataset
        CKPT=$(find $BASE -name 'best_model.pth' -path "*${DS}*" 2>/dev/null | sort | tail -1)
        if [ -z "$CKPT" ]; then
            continue
        fi

        echo "" | tee -a $LOG
        echo "[$(date '+%H:%M:%S')] Found checkpoint for $DS: $CKPT" | tee -a $LOG

        # --- Concept Eval ---
        echo "[$(date '+%H:%M:%S')] Running concept eval for $DS..." | tee -a $LOG
        cd $CBM_DIR
        $PY eval_concept.py \
            --checkpoint   "$CKPT" \
            --dataset      "$DS" \
            --concept_file "$CONCEPT_FILE" \
            --topk         20 \
            --batch_size   8 \
            --output_dir   "$BASE/concept_eval" \
            --gpu          0 \
            2>&1 | tee -a $LOG
        echo "[$(date '+%H:%M:%S')] Concept eval done for $DS" | tee -a $LOG

        # --- Ablation ---
        echo "[$(date '+%H:%M:%S')] Running ablation for $DS..." | tee -a $LOG
        $PY -m src.ablation \
            --base_dir   "$BASE" \
            --checkpoint "$CKPT" \
            --dataset    "$DS" \
            --gpu        0 \
            2>&1 | tee -a $LOG
        echo "[$(date '+%H:%M:%S')] Ablation done for $DS" | tee -a $LOG

        DONE[$DS]=1
    done

    # Exit when all three are done
    if [ "${DONE[dad]}" = "1" ] && [ "${DONE[crash]}" = "1" ] && [ "${DONE[a3d]}" = "1" ]; then
        echo "[$(date '+%H:%M:%S')] All datasets evaluated. Exiting." | tee -a $LOG
        break
    fi

    sleep 60   # check every 60 seconds
done
