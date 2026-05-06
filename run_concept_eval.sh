#!/bin/bash
# run_concept_eval.sh
# Run concept evaluation on best checkpoints after training completes.
# Usage: bash run_concept_eval.sh [output_base_dir]
#
# Auto-discovers the latest best_model.pth for each dataset.

set -e
CUDA_VISIBLE_DEVICES=6
PY=/data/sony/anaconda3/bin/python
SCRIPT=/data/sony/LFCRASH/LFCRASH-CBM/eval_concept.py
BASE=${1:-/data/sony/LFCRASH/LFCRASH-CBM/output/v2_20260314}
CONCEPT_FILE=/data/sony/LFCRASH/000_all_concept_set.txt
OUT_DIR=$BASE/concept_eval

echo "========================================"
echo " CG-CRASH Concept Evaluation"
echo " Base: $BASE"
echo " Output: $OUT_DIR"
echo "========================================"

for DATASET in dad crash a3d; do
    # Find latest best_model.pth for this dataset
    CKPT=$(find $BASE -name 'best_model.pth' -path "*${DATASET}*" 2>/dev/null | sort | tail -1)
    if [ -z "$CKPT" ]; then
        echo "[SKIP] No checkpoint found for $DATASET in $BASE"
        continue
    fi
    echo ""
    echo ">>> Evaluating $DATASET"
    echo "    Checkpoint: $CKPT"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PY $SCRIPT \
        --checkpoint  "$CKPT" \
        --dataset     "$DATASET" \
        --concept_file "$CONCEPT_FILE" \
        --topk        20 \
        --batch_size  8 \
        --output_dir  "$OUT_DIR" \
        --gpu         0
    echo "    Done: $DATASET"
done

echo ""
echo "========================================"
echo " All concept evals complete."
echo " Results in: $OUT_DIR"
echo "========================================"
