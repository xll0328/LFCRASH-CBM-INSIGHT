#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
OUT="$ROOT/output/dad_curriculum"
TRAINER="$ROOT/train_dad_curriculum.py"
PYTHON_BIN="${PYTHON_BIN:-/data/sony/anaconda3/bin/python}"

MODE="dry-run"
GPU_LIST="0,1,7"
SEED_LIST="314,2718,3407"
TARGET_MODE="missing-only"
LIMIT=999
STALE_AFTER_MIN=45

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --execute)
      MODE="execute"
      shift
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --seeds)
      SEED_LIST="$2"
      shift 2
      ;;
    --all)
      TARGET_MODE="all"
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --stale-after-min)
      STALE_AFTER_MIN="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
IFS=',' read -r -a SEEDS <<< "$SEED_LIST"

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs provided"
  exit 3
fi

if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "No seeds provided"
  exit 4
fi

mkdir -p "$OUT"

tag_live_pids() {
  local tag="$1"
  ps -eo pid=,comm=,args= | awk -v trainer="$TRAINER" -v tag_pat="--tag ${tag}" '
    ($2 == "python" || $2 == "python3") && index($0, trainer) && index($0, tag_pat) { print $1 }
  '
}

echo "[dad-curriculum-recovery] mode=$MODE"
echo "[dad-curriculum-recovery] gpus=${GPUS[*]}"
echo "[dad-curriculum-recovery] seeds=${SEEDS[*]}"
echo "[dad-curriculum-recovery] target_mode=$TARGET_MODE"
echo "[dad-curriculum-recovery] limit=$LIMIT"
echo "[dad-curriculum-recovery] stale_after_min=$STALE_AFTER_MIN"

i=0
launched=0
for seed in "${SEEDS[@]}"; do
  tag="dad_curriculum_v2_s${seed}"
  session="lf_dcv2_s${seed}"
  out_dir="$OUT/$tag"
  results_path="$out_dir/results.json"
  train_log="$out_dir/train.log"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  cmd=(
    "$PYTHON_BIN" "$TRAINER"
    --gpu "$gpu"
    --tag "$tag"
    --seed "$seed"
    --epochs 150
    --warmup_epochs 20
    --eval_every 5
    --batch_size 16
    --lr 3e-4
    --weight_decay 1e-4
    --h_dim 256
    --z_dim 128
    --lambda_align 1e-4
    --lambda_sparse 0
    --lambda_recon 1e-2
    --num_workers 4
  )

  if [[ "$TARGET_MODE" == "missing-only" && -f "$results_path" ]]; then
    echo "[skip completed] $tag -> $results_path"
    i=$((i + 1))
    continue
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip live-session] $tag -> $session"
    i=$((i + 1))
    continue
  fi

  if [[ "$TARGET_MODE" == "missing-only" ]]; then
    live_pids="$(tag_live_pids "$tag")"
    if [[ -n "$live_pids" ]]; then
      live_csv="$(echo "$live_pids" | paste -sd, -)"
      echo "[skip live-process] $tag -> pids=$live_csv"
      i=$((i + 1))
      continue
    fi
  fi

  if [[ "$TARGET_MODE" == "missing-only" && -f "$train_log" && ! -f "$results_path" ]]; then
    now_ts="$(date +%s)"
    log_ts="$(stat -c %Y "$train_log")"
    age_min="$(((now_ts - log_ts) / 60))"
    if (( age_min < STALE_AFTER_MIN )); then
      echo "[skip active-ish] $tag -> $train_log (${age_min} min old)"
      i=$((i + 1))
      continue
    fi
  fi

  if (( launched >= LIMIT )); then
    echo "[limit reached] stopping after $launched launches"
    break
  fi

  if [[ "$MODE" == "dry-run" ]]; then
    printf '[dry-run session=%s gpu=%s seed=%s]' "$session" "$gpu" "$seed"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    printf -v train_cmd '%q ' "${cmd[@]}"
    train_cmd="${train_cmd% }"
    printf -v run_cmd 'cd %q && exec %s' "$ROOT" "$train_cmd"
    tmux new-session -d -s "$session" "$run_cmd"
    printf '[launch session=%s gpu=%s seed=%s tag=%s]\n' "$session" "$gpu" "$seed" "$tag"
  fi

  launched=$((launched + 1))
  i=$((i + 1))
done

if [[ "$MODE" == "execute" ]]; then
  echo "[dad-curriculum-recovery] launched=$launched"
  echo "[dad-curriculum-recovery] monitor with:"
  echo "  tmux ls | rg 'lf_dcv2'"
  echo "  tail -f $OUT/dad_curriculum_v2_s314/train.log"
  echo "  python $ROOT/paper/emnlp2026/summarize_dad_curriculum_recovery.py"
fi
