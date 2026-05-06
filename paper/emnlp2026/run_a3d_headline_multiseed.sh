#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
OUT="$ROOT/output/sota_push"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="dry-run"
GPU_LIST="0,1"
SEEDS=(123 3407)
TARGET_MODE="missing-only"
STALE_AFTER_MIN=30
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

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
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --all)
      TARGET_MODE="all"
      shift
      ;;
    --stale-after-min)
      STALE_AFTER_MIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
mkdir -p "$OUT"

echo "[a3d-headline] mode=$MODE"
echo "[a3d-headline] gpus=${GPUS[*]}"
echo "[a3d-headline] seeds=${SEEDS[*]}"
echo "[a3d-headline] target_mode=$TARGET_MODE"
echo "[a3d-headline] stale_after_min=$STALE_AFTER_MIN"

i=0
for seed in "${SEEDS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  tag="a3d_sota_s${seed}"
  out_dir="$OUT/$tag"
  results_path="$out_dir/results.json"
  train_log="$out_dir/train.log"
  backup_dir="$OUT/${tag}.pre_rerun_${STAMP}"
  cmd=(
    "$PYTHON_BIN" -B "$ROOT/train_enhanced.py"
    --dataset a3d
    --gpu "$gpu"
    --epochs 120
    --batch_size 16
    --lr 2e-4
    --weight_decay 1e-5
    --h_dim 384
    --z_dim 256
    --lambda_align 1e-5
    --lambda_sparse 1e-4
    --lambda_recon 5e-4
    --num_concepts 837
    --num_workers 0
    --eval_interval 5
    --seed "$seed"
    --output_dir "$OUT"
    --tag "$tag"
  )

  if [[ "$TARGET_MODE" == "missing-only" && -f "$results_path" ]]; then
    echo "[skip completed] $tag -> $results_path"
    i=$((i + 1))
    continue
  fi

  relaunch_stale=0
  if [[ "$TARGET_MODE" == "missing-only" && -f "$train_log" && ! -f "$results_path" ]]; then
    now_ts="$(date +%s)"
    log_ts="$(stat -c %Y "$train_log")"
    age_min="$(((now_ts - log_ts) / 60))"
    if (( age_min < STALE_AFTER_MIN )); then
      echo "[skip active-ish] $tag -> $train_log (${age_min} min old)"
      i=$((i + 1))
      continue
    fi
    relaunch_stale=1
  fi

  if [[ "$MODE" == "dry-run" ]]; then
    if [[ "$TARGET_MODE" == "all" && -d "$out_dir" ]]; then
      printf '[dry-run][backup] mv %q %q\n' "$out_dir" "$backup_dir"
    elif (( relaunch_stale )) && [[ -d "$out_dir" ]]; then
      printf '[dry-run][stale-backup] mv %q %q\n' "$out_dir" "$backup_dir"
    fi
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    if [[ "$TARGET_MODE" == "all" && -d "$out_dir" ]]; then
      mv "$out_dir" "$backup_dir"
    elif (( relaunch_stale )) && [[ -d "$out_dir" ]]; then
      mv "$out_dir" "$backup_dir"
    fi
    log_path="$OUT/${tag}.launch.log"
    printf '[launch gpu=%s]' "$gpu"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    setsid -f "${cmd[@]}" >"$log_path" 2>&1 < /dev/null
  fi
  i=$((i + 1))
done

if [[ "$MODE" == "execute" ]]; then
  echo "[a3d-headline] launched all jobs"
  echo "[a3d-headline] monitor with:"
  echo "  python $ROOT/paper/emnlp2026/audit_a3d_headline_multiseed.py"
fi
