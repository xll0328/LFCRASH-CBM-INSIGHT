#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/output/dad_mechanism_lightreg_block"
PYTHON_BIN="${PYTHON_BIN:-/data/sony/anaconda3/bin/python}"
MODE="dry-run"
GPU_LIST="0"
TARGET_MODE="missing-only"
LIMIT=999
STALE_AFTER_MIN=30

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
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" == "execute" && "${LFCRASH_ALLOW_GPU:-0}" != "1" ]]; then
  echo "[blocked] --execute requires explicit GPU approval via LFCRASH_ALLOW_GPU=1" >&2
  exit 3
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
mkdir -p "$OUT"

tag_live_pids() {
  local tag="$1"
  ps -eo pid=,comm=,args= | awk -v root="$ROOT/train.py" -v tag_pat="--tag ${tag}" '
    ($2 == "python" || $2 == "python3") && index($0, tag_pat) && (index($0, root) || index($0, " train.py ")) { print $1 }
  '
}

echo "[dad-mechanism-lightreg] mode=$MODE"
echo "[dad-mechanism-lightreg] gpus=${GPUS[*]}"
echo "[dad-mechanism-lightreg] target_mode=$TARGET_MODE"
echo "[dad-mechanism-lightreg] limit=$LIMIT"
echo "[dad-mechanism-lightreg] stale_after_min=$STALE_AFTER_MIN"

i=0
launched=0
for run_idx in 1 2 3; do
  tag="insight_journal_dad_lightreg_r${run_idx}"
  out_dir="$OUT/$tag"
  results_path="$out_dir/results.json"
  train_log="$out_dir/train.log"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  cmd=(
    "$PYTHON_BIN" "$ROOT/train.py"
    --dataset dad
    --gpu "$gpu"
    --epochs 40
    --batch_size 16
    --lr 2e-4
    --weight_decay 3e-5
    --h_dim 256
    --z_dim 128
    --lambda_align 3e-5
    --lambda_sparse 1e-4
    --lambda_recon 5e-4
    --num_concepts 837
    --num_workers 0
    --eval_every 2
    --patience 10
    --output_dir "$OUT"
    --tag "$tag"
  )

  if [[ "$TARGET_MODE" == "missing-only" && -f "$results_path" ]]; then
    echo "[skip completed] $tag -> $results_path"
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
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    session="lfcrash_dad_lightreg_r${run_idx}_${ts}"
    session_file="$OUT/${tag}.tmux_session"
    command_str=""
    printf -v command_str '%q ' "${cmd[@]}"
    tmux new-session -d -s "$session" "cd $(printf '%q' "$ROOT") && exec $command_str"
    echo "$session" >"$session_file"
    printf '[launch tmux=%s gpu=%s]' "$session" "$gpu"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  fi
  launched=$((launched + 1))
  i=$((i + 1))
done

if [[ "$MODE" == "execute" ]]; then
  echo "[dad-mechanism-lightreg] launched=$launched"
  echo "[dad-mechanism-lightreg] monitor with:"
  echo "  tail -f $OUT/insight_journal_dad_lightreg_r1/train.log"
  echo "  python $ROOT/paper/emnlp2026/summarize_dad_mechanism_lightreg_status.py"
fi
