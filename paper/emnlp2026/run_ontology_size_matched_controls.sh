#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
MODE="dry-run"
GPU_LIST="5,7"
DATASETS=("dad" "a3d")
SEEDS=(42 123 3407)
SEQUENTIAL_WORKERS=1
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      MODE="execute"
      shift
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --datasets)
      IFS=',' read -r -a DATASETS <<< "$2"
      shift 2
      ;;
    --seeds)
      IFS=',' read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    --no-sequential-workers)
      SEQUENTIAL_WORKERS=0
      shift
      ;;
    --sequential-workers)
      SEQUENTIAL_WORKERS=1
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs provided"
  exit 3
fi

python3 "$ROOT/paper/emnlp2026/build_historical_size_matched_subsets.py" --seed 2026 >/tmp/size_subset_build.log 2>&1

declare -a CONDITIONS=(
  "historical_stratified_30|30|$ROOT/output/concept_sets/historical_full_stratified_30.txt"
  "risk_core_v1|30|$ROOT/output/concept_sets/risk_core_concept_set_v1.txt"
  "historical_stratified_80|80|$ROOT/output/concept_sets/historical_full_stratified_80.txt"
  "perfect_v1|80|$ROOT/output/concept_sets/perfect_concept_set_v1.txt"
)

is_tag_running() {
  local tag="$1"
  ps -ef | grep -F "train_multi.py" | grep -F -- "--tag $tag" >/dev/null 2>&1
}

result_path_for() {
  local dataset="$1"
  local tag="$2"
  echo "$ROOT/output/${dataset}_ac/${tag}/results.json"
}

TARGETS=()
for dataset in "${DATASETS[@]}"; do
  for c in "${CONDITIONS[@]}"; do
    IFS='|' read -r cond_name cond_k cond_file <<< "$c"
    if [[ ! -f "$cond_file" ]]; then
      echo "[missing concept file] $cond_file"
      exit 4
    fi
    for seed in "${SEEDS[@]}"; do
      tag="${dataset}_sizectrl_${cond_name}_s${seed}"
      out_json="$(result_path_for "$dataset" "$tag")"
      if [[ -f "$out_json" ]]; then
        echo "[skip existing] $tag"
        continue
      fi
      if is_tag_running "$tag"; then
        echo "[skip running] $tag"
        continue
      fi
      TARGETS+=("$dataset|$cond_name|$cond_k|$cond_file|$seed")
    done
  done
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "[size-matched-controls] no targets to launch"
  exit 0
fi

echo "[size-matched-controls] mode=$MODE"
echo "[size-matched-controls] gpus=${GPUS[*]}"
echo "[size-matched-controls] datasets=${DATASETS[*]}"
echo "[size-matched-controls] seeds=${SEEDS[*]}"
echo "[size-matched-controls] sequential_workers=$SEQUENTIAL_WORKERS"
echo "[size-matched-controls] pending_targets=${#TARGETS[@]}"

if [[ "$MODE" == "execute" && "$SEQUENTIAL_WORKERS" == "1" ]]; then
  QUEUE_DIR="$ROOT/output/emnlp2026_support/size_matched_controls_queue_${STAMP}"
  mkdir -p "$QUEUE_DIR"

  for gpu in "${GPUS[@]}"; do
    worker_script="$QUEUE_DIR/gpu_${gpu}.sh"
    cat > "$worker_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
EOF
    chmod +x "$worker_script"
  done

  i=0
  for t in "${TARGETS[@]}"; do
    IFS='|' read -r dataset cond_name cond_k cond_file seed <<< "$t"
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    tag="${dataset}_sizectrl_${cond_name}_s${seed}"
    worker_script="$QUEUE_DIR/gpu_${gpu}.sh"
    log_path="$ROOT/logs/${tag}.launch.log"

    printf 'python3 %q --dataset %q --gpu %q --tag %q --num_concepts %q --concept_file %q --seed %q --num_workers 0 > %q 2>&1\n' \
      "$ROOT/train_multi.py" \
      "$dataset" "$gpu" "$tag" "$cond_k" "$cond_file" "$seed" "$log_path" >> "$worker_script"
    echo "echo '[done] $tag'" >> "$worker_script"
    i=$((i + 1))
  done

  for gpu in "${GPUS[@]}"; do
    worker_script="$QUEUE_DIR/gpu_${gpu}.sh"
    if [[ -s "$worker_script" ]]; then
      echo "[launch worker gpu=$gpu] $worker_script"
      setsid -f bash "$worker_script" >"$QUEUE_DIR/gpu_${gpu}.log" 2>&1 < /dev/null
    fi
  done

  echo "[size-matched-controls] queue_dir=$QUEUE_DIR"
else
  i=0
  for t in "${TARGETS[@]}"; do
    IFS='|' read -r dataset cond_name cond_k cond_file seed <<< "$t"
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    tag="${dataset}_sizectrl_${cond_name}_s${seed}"
    cmd=(
      python3 "$ROOT/train_multi.py"
      --dataset "$dataset"
      --gpu "$gpu"
      --tag "$tag"
      --num_concepts "$cond_k"
      --concept_file "$cond_file"
      --seed "$seed"
      --num_workers 0
    )
    if [[ "$MODE" == "dry-run" ]]; then
      printf '[dry-run][gpu=%s]' "$gpu"
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      log_path="$ROOT/logs/${tag}.launch.log"
      printf '[launch][gpu=%s]' "$gpu"
      printf ' %q' "${cmd[@]}"
      printf '\n'
      setsid -f "${cmd[@]}" >"$log_path" 2>&1 < /dev/null
    fi
    i=$((i + 1))
  done
fi

if [[ "$MODE" == "execute" ]]; then
  echo "[size-matched-controls] launched"
  echo "[size-matched-controls] next: write an audit script after first completions"
fi
