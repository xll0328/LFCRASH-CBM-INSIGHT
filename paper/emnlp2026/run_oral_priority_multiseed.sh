#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
LAUNCHER="$ROOT/run_controlled_ontology_experiment.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODE="dry-run"
GPU_LIST="0,1,2,3,4,5,6,7"
SEEDS=(42 123 3407)
DATASETS=(dad a3d)
CONCEPT_SETS=(historical_full risk_core_v1 perfect_v1)
TARGET_MODE="missing-only"
AUDIT_JSON="$ROOT/output/emnlp2026_support/multiseed_ontology_status.json"

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
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs provided"
  exit 3
fi

echo "[oral-priority] mode=$MODE"
echo "[oral-priority] gpus=${GPUS[*]}"
echo "[oral-priority] seeds=${SEEDS[*]}"
echo "[oral-priority] target_mode=$TARGET_MODE"

declare -a TARGETS=()
if [[ "$TARGET_MODE" == "all" ]]; then
  for dataset in "${DATASETS[@]}"; do
    for concept_set in "${CONCEPT_SETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        TARGETS+=("$dataset $concept_set $seed")
      done
    done
  done
else
  if [[ ! -f "$AUDIT_JSON" ]]; then
    "$PYTHON_BIN" "$ROOT/paper/emnlp2026/audit_multiseed_ontology_runs.py" >/dev/null
  fi
  mapfile -t TARGETS < <(
    "$PYTHON_BIN" - "$AUDIT_JSON" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

for row in data["rows"]:
    for seed in row["missing_seeds"]:
        print(f"{row['dataset']} {row['concept_set']} {seed}")
PY
  )
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "[oral-priority] no targets selected"
  exit 0
fi

i=0
for target in "${TARGETS[@]}"; do
  read -r dataset concept_set seed <<< "$target"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  tag="${dataset}_shared_${concept_set}_s${seed}"
  cmd=(
    bash "$LAUNCHER" "$dataset" "$concept_set" "$gpu" "$tag"
    --seed "$seed"
  )
  if [[ "$MODE" == "dry-run" ]]; then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    log_path="$ROOT/logs/${tag}.launch.log"
    printf '[launch gpu=%s]' "$gpu"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    setsid -f "${cmd[@]}" >"$log_path" 2>&1 < /dev/null
  fi
  i=$((i + 1))
done

if [[ "$MODE" == "execute" ]]; then
  echo "[oral-priority] launched all jobs"
  echo "[oral-priority] monitor with:"
  echo "  python $ROOT/paper/emnlp2026/audit_multiseed_ontology_runs.py"
fi
