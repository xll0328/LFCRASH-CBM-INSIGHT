#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="dry-run"
GPU_LIST="0,1,2,3,4,5,6,7"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
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

echo "[rerun-ontology] mode=$MODE"
echo "[rerun-ontology] gpus=${GPUS[*]}"
echo "[rerun-ontology] stamp=$STAMP"
echo "[rerun-ontology] target_mode=$TARGET_MODE"

if [[ "$TARGET_MODE" == "all" ]]; then
  TARGETS=(
    "dad historical_full 42"
    "dad historical_full 123"
    "dad historical_full 3407"
    "dad risk_core_v1 42"
    "dad risk_core_v1 123"
    "dad risk_core_v1 3407"
    "dad perfect_v1 42"
    "dad perfect_v1 123"
    "dad perfect_v1 3407"
    "a3d historical_full 42"
    "a3d historical_full 123"
    "a3d historical_full 3407"
    "a3d risk_core_v1 42"
    "a3d risk_core_v1 123"
    "a3d risk_core_v1 3407"
    "a3d perfect_v1 42"
    "a3d perfect_v1 123"
    "a3d perfect_v1 3407"
  )
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
  echo "[rerun-ontology] no targets selected"
  exit 0
fi

QUEUE_DIR="$ROOT/output/emnlp2026_support/ontology_rerun_queue_${STAMP}"
mkdir -p "$QUEUE_DIR"

declare -a worker_scripts=()
for gpu in "${GPUS[@]}"; do
  worker_script="$QUEUE_DIR/gpu_${gpu}.sh"
  worker_log="$QUEUE_DIR/gpu_${gpu}.log"
  worker_scripts+=("$worker_script")
  cat > "$worker_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
EOF
  chmod +x "$worker_script"
  : > "$worker_log"
done

i=0
for target in "${TARGETS[@]}"; do
  read -r dataset concept_set seed <<< "$target"
  tag="${dataset}_shared_${concept_set}_s${seed}"
  out_dir="$ROOT/output/${dataset}_ac/${tag}"
  backup_dir="$ROOT/output/${dataset}_ac/${tag}.pre_rerun_${STAMP}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  worker_script="$QUEUE_DIR/gpu_${gpu}.sh"
  log_path="$ROOT/logs/${tag}.launch.log"

  if [[ "$MODE" == "dry-run" ]]; then
    if [[ -d "$out_dir" ]]; then
      printf '[dry-run][gpu=%s] mv %q %q\n' "$gpu" "$out_dir" "$backup_dir"
    fi
    printf '[dry-run][gpu=%s] bash %q %q %q %q %q --seed %q --num_workers 0\n' \
      "$gpu" "$ROOT/run_controlled_ontology_experiment.sh" \
      "$dataset" "$concept_set" "$gpu" "$tag" "$seed"
  else
    if [[ -d "$out_dir" ]]; then
      printf 'mv %q %q\n' "$out_dir" "$backup_dir" >> "$worker_script"
    fi
    printf 'bash %q %q %q %q %q --seed %q --num_workers 0 > %q 2>&1\n' \
      "$ROOT/run_controlled_ontology_experiment.sh" \
      "$dataset" "$concept_set" "$gpu" "$tag" "$seed" "$log_path" >> "$worker_script"
    echo "echo '[done] $tag'" >> "$worker_script"
  fi
  i=$((i + 1))
done

if [[ "$MODE" == "execute" ]]; then
  for gpu in "${GPUS[@]}"; do
    worker_script="$QUEUE_DIR/gpu_${gpu}.sh"
    worker_log="$QUEUE_DIR/gpu_${gpu}.log"
    if [[ -s "$worker_script" ]]; then
      echo "[launch worker gpu=$gpu] $worker_script"
      setsid -f bash "$worker_script" >"$worker_log" 2>&1 < /dev/null
    fi
  done
  "$PYTHON_BIN" "$ROOT/paper/emnlp2026/refresh_emnlp_status.py" >/dev/null || true
  echo "[rerun-ontology] launched sequential workers"
  echo "[rerun-ontology] queue dir: $QUEUE_DIR"
  echo "[rerun-ontology] monitor with:"
  echo "  python $ROOT/paper/emnlp2026/audit_multiseed_ontology_runs.py"
fi
