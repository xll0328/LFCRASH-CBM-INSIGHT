#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
MODE="dry-run"
GPU_LIST="4"
CONCEPT_SETS=("perfect_v1")
SEEDS=(7 11 2718 314 2026)
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
    --all-concepts)
      CONCEPT_SETS=("historical_full" "risk_core_v1" "perfect_v1")
      shift
      ;;
    --perfect-only)
      CONCEPT_SETS=("perfect_v1")
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

echo "[dad-ontology-seed-extension] mode=$MODE"
echo "[dad-ontology-seed-extension] gpus=${GPUS[*]}"
echo "[dad-ontology-seed-extension] concept_sets=${CONCEPT_SETS[*]}"
echo "[dad-ontology-seed-extension] seeds=${SEEDS[*]}"

TARGETS=()
for concept_set in "${CONCEPT_SETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="dad_shared_${concept_set}_s${seed}"
    out_json="$ROOT/output/dad_ac/${tag}/results.json"
    if [[ -f "$out_json" ]]; then
      echo "[skip existing] $tag"
      continue
    fi
    TARGETS+=("$concept_set $seed")
  done
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "[dad-ontology-seed-extension] no targets to launch"
  exit 0
fi

i=0
for target in "${TARGETS[@]}"; do
  read -r concept_set seed <<< "$target"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  tag="dad_shared_${concept_set}_s${seed}"
  log_path="$ROOT/logs/${tag}.launch.log"
  cmd=(
    bash "$ROOT/run_controlled_ontology_experiment.sh"
    dad "$concept_set" "$gpu" "$tag"
    --seed "$seed"
    --num_workers 0
  )
  if [[ "$MODE" == "dry-run" ]]; then
    printf '[dry-run][gpu=%s]' "$gpu"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    printf '[launch][gpu=%s]' "$gpu"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    setsid -f "${cmd[@]}" >"$log_path" 2>&1 < /dev/null
  fi
  i=$((i + 1))
done

if [[ "$MODE" == "execute" ]]; then
  echo "[dad-ontology-seed-extension] launched"
  echo "[dad-ontology-seed-extension] monitor:"
  echo "  python3 $ROOT/paper/emnlp2026/audit_dad_ontology_seed_extension.py"
fi
