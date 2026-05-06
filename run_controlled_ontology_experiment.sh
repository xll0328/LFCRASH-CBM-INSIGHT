#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <dataset: dad|a3d|crash> <concept_set: historical_full|risk_core_v1|perfect_v1> <gpu> <tag> [extra args...]"
  exit 1
fi

dataset="$1"
concept_set="$2"
gpu="$3"
tag="$4"
shift 4

case "$dataset" in
  dad|a3d|crash) ;;
  *)
    echo "Unsupported dataset: $dataset"
    exit 2
    ;;
esac

case "$concept_set" in
  historical_full)
    concept_file="/data/sony/LFCRASH/000_all_concept_set.txt"
    num_concepts=837
    ;;
  risk_core_v1)
    concept_file="$ROOT/output/concept_sets/risk_core_concept_set_v1.txt"
    num_concepts=30
    ;;
  perfect_v1)
    concept_file="$ROOT/output/concept_sets/perfect_concept_set_v1.txt"
    num_concepts=80
    ;;
  *)
    echo "Unsupported concept_set: $concept_set"
    exit 3
    ;;
esac

cmd=(
  "$PYTHON_BIN" "$ROOT/train_multi.py"
  --dataset "$dataset"
  --gpu "$gpu"
  --tag "$tag"
  --num_concepts "$num_concepts"
  --concept_file "$concept_file"
)

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[run_controlled_ontology] dataset=$dataset concept_set=$concept_set gpu=$gpu tag=$tag"
echo "[run_controlled_ontology] concept_file=$concept_file num_concepts=$num_concepts"
printf '[run_controlled_ontology] cmd:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
