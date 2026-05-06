#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sony/LFCRASH/LFCRASH-CBM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONCEPT_FILE="$ROOT/output/concept_sets/perfect_concept_set_v1.txt"
LOG="$ROOT/output/perfect_v1_winner_local_watch.log"

mkdir -p "$ROOT/output/dad_ac" "$ROOT/output/a3d_ac"

ts() { date '+%F %T'; }

CURRENT_KEYS=(
  dad_ac_perfect_v1_h384_short_q1
  dad_ac_perfect_v1_h384_lowac_short_q2
  dad_multi_perfect_v1_h384_short_q3
  a3d_ac_perfect_v1_minuscgta_short_q1
  a3d_ac_perfect_v1_minuscgta_h384_short_q2
  a3d_ac_perfect_v1_minuscgta_nosparse_short_q3
  a3d_ac_perfect_v1_minuscgta_lowac_short_q4
)

count_active() {
  python3 - <<'PY'
import subprocess
keys = [
  'dad_ac_perfect_v1_h384_short_q1',
  'dad_ac_perfect_v1_h384_lowac_short_q2',
  'dad_multi_perfect_v1_h384_short_q3',
  'a3d_ac_perfect_v1_minuscgta_short_q1',
  'a3d_ac_perfect_v1_minuscgta_h384_short_q2',
  'a3d_ac_perfect_v1_minuscgta_nosparse_short_q3',
  'a3d_ac_perfect_v1_minuscgta_lowac_short_q4',
]
out = subprocess.check_output(['ps','-eo','args'], text=True)
count = 0
for k in keys:
    if any(k in line and 'python3 - <<' not in line for line in out.splitlines()):
        count += 1
print(count)
PY
}

run_dad() {
  local tag="$1"
  shift
  local out_dir="$ROOT/output/dad_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(ts)] START $tag" | tee -a "$LOG"
  "$PYTHON_BIN" "$ROOT/train_dad_ac.py" "$@" > "$out_dir/nohup.log" 2>&1
  echo "[$(ts)] END $tag exit=$?" | tee -a "$LOG"
}

run_multi() {
  local dataset="$1"
  local tag="$2"
  shift 2
  local out_dir="$ROOT/output/${dataset}_ac/$tag"
  mkdir -p "$out_dir"
  echo "[$(ts)] START $tag" | tee -a "$LOG"
  "$PYTHON_BIN" "$ROOT/train_multi.py" --dataset "$dataset" "$@" > "$out_dir/nohup.log" 2>&1
  echo "[$(ts)] END $tag exit=$?" | tee -a "$LOG"
}

echo "[$(ts)] Winner-local watcher started" > "$LOG"
while true; do
  active=$(count_active | tr -d '[:space:]')
  echo "[$(ts)] Waiting for current short-horizon runs; active=$active" >> "$LOG"
  if [ "$active" = "0" ]; then
    break
  fi
  sleep 300
done

echo "[$(ts)] Launching winner-local focused search" | tee -a "$LOG"

(run_dad dad_ac_perfect_v1_h384_tinyalign_local_q1 \
  --gpu 0 --tag dad_ac_perfect_v1_h384_tinyalign_local_q1 \
  --epochs 60 --warmup_epochs 12 --cbm_ramp_epochs 18 \
  --batch_size 16 --lr 3e-4 --h_dim 384 --z_dim 384 \
  --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 --ac_gamma 0.95 --ac_entropy 0.01 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_dad dad_ac_perfect_v1_h384_lowac_weaksparse_local_q2 \
  --gpu 2 --tag dad_ac_perfect_v1_h384_lowac_weaksparse_local_q2 \
  --epochs 60 --warmup_epochs 12 --cbm_ramp_epochs 18 \
  --batch_size 16 --lr 3e-4 --h_dim 384 --z_dim 384 \
  --lambda_align 0 --lambda_sparse 1e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.1 --lambda_ac_value 0.1 --ac_gamma 0.95 --ac_entropy 0.01 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_multi dad dad_multi_perfect_v1_h384_lowac_local_q3 \
  --gpu 3 --tag dad_multi_perfect_v1_h384_lowac_local_q3 \
  --epochs 50 --warmup_epochs 10 --cbm_ramp_epochs 15 \
  --batch_size 16 --lr 3e-4 --h_dim 384 \
  --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.1 --lambda_ac_value 0.1 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_multi a3d a3d_ac_perfect_v1_minuscgta_lowac_nosparse_local_q1 \
  --gpu 1 --tag a3d_ac_perfect_v1_minuscgta_lowac_nosparse_local_q1 \
  --epochs 50 --warmup_epochs 10 --cbm_ramp_epochs 15 \
  --batch_size 16 --lr 3e-4 --h_dim 256 --disable_cgta \
  --lambda_align 0 --lambda_sparse 0 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.05 --lambda_ac_value 0.1 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_multi a3d a3d_ac_perfect_v1_minuscgta_lowac_norecon_local_q2 \
  --gpu 5 --tag a3d_ac_perfect_v1_minuscgta_lowac_norecon_local_q2 \
  --epochs 50 --warmup_epochs 10 --cbm_ramp_epochs 15 \
  --batch_size 16 --lr 3e-4 --h_dim 256 --disable_cgta \
  --lambda_align 0 --lambda_sparse 5e-5 --lambda_recon 0 \
  --lambda_ac_policy 0.05 --lambda_ac_value 0.1 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

(run_multi a3d a3d_ac_perfect_v1_minuscgta_tinyalign_local_q3 \
  --gpu 6 --tag a3d_ac_perfect_v1_minuscgta_tinyalign_local_q3 \
  --epochs 50 --warmup_epochs 10 --cbm_ramp_epochs 15 \
  --batch_size 16 --lr 3e-4 --h_dim 256 --disable_cgta \
  --lambda_align 1e-6 --lambda_sparse 5e-5 --lambda_recon 1e-4 \
  --lambda_ac_policy 0.3 --lambda_ac_value 0.3 \
  --num_concepts 80 --concept_file "$CONCEPT_FILE" --eval_every 5 --num_workers 4) &

wait
echo "[$(ts)] Winner-local focused search done" | tee -a "$LOG"
