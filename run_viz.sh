#!/bin/bash
mkdir -p /data/sony/LFCRASH/LFCRASH-CBM/paper/figures
exec python3 /data/sony/LFCRASH/LFCRASH-CBM/visualize_concepts.py \
  --ckpt /data/sony/LFCRASH/LFCRASH-CBM/output/dad_ac/dad_ac_v3_fixed_lr/best_model.pt \
  --tag dad_ac_v3_fixed_lr \
  --gpu 6 \
  --n_cases 3 \
  --out_dir /data/sony/LFCRASH/LFCRASH-CBM/paper/figures
