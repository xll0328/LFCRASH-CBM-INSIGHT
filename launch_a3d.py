#!/usr/bin/env python3
import subprocess, os
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '3'
log = open('/data/sony/LFCRASH/LFCRASH-CBM/output/run_20260314_151328/a3d2.log', 'w')
p = subprocess.Popen(
    [
        '/data/sony/anaconda3/bin/python', 'train.py',
        '--dataset', 'a3d',
        '--epochs', '100',
        '--batch_size', '32',
        '--h_dim', '768',
        '--z_dim', '128',
        '--lr', '2.5e-6',
        '--weight_decay', '1.2e-6',
        '--lambda_align', '6.6e-4',
        '--lambda_sparse', '4.8e-3',
        '--num_concepts', '837',
        '--num_workers', '4',
        '--eval_every', '5',
        '--output_dir', 'output/run_20260314_151328',
    ],
    cwd='/data/sony/LFCRASH/LFCRASH-CBM',
    env=env,
    stdout=log,
    stderr=log,
    start_new_session=True,
)
print(f'A3D PID={p.pid}')
