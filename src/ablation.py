#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/ablation.py
===============
Ablation experiment runner for CG-CRASH.

Runs four conditions:
  A. Full model      (use_cbm=True,  align=True,  sparse=True,  recon=True)
  B. No CBM          (use_cbm=False) — plain CRASH baseline
  C. No align        (use_cbm=True,  align=False)
  D. No sparse       (use_cbm=True,  sparse=False)

For each condition, evaluates AP, mTTA, TTA@R80, P@R80
and saves a comparison table.

Usage
-----
python -m src.ablation \
    --base_dir output/v2_20260314 \
    --dataset dad \
    --gpu 6
"""
import os, sys, json, argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from copy import deepcopy

ROOT       = Path(__file__).resolve().parent.parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT  = CRASH_ROOT / 'data'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru  import LFCRASH_CBM_GRU
from src.data_loader import DADDataset, CrashDataset, A3DDataset
from src.eval_tools  import evaluation

DS_META = {
    'dad':   {'cls': DADDataset,   'feature': 'vgg16',
               'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0,
               'phase_test': 'testing'},
    'crash': {'cls': CrashDataset, 'feature': 'vgg16',
               'x_dim': 4096, 'n_obj': 19, 'n_frames': 50,  'fps': 10.0,
               'phase_test': 'test'},
    'a3d':   {'cls': A3DDataset,   'feature': 'vgg16',
               'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0,
               'phase_test': 'test'},
}

ABLATION_CONDITIONS = [
    {'name': 'Full-CG-CRASH',    'use_cbm': True,  'no_align': False, 'no_sparse': False, 'no_recon': False},
    {'name': 'No-CBM (baseline)','use_cbm': False, 'no_align': True,  'no_sparse': True,  'no_recon': True},
    {'name': 'No-Align',         'use_cbm': True,  'no_align': True,  'no_sparse': False, 'no_recon': False},
    {'name': 'No-Sparse',        'use_cbm': True,  'no_align': False, 'no_sparse': True,  'no_recon': False},
    {'name': 'No-Recon',         'use_cbm': True,  'no_align': False, 'no_sparse': False, 'no_recon': True},
]


def collate_fn(batch):
    xs, ys, toas = zip(*batch)
    xs = torch.from_numpy(np.stack(xs)).float()
    ys = torch.from_numpy(np.stack(ys)).float()
    toa_flat = []
    for t in toas:
        if isinstance(t, (list, tuple, np.ndarray)):
            toa_flat.append(float(t[0]) if hasattr(t, '__len__') else float(t))
        elif isinstance(t, torch.Tensor):
            toa_flat.append(t.item())
        else:
            toa_flat.append(float(t))
    return xs, ys, torch.tensor(toa_flat, dtype=torch.float32)


@torch.no_grad()
def evaluate_model(model, loader, meta, device):
    model.eval()
    n_frames = meta['n_frames']
    fps      = meta['fps']
    all_pred, all_labels, all_toas = [], [], []
    for xs, ys, toas in loader:
        xs  = xs.to(device)
        B   = xs.size(0)
        _, outputs, _ = model(xs, None, None)
        T = len(outputs)
        probs = np.zeros((B, n_frames), dtype=np.float32)
        for t, out_t in enumerate(outputs):
            p = torch.softmax(out_t, dim=-1)[:, 1].cpu().numpy()
            probs[:, t] = p
        if T < n_frames:
            probs[:, T:] = probs[:, T-1:T]
        all_pred.append(probs)
        all_labels.append(ys[:, 1].numpy())
        all_toas.append(toas.numpy())
    all_pred   = np.concatenate(all_pred,   axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_toas   = np.concatenate(all_toas,   axis=0)
    return evaluation(all_pred, all_labels, all_toas, fps=fps)


def load_model_for_condition(ckpt_path, dataset, cond, device, concept_file):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        saved_args = ckpt.get('args', {})
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        saved_args = ckpt.get('config', {})
    else:
        state_dict = ckpt
        saved_args = {}

    meta = DS_META[dataset]
    model = LFCRASH_CBM_GRU(
        x_dim         = saved_args.get('x_dim',        meta['x_dim']),
        h_dim         = saved_args.get('h_dim',        256),
        z_dim         = saved_args.get('z_dim',        128),
        n_layers      = 2,
        n_obj         = meta['n_obj'],
        n_frames      = meta['n_frames'],
        fps           = meta['fps'],
        with_saa      = True,
        num_concepts  = saved_args.get('num_concepts', 837),
        concept_file  = concept_file if os.path.exists(concept_file) else None,
        lambda_align  = 0.0,
        lambda_sparse = 0.0,
        lambda_recon  = 0.0,
        use_cbm       = cond['use_cbm'],
        device        = str(device),
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def run_ablation(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    concept_file = str(ROOT.parent / '000_all_concept_set.txt')
    meta = DS_META[args.dataset]

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = sorted(
            Path(args.base_dir).glob(f'{args.dataset}*/best_model.pth'))
        if not candidates:
            raise FileNotFoundError(
                f'No best_model.pth found for {args.dataset} in {args.base_dir}')
        ckpt_path = str(candidates[-1])
    print(f'Checkpoint: {ckpt_path}')

    # Data loader
    DS = meta['cls']
    ds = DS(str(DATA_ROOT / args.dataset), meta['feature'],
            phase=meta['phase_test'], toTensor=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    print(f'Test set: {len(ds)} samples')

    results = []
    for cond in ABLATION_CONDITIONS:
        print(f'\nCondition: {cond["name"]}')
        model = load_model_for_condition(
            ckpt_path, args.dataset, cond, device, concept_file)
        AP, mTTA, TTA_R80, P_R80 = evaluate_model(model, loader, meta, device)
        row = {
            'condition': cond['name'],
            'AP':        round(float(AP),     4),
            'mTTA':      round(float(mTTA),   4),
            'TTA_R80':   round(float(TTA_R80),4),
            'P_R80':     round(float(P_R80),  4),
        }
        results.append(row)
        print(f'  AP={AP:.4f}  mTTA={mTTA:.4f}  '
              f'TTA@R80={TTA_R80:.4f}  P@R80={P_R80:.4f}')
        del model
        torch.cuda.empty_cache()

    # Save
    out_dir = Path(args.base_dir) / 'ablation'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.dataset}_ablation.json'
    with open(out_path, 'w') as f:
        json.dump({'dataset': args.dataset,
                   'checkpoint': ckpt_path,
                   'results': results}, f, indent=2)
    print(f'\nSaved: {out_path}')

    # Pretty table
    print(f'\n{"="*72}')
    print(f' Ablation Results: {args.dataset.upper()}')
    print(f'{"="*72}')
    print(f'{"Condition":<25s}  {"AP":6s}  {"mTTA":6s}  {"TTA@R80":8s}  {"P@R80":6s}')
    print('-'*72)
    for row in results:
        print(f"{row['condition']:<25s}  "
              f"{row['AP']:6.4f}  "
              f"{row['mTTA']:6.4f}  "
              f"{row['TTA_R80']:8.4f}  "
              f"{row['P_R80']:6.4f}")
    print('='*72)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base_dir',   default='output/v2_20260314')
    p.add_argument('--checkpoint', default=None,
                   help='Override checkpoint path')
    p.add_argument('--dataset',    required=True,
                   choices=['dad', 'crash', 'a3d'])
    p.add_argument('--gpu',        type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    run_ablation(parse_args())
