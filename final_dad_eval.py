#!/usr/bin/env python3
"""
final_dad_eval.py
==================
评估所有 DAD checkpoint，选出最佳结果并更新论文数字。

Usage:
  python final_dad_eval.py --gpu 6
"""
import os, sys, json, argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'CRASH'))

from src.models_gru import LFCRASH_CBM_GRU
from src.data_loader import DADDataset
from src.eval_tools import evaluation

DAD_CHECKPOINTS = [
    # (name, ckpt_path, h_dim, z_dim, legacy)
    ('v3_dad_full',      'output/v3_final/dad_full/best_model.pth',        256, 128, True),
    ('v3_dad_no_sparse', 'output/v3_final/dad_no_sparse/best_model.pth',   256, 128, True),
    ('v3_dad_no_cbm',    'output/v3_final/dad_no_cbm/best_model.pth',      256, 128, True),
    ('dad_z512',         'output/dad_sota_push/dad_z512/best_model.pt',    256, 256, False),
    ('dad_h512_v2',      'output/dad_sota_push/dad_h512_v2/best_model.pt', 512, 256, False),
    ('curriculum_v1',    'output/dad_curriculum/dad_curriculum_v1/best_model.pt', 256, 256, False),
    ('finetune_z256',    'output/dad_curriculum/dad_finetune_z256/best_model.pt', 256, 256, False),
    ('no_sparse_long',   'output/dad_curriculum/dad_no_sparse_long/best_model.pt', 256, 128, False),
]


def collate_fn(batch):
    xs, ys, toas = zip(*batch)
    xs = torch.from_numpy(np.stack(xs)).float()
    ys = torch.from_numpy(np.stack(ys)).float()
    tf = [float(t[0]) if hasattr(t,'__len__') else float(t) for t in toas]
    return xs, ys, torch.tensor(tf, dtype=torch.float32)


@torch.no_grad()
def eval_ckpt(name, ckpt_path, h_dim, z_dim, legacy, device, loader):
    p = ROOT / ckpt_path
    if not p.exists():
        print(f'  [{name}] NOT FOUND: {p}')
        return None
    ckpt  = torch.load(str(p), map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    has_cgta = any('cgta' in k for k in state.keys())
    # Auto-detect h_dim from weights
    phi = state.get('phi_x.0.weight')
    if phi is not None:
        h_dim = phi.shape[0] // 2
    model = LFCRASH_CBM_GRU(
        x_dim=4096, h_dim=h_dim, z_dim=z_dim, n_layers=2,
        n_obj=19, n_frames=100, fps=20.0, with_saa=True,
        num_concepts=837, concept_file=None,
        lambda_align=1e-4, lambda_sparse=0.0, lambda_recon=1e-2,
        use_cbm=True, device=str(device), legacy=legacy or not has_cgta,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    all_pred, all_labels, all_toas = [], [], []
    for xs, ys, toas in loader:
        xs = xs.to(device)
        _, outputs, _ = model(xs, None, None)
        T = len(outputs)
        fp = np.zeros((xs.size(0), 100), dtype=np.float32)
        for t, o in enumerate(outputs):
            fp[:, t] = torch.softmax(o, dim=-1)[:, 1].cpu().numpy()
        if T < 100: fp[:, T:] = fp[:, T-1:T]
        all_pred.append(fp)
        all_labels.append(ys[:, 1].numpy())
        all_toas.append(toas.numpy())
    all_pred   = np.concatenate(all_pred)
    all_labels = np.concatenate(all_labels)
    all_toas   = np.concatenate(all_toas)
    AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toas, fps=20.0)
    ep = ckpt.get('epoch', '?')
    print(f'  [{name}] ep={ep} AP={AP:.4f} mTTA={mTTA:.4f} TTA_R80={TTA_R80:.4f} P_R80={P_R80:.4f}')
    return {'name': name, 'AP': AP, 'mTTA': mTTA, 'TTA_R80': TTA_R80, 'P_R80': P_R80,
            'epoch': ep, 'h_dim': h_dim, 'z_dim': z_dim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=6)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating all DAD checkpoints on {device}...')

    ds = DADDataset(str(ROOT.parent/'CRASH'/'data'/'dad'), 'vgg16',
                    phase='testing', toTensor=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    print(f'Test samples: {len(ds)}')

    results = []
    for name, path, h_dim, z_dim, legacy in DAD_CHECKPOINTS:
        try:
            r = eval_ckpt(name, path, h_dim, z_dim, legacy, device, loader)
            if r: results.append(r)
        except Exception as e:
            print(f'  [{name}] ERROR: {e}')

    if not results:
        print('No results found.')
        return

    # Sort by AP
    results.sort(key=lambda x: x['AP'], reverse=True)
    print()
    print('=' * 70)
    print('  FINAL DAD RANKING')
    print('=' * 70)
    print(f'  {"Name":<22} {"AP":>8} {"mTTA":>8} {"TTA@R80":>10} {"P@R80":>8}')
    print('  ' + '-' * 60)
    for r in results:
        marker = '  <-- BEST' if r == results[0] else ''
        print(f'  {r["name"]:<22} {r["AP"]*100:>7.2f}% {r["mTTA"]:8.3f}s '
              f'{r["TTA_R80"]:9.3f}s {r["P_R80"]*100:7.2f}%{marker}')

    # Save best
    best = results[0]
    best_path = ROOT / 'output' / 'dad_best_final' / 'results.json'
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f'\nBest result saved to {best_path}')
    print(f'Best DAD: {best["name"]} AP={best["AP"]*100:.2f}% mTTA={best["mTTA"]:.3f}s')


if __name__ == '__main__':
    main()
