#!/usr/bin/env python3
"""eval_missing_ablations.py - Eval 3 old-format checkpoints with legacy=True"""
import os, sys, json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

ROOT       = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT  = CRASH_ROOT / 'data'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru  import LFCRASH_CBM_GRU
from src.data_loader import DADDataset, CrashDataset, A3DDataset
from src.eval_tools  import evaluation

DS_META = {
    'dad':   {'cls': DADDataset,   'feature': 'vgg16', 'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0, 'phase_test': 'testing'},
    'crash': {'cls': CrashDataset, 'feature': 'vgg16', 'x_dim': 4096, 'n_obj': 19, 'n_frames': 50,  'fps': 10.0, 'phase_test': 'test'},
    'a3d':   {'cls': A3DDataset,   'feature': 'vgg16', 'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0, 'phase_test': 'test'},
}

TARGETS = [
    {
        'name': 'a3d_full',
        'ckpt': 'output/phase2_ablation/a3d_full/best_model.pth',
        'dataset': 'a3d', 'gpu': 5,
        'args': {'h_dim':256,'z_dim':512,'lambda_align':1e-4,'lambda_sparse':1e-3,
                 'lambda_recon':1e-2,'no_cbm':False,'no_align':False,
                 'no_sparse':False,'no_recon':False,'num_concepts':837},
    },
    {
        'name': 'crash_no_align',
        'ckpt': 'output/phase2_ablation/crash_no_align/best_model.pth',
        'dataset': 'crash', 'gpu': 5,
        'args': {'h_dim':256,'z_dim':512,'lambda_align':0.0,'lambda_sparse':1e-3,
                 'lambda_recon':1e-2,'no_cbm':False,'no_align':True,
                 'no_sparse':False,'no_recon':False,'num_concepts':837},
    },
    {
        'name': 'dad_no_recon',
        'ckpt': 'output/phase2_ablation/dad_no_recon/best_model.pth',
        'dataset': 'dad', 'gpu': 5,
        'args': {'h_dim':256,'z_dim':128,'lambda_align':2.4e-5,'lambda_sparse':2.6e-4,
                 'lambda_recon':0.0,'no_cbm':False,'no_align':False,
                 'no_sparse':False,'no_recon':True,'num_concepts':837},
    },
]


def collate_fn(batch):
    xs, ys, toas = zip(*batch)
    xs = torch.from_numpy(np.stack(xs, axis=0)).float()
    ys = torch.from_numpy(np.stack(ys, axis=0)).float()
    toa_flat = []
    for t in toas:
        if isinstance(t, (list, tuple, np.ndarray)):
            toa_flat.append(float(t[0]) if hasattr(t, '__len__') and len(t) > 0 else float(t))
        elif isinstance(t, torch.Tensor):
            toa_flat.append(t.item() if t.numel() == 1 else t[0].item())
        else:
            toa_flat.append(float(t))
    return xs, ys, torch.tensor(toa_flat, dtype=torch.float32)


@torch.no_grad()
def run_eval(target):
    name    = target['name']
    ckpt_p  = ROOT / target['ckpt']
    dataset = target['dataset']
    a       = target['args']
    gpu     = target['gpu']

    print(f'\n{"="*60}\n  Evaluating: {name} ({dataset})\n{"="*60}', flush=True)

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    meta   = DS_META[dataset]

    ckpt  = torch.load(str(ckpt_p), map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    epoch = ckpt.get('epoch', '?')
    print(f'  epoch={epoch}', flush=True)

    model = LFCRASH_CBM_GRU(
        x_dim=meta['x_dim'], h_dim=a['h_dim'], z_dim=a['z_dim'],
        n_layers=2, n_obj=meta['n_obj'], n_frames=meta['n_frames'], fps=meta['fps'],
        with_saa=True, num_concepts=a['num_concepts'], concept_file=None,
        lambda_align=a['lambda_align'], lambda_sparse=a['lambda_sparse'],
        lambda_recon=a['lambda_recon'], use_cbm=not a['no_cbm'],
        device=str(device), legacy=True,
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f'  WARN missing   ({len(missing)}): {missing[:3]}', flush=True)
    if unexpected: print(f'  WARN unexpect  ({len(unexpected)}): {unexpected[:3]}', flush=True)
    model.eval()

    DS_cls = meta['cls']
    te_ds  = DS_cls(str(DATA_ROOT / dataset), meta['feature'], phase=meta['phase_test'], toTensor=False)
    loader = DataLoader(te_ds, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(f'  test samples: {len(te_ds)}', flush=True)

    all_pred, all_labels, all_toas = [], [], []
    for xs, ys, toas in loader:
        xs = xs.to(device)
        _, outputs, _ = model(xs, None, None)
        T = len(outputs)
        fp = np.zeros((xs.size(0), meta['n_frames']), dtype=np.float32)
        for t_i, out_t in enumerate(outputs):
            fp[:, t_i] = torch.softmax(out_t, dim=-1)[:, 1].cpu().numpy()
        if T < meta['n_frames']:
            fp[:, T:] = fp[:, T-1:T]
        all_pred.append(fp)
        all_labels.append(ys[:, 1].numpy())
        all_toas.append(toas.numpy())

    all_pred   = np.concatenate(all_pred,   axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_toas   = np.concatenate(all_toas,   axis=0)

    AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toas, fps=meta['fps'])
    print(f'  AP={AP:.4f}  mTTA={mTTA:.4f}  TTA_R80={TTA_R80:.4f}  P_R80={P_R80:.4f}', flush=True)

    res = {'AP': AP, 'mTTA': mTTA, 'TTA_R80': TTA_R80, 'P_R80': P_R80,
           'epoch': epoch, 'dataset': dataset, 'name': name}
    out_path = ROOT / 'output' / 'phase2_ablation' / name / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'  Saved -> {out_path}', flush=True)
    return res


def main():
    os.chdir(ROOT)
    results = []
    for t in TARGETS:
        try:
            results.append(run_eval(t))
        except Exception as e:
            import traceback
            print(f'[ERROR] {t["name"]}: {e}', flush=True)
            traceback.print_exc()

    print(f'\n{"="*60}\nFINAL SUMMARY\n{"="*60}')
    print(f'  {"Name":<22} {"AP":>7} {"mTTA":>7} {"TTA_R80":>9} {"P_R80":>7}')
    for r in results:
        print(f'  {r["name"]:<22} {r["AP"]:>7.4f} {r["mTTA"]:>7.4f} {r["TTA_R80"]:>9.4f} {r["P_R80"]:>7.4f}')


if __name__ == '__main__':
    main()
