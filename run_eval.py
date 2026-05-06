#!/usr/bin/env python3
"""
run_eval.py
===========
独立评估脚本：加载已训练的 best_model.pt，输出完整评估指标。

Usage:
  python3 run_eval.py --checkpoint output/sota_push/crash_sota/best_model.pt \
                      --dataset crash --gpu 0
  python3 run_eval.py --checkpoint output/dad_sota_push/dad_lowlambda/best_model.pt \
                      --dataset dad --gpu 0
  # 批量评估多个 checkpoint：
  python3 run_eval.py --batch
"""
import os, sys, json, argparse
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
    'dad': {
        'cls': DADDataset, 'feature': 'vgg16',
        'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0,
        'phase_test': 'testing',
    },
    'crash': {
        'cls': CrashDataset, 'feature': 'vgg16',
        'x_dim': 4096, 'n_obj': 19, 'n_frames': 50,  'fps': 10.0,
        'phase_test': 'test',
    },
    'a3d': {
        'cls': A3DDataset, 'feature': 'vgg16',
        'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0,
        'phase_test': 'test',
    },
}

# 批量评估配置：(标签, checkpoint路径, dataset, override_args)
# override_args 优先于 checkpoint 内部保存的 args
BATCH_TARGETS = [
    # crash — h_dim=256, z_dim=512
    ('crash_sota',          'output/sota_push/crash_sota/best_model.pt',
     'crash', {'h_dim':256,'z_dim':512,'lambda_align':5e-6,'lambda_sparse':5e-5,'lambda_recon':5e-4}),
    ('crash_frac75',        'output/data_efficiency/crash_frac75/best_model.pt',
     'crash', {'h_dim':256,'z_dim':512,'lambda_align':1e-4,'lambda_sparse':1e-3,'lambda_recon':1e-3}),
    ('crash_full_ablation', 'output/phase2_ablation/crash_full/best_model.pth',
     'crash', {'h_dim':256,'z_dim':512,'lambda_align':1e-4,'lambda_sparse':1e-3,'lambda_recon':1e-2}),
    # dad — h_dim=256, z_dim=128
    ('dad_lowlambda',       'output/dad_sota_push/dad_lowlambda/best_model.pt',
     'dad',   {'h_dim':256,'z_dim':128,'lambda_align':1e-6,'lambda_sparse':5e-5,'lambda_recon':1e-4}),
    ('dad_full_ablation',   'output/phase2_ablation/dad_full/best_model.pth',
     'dad',   {'h_dim':256,'z_dim':128,'lambda_align':2.4e-5,'lambda_sparse':2.6e-4,'lambda_recon':1e-2}),
    # a3d — run_20260314: phi_x.0=[768,4096] => [2*h_dim,x_dim] => h_dim=384, z_dim=128
    ('a3d_run314',          'output/run_20260314_151328/a3d_20260314_153950/best_model.pth',
     'a3d',  {'h_dim':384,'z_dim':128,'lambda_align':6.6e-4,'lambda_sparse':4.8e-3,'lambda_recon':0.0,
              'x_dim_override':4096}),
    ('a3d_full_ablation',   'output/phase2_ablation/a3d_full/best_model.pth',
     'a3d',  {'h_dim':256,'z_dim':512,'lambda_align':1e-4,'lambda_sparse':1e-3,'lambda_recon':1e-2}),
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
def eval_checkpoint(ckpt_path, dataset, gpu=0, batch_size=32, num_workers=0, override_args=None):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return None, f"not found: {ckpt_path}"

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    meta   = DS_META[dataset]

    # Load checkpoint
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Reconstruct model args from checkpoint
    if 'args' in ckpt:
        saved_args = ckpt['args']
        if isinstance(saved_args, argparse.Namespace):
            saved_args = vars(saved_args)
    elif 'model_args' in ckpt:
        saved_args = ckpt['model_args']
    else:
        saved_args = {}

    # override_args takes highest priority
    if override_args:
        saved_args = {**saved_args, **override_args}

    h_dim        = saved_args.get('h_dim', 256)
    z_dim        = saved_args.get('z_dim', 512)
    num_concepts = saved_args.get('num_concepts', 837)
    no_cbm       = saved_args.get('no_cbm', False)
    no_align     = saved_args.get('no_align', False)
    no_sparse    = saved_args.get('no_sparse', False)
    no_recon     = saved_args.get('no_recon', False)
    lambda_align = 0.0 if no_align else saved_args.get('lambda_align', 1e-4)
    lambda_sparse= 0.0 if no_sparse else saved_args.get('lambda_sparse', 1e-3)
    lambda_recon = 0.0 if no_recon  else saved_args.get('lambda_recon',  1e-2)

    concept_file = str(ROOT.parent / '000_all_concept_set.txt')
    if not os.path.exists(concept_file):
        concept_file = None

    model = LFCRASH_CBM_GRU(
        x_dim=meta['x_dim'], h_dim=h_dim, z_dim=z_dim,
        n_layers=2, n_obj=meta['n_obj'], n_frames=meta['n_frames'], fps=meta['fps'],
        with_saa=True, num_concepts=num_concepts, concept_file=concept_file,
        lambda_align=lambda_align, lambda_sparse=lambda_sparse, lambda_recon=lambda_recon,
        use_cbm=not no_cbm, device=str(device),
    ).to(device)

    # Load weights
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    # Detect old checkpoint format (no cgta/crs layers → gru_in was 3*h_dim)
    has_cgta = any('cgta' in k for k in state.keys())
    if not has_cgta:
        # Rebuild model with legacy=True so gru_in = 3*h_dim
        model = LFCRASH_CBM_GRU(
            x_dim=meta['x_dim'], h_dim=h_dim, z_dim=z_dim,
            n_layers=2, n_obj=meta['n_obj'], n_frames=meta['n_frames'], fps=meta['fps'],
            with_saa=True, num_concepts=num_concepts, concept_file=None,
            lambda_align=lambda_align, lambda_sparse=lambda_sparse, lambda_recon=lambda_recon,
            use_cbm=not no_cbm, device=str(device), legacy=True,
        ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()

    # Build test loader
    DS      = meta['cls']
    te      = DS(str(DATA_ROOT / dataset), meta['feature'], phase=meta['phase_test'], toTensor=False)
    loader  = DataLoader(te, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_fn)

    all_pred, all_labels, all_toas = [], [], []
    for xs, ys, toas in loader:
        xs = xs.to(device)
        try:
            _, outputs, _ = model(xs, None, None)
            T = len(outputs)
            frame_probs = np.zeros((xs.size(0), meta['n_frames']), dtype=np.float32)
            for t, out_t in enumerate(outputs):
                p = torch.softmax(out_t, dim=-1)[:, 1].cpu().numpy()
                frame_probs[:, t] = p
            if T < meta['n_frames']:
                frame_probs[:, T:] = frame_probs[:, T-1:T]
            all_pred.append(frame_probs)
            all_labels.append(ys[:, 1].numpy())
            all_toas.append(toas.numpy())
        except Exception as e:
            return None, f"eval error: {e}"

    all_pred   = np.concatenate(all_pred,   axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_toas   = np.concatenate(all_toas,   axis=0)

    AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toas, fps=meta['fps'])
    return {'AP': AP, 'mTTA': mTTA, 'TTA_R80': TTA_R80, 'P_R80': P_R80,
            'checkpoint': str(ckpt_path), 'dataset': dataset,
            'epoch': ckpt.get('epoch', '?')}, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset',    type=str, default=None, choices=['dad','crash','a3d'])
    parser.add_argument('--gpu',        type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch',      action='store_true', help='Run all BATCH_TARGETS')
    args = parser.parse_args()

    os.chdir(ROOT)

    if args.batch:
        print(f"{'Label':<28} {'Dataset':<7} {'AP':>7} {'mTTA':>7} {'TTA_R80':>9} {'P_R80':>7} {'Ep':>5}")
        print('=' * 80)
        summary = []
        for label, ckpt, ds, ov_args in BATCH_TARGETS:
            res, err = eval_checkpoint(ckpt, ds, gpu=args.gpu, batch_size=args.batch_size, override_args=ov_args)
            if err:
                print(f"{label:<28} {ds:<7}  {err}")
            else:
                print(f"{label:<28} {ds:<7} {res['AP']:>7.4f} {res['mTTA']:>7.4f} "
                      f"{res['TTA_R80']:>9.4f} {res['P_R80']:>7.4f} {str(res['epoch']):>5}")
                summary.append({'label': label, **res})
        print('=' * 80)
        out_json = ROOT / 'output' / 'main_results_summary.json'
        out_json.parent.mkdir(exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {out_json}")
    else:
        if not args.checkpoint or not args.dataset:
            parser.error('--checkpoint and --dataset are required (or use --batch)')
        res, err = eval_checkpoint(args.checkpoint, args.dataset, gpu=args.gpu, batch_size=args.batch_size)
        if err:
            print(f"Error: {err}")
        else:
            print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
