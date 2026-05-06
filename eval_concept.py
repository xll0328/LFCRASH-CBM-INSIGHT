#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_concept.py
===============
Concept quality evaluation for CG-CRASH.
Generates:
  1. Top-K concept activation table per class (positive / negative)
  2. Per-concept discriminability ranking
  3. Temporal concept activation curves
  4. Bar chart + timeline plots
  5. JSON summary saved to <output_dir>/

Usage
-----
python eval_concept.py \\
    --checkpoint output/v2_20260314/dad_20260314_162432/best_model.pth \\
    --dataset dad \\
    --output_dir output/v2_20260314/concept_eval

# Quick sanity check with 50 batches:
python eval_concept.py --checkpoint ... --dataset dad --max_batches 50
"""
import os, sys, argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent          # LFCRASH-CBM/
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT  = CRASH_ROOT / 'data'
DEFAULT_CONCEPT_FILE = str(ROOT.parent / '000_all_concept_set.txt')

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru  import LFCRASH_CBM_GRU
from src.data_loader import DADDataset, CrashDataset, A3DDataset
from src.concept_utils import (
    load_concept_names, save_concept_report,
    plot_concept_timeline, plot_top_concepts_bar, concept_importance_heatmap,
    compute_discriminability,
)

import numpy as np
from torch.utils.data import DataLoader


# ── Dataset registry (mirrors train.py) ───────────────────────────────────────
DS_META = {
    'dad': {
        'cls': DADDataset, 'feature': 'vgg16',
        'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0,
        'phase_train': 'training', 'phase_test': 'testing',
    },
    'crash': {
        'cls': CrashDataset, 'feature': 'vgg16',
        'x_dim': 4096, 'n_obj': 19, 'n_frames': 50, 'fps': 10.0,
        'phase_train': 'train', 'phase_test': 'test',
    },
    'a3d': {
        'cls': A3DDataset, 'feature': 'vgg16',
        'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0,
        'phase_train': 'train', 'phase_test': 'test',
    },
}


def collate_fn(batch):
    xs, ys, toas = zip(*batch)
    xs   = torch.from_numpy(np.stack(xs)).float()
    ys   = torch.from_numpy(np.stack(ys)).float()
    toa_flat = []
    for t in toas:
        if isinstance(t, (list, tuple, np.ndarray)):
            toa_flat.append(float(t[0]) if hasattr(t, '__len__') else float(t))
        elif isinstance(t, torch.Tensor):
            toa_flat.append(t.item())
        else:
            toa_flat.append(float(t))
    return xs, ys, torch.tensor(toa_flat, dtype=torch.float32)


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, dataset: str, concept_file: str,
               device: torch.device) -> LFCRASH_CBM_GRU:
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Support both checkpoint formats:
    # Format A (train.py v2): keys = epoch, state_dict, optimizer, AP, args
    # Format B (older):       keys = model_state_dict, config
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        saved_args  = ckpt.get('args', {})
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        saved_args  = ckpt.get('config', {})
    else:
        # bare state dict
        state_dict = ckpt
        saved_args  = {}

    meta = DS_META[dataset]
    model = LFCRASH_CBM_GRU(
        x_dim         = saved_args.get('x_dim',        meta['x_dim']),
        h_dim         = saved_args.get('h_dim',        256),
        z_dim         = saved_args.get('z_dim',        128),
        n_layers      = saved_args.get('n_layers',     2),
        n_obj         = saved_args.get('n_obj',        meta['n_obj']),
        n_frames      = saved_args.get('n_frames',     meta['n_frames']),
        fps           = saved_args.get('fps',          meta['fps']),
        with_saa      = True,
        num_concepts  = saved_args.get('num_concepts', 837),
        concept_file  = concept_file if os.path.exists(concept_file) else None,
        lambda_align  = 0.0,
        lambda_sparse = 0.0,
        lambda_recon  = 0.0,
        use_cbm       = True,
        device        = str(device),
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[eval_concept] Missing keys ({len(missing)}): {missing[:5]}...')
    if unexpected:
        print(f'[eval_concept] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...')

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[eval_concept] Model loaded. Params={n_params:}  '
          f'AP_saved={ckpt.get("AP", "?")}')
    return model


# ── Activation extraction ─────────────────────────────────────────────────────
@torch.no_grad()
def extract_activations(model, loader, device, max_batches=None):
    """
    Returns:
      acts   : np.ndarray (N, T, C)
      labels : np.ndarray (N,)   1=accident, 0=normal
      toas   : np.ndarray (N,)
    """
    model.eval()
    acts_l, lab_l, toa_l = [], [], []
    for i, (xs, ys, toas) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        xs   = xs.to(device)
        acts = model.get_concept_activations(xs)   # (B, T, C)
        acts_l.append(acts.cpu().numpy())
        lab_l.append(ys[:, 1].numpy())
        toa_l.append(toas.numpy())
        if (i + 1) % 20 == 0:
            print(f'  [{i+1} batches done]', flush=True)
    return (
        np.concatenate(acts_l, axis=0),
        np.concatenate(lab_l,  axis=0),
        np.concatenate(toa_l,  axis=0),
    )


# ── Pre-accident temporal focus ───────────────────────────────────────────────
def pre_accident_curves(
    acts: np.ndarray,    # (N, T, C)
    labels: np.ndarray,
    toas: np.ndarray,
    concept_names,
    fps: float,
    pre_seconds: float = 3.0,
    top_k: int = 8,
):
    """
    For positive samples, align concept curves to time-of-accident and
    average over the last `pre_seconds` before crash.
    Returns dict: concept_name -> aligned_curve (T,)
    """
    disc = compute_discriminability(acts, labels, concept_names, top_k=top_k)
    if not disc:
        return {}
    top_idx = [d['idx'] for d in disc]

    pre_frames = int(pre_seconds * fps)
    pos_mask   = labels == 1
    pos_acts   = acts[pos_mask]    # (Np, T, C)
    pos_toas   = toas[pos_mask]
    T          = acts.shape[1]

    aligned = {i: [] for i in top_idx}
    for sample_acts, toa in zip(pos_acts, pos_toas):
        toa_f = int(min(toa, T - 1))
        start = max(0, toa_f - pre_frames)
        segment = sample_acts[start:toa_f + 1]   # (<=pre_frames+1, C)
        for i in top_idx:
            aligned[i].append(segment[:, i])

    curves = {}
    for i in top_idx:
        name = concept_names[i] if concept_names else f'concept_{i}'
        segs = aligned[i]
        if not segs:
            continue
        # Pad/truncate to pre_frames+1 and average
        padded = np.zeros((len(segs), pre_frames + 1))
        for j, seg in enumerate(segs):
            l = min(len(seg), pre_frames + 1)
            padded[j, -l:] = seg[-l:]
        curves[name] = padded.mean(0).tolist()
    return curves


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='CG-CRASH concept evaluation')
    p.add_argument('--checkpoint',   required=True,
                   help='Path to best_model.pth')
    p.add_argument('--dataset',      required=True,
                   choices=['dad', 'crash', 'a3d'])
    p.add_argument('--concept_file', default=DEFAULT_CONCEPT_FILE,
                   help='Path to concept text file (one per line)')
    p.add_argument('--topk',         type=int,  default=20)
    p.add_argument('--batch_size',   type=int,  default=4)
    p.add_argument('--max_batches',  type=int,  default=None,
                   help='Limit batches for quick eval (None=all)')
    p.add_argument('--split',        default='test',
                   choices=['train', 'test'],
                   help='Which split to evaluate on')
    p.add_argument('--output_dir',   default=None,
                   help='Output dir (default: checkpoint_dir/concept_eval)')
    p.add_argument('--gpu',          type=int,  default=0)
    p.add_argument('--pre_seconds',  type=float, default=3.0,
                   help='Pre-accident window for aligned curves (seconds)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # ── Output dir ────────────────────────────────────────────────────────────
    if args.output_dir is None:
        args.output_dir = str(
            Path(args.checkpoint).parent / 'concept_eval')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[eval_concept] Output dir: {args.output_dir}')

    # ── Concept names ─────────────────────────────────────────────────────────
    concept_names = None
    if os.path.exists(args.concept_file):
        concept_names = load_concept_names(args.concept_file)
        print(f'[eval_concept] {len(concept_names)} concept names loaded')
    else:
        print(f'[eval_concept] WARNING: concept file not found: {args.concept_file}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, args.dataset,
                       args.concept_file, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    meta    = DS_META[args.dataset]
    ds_path = DATA_ROOT / args.dataset
    DS      = meta['cls']
    phase   = meta[f'phase_{args.split}']
    dataset = DS(str(ds_path), meta['feature'], phase=phase, toTensor=False)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    print(f'[eval_concept] {args.dataset}/{args.split}: {len(dataset)} samples')

    # ── Extract ───────────────────────────────────────────────────────────────
    print('Extracting concept activations...')
    acts, labels, toas = extract_activations(
        model, loader, device, max_batches=args.max_batches)
    print(f'  acts={acts.shape}  pos={int((labels==1).sum())}  '
          f'neg={int((labels==0).sum())}')

    # ── Full report ───────────────────────────────────────────────────────────
    print('Generating report...')
    summary = save_concept_report(
        acts, labels, concept_names,
        output_dir=args.output_dir,
        dataset=args.dataset,
        fps=meta['fps'],
        top_k=args.topk,
    )

    # ── Pre-accident aligned curves ───────────────────────────────────────────
    print(f'Computing pre-accident curves ({args.pre_seconds}s window)...')
    aligned = pre_accident_curves(
        acts, labels, toas, concept_names,
        fps=meta['fps'], pre_seconds=args.pre_seconds, top_k=8,
    )
    aligned_path = os.path.join(args.output_dir,
                                f'{args.dataset}_pre_accident_curves.json')
    with open(aligned_path, 'w') as f:
        json.dump(aligned, f, indent=2, ensure_ascii=False)
    print(f'  Saved: {aligned_path}')

    # ── Single-sample heatmap (first positive sample) ─────────────────────────
    pos_idx = np.where(labels == 1)[0]
    if len(pos_idx) and concept_names:
        sample_acts = acts[pos_idx[0]]   # (T, C)
        heatmap_path = os.path.join(
            args.output_dir, f'{args.dataset}_sample_heatmap.png')
        concept_importance_heatmap(
            sample_acts, concept_names, top_k=15,
            output_path=heatmap_path,
            title=f'[{args.dataset.upper()}] Per-Frame Concept Activations (sample #{pos_idx[0]})',
        )
        print(f'  Saved heatmap: {heatmap_path}')

    # ── Console summary ───────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'DATASET: {args.dataset.upper()}  |  '
          f'Samples={len(labels)}  Pos={int((labels==1).sum())}  '
          f'Neg={int((labels==0).sum())}')
    print(f'{"="*60}')
    print(f'Top-10 DISCRIMINATIVE concepts:')
    for item in summary.get('top_discriminative', [])[:10]:
        print(f"  {item['rank']:2d}. {item['concept'][:50]:<52s}  "
              f"disc={item['discriminability']:.4f}  "
              f"pos={item['mean_pos']:.4f}  neg={item['mean_neg']:.4f}")
    print(f'\nTop-10 POSITIVE concepts (mean activation):')
    for item in summary.get('top_positive', [])[:10]:
        print(f"  {item['rank']:2d}. {item['concept'][:50]:<52s}  "
              f"mean={item['mean_activation']:.4f}")
    print(f'\nOutput saved to: {args.output_dir}')
    print('='*60)


if __name__ == '__main__':
    main()
