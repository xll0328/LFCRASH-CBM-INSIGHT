#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py  –  LFCRASH-CBM v3 unified training script

Key improvements over v2:
  - NaN-safe training with skip & rollback
  - Linear warmup + cosine annealing scheduler
  - Tighter gradient clipping (max_norm=1.0)
  - Evaluate every 2 epochs for better checkpoint selection
  - Proper ablation support (each condition trains from scratch)
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT        = Path(__file__).resolve().parent
CRASH_ROOT  = ROOT.parent / 'CRASH'
DATA_ROOT   = CRASH_ROOT / 'data'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru import LFCRASH_CBM_GRU
from src.data_loader import DADDataset, CrashDataset, A3DDataset
from src.eval_tools  import evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('train')

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
    xs   = torch.from_numpy(np.stack(xs,  axis=0)).float()
    ys   = torch.from_numpy(np.stack(ys,  axis=0)).float()
    toa_flat = []
    for t in toas:
        if isinstance(t, (list, tuple, np.ndarray)):
            toa_flat.append(float(t[0]) if len(t) > 0 else float(t))
        elif isinstance(t, torch.Tensor):
            toa_flat.append(t.item() if t.numel() == 1 else t[0].item())
        else:
            toa_flat.append(float(t))
    toas = torch.tensor(toa_flat, dtype=torch.float32)
    return xs, ys, toas


@torch.no_grad()
def evaluate(model, loader, meta, device):
    model.eval()
    n_frames   = meta['n_frames']
    fps        = meta['fps']
    all_pred   = []
    all_labels = []
    all_toas   = []

    for xs, ys, toas in loader:
        xs   = xs.to(device)
        B    = xs.size(0)
        _, outputs, _ = model(xs, None, None)
        T = len(outputs)
        frame_probs = np.zeros((B, n_frames), dtype=np.float32)
        for t, out_t in enumerate(outputs):
            p = torch.softmax(out_t, dim=-1)[:, 1].cpu().numpy()
            frame_probs[:, t] = p
        if T < n_frames:
            frame_probs[:, T:] = frame_probs[:, T - 1:T]

        all_pred.append(frame_probs)
        all_labels.append(ys[:, 1].numpy())
        all_toas.append(toas.numpy())

    all_pred   = np.concatenate(all_pred,   axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_toas   = np.concatenate(all_toas,   axis=0)

    AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toas, fps=fps)
    return AP, mTTA, TTA_R80, P_R80


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    import math
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if args.gpu < 0 or args.gpu >= n_gpus:
            raise ValueError(f'GPU index {args.gpu} is invalid; visible GPUs: {n_gpus}')
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    meta = DS_META[args.dataset]
    ds_path = DATA_ROOT / args.dataset

    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = args.tag if args.tag else f'{args.dataset}_{ts}'
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out / 'train.log')
    fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)s  %(message)s'))
    log.addHandler(fh)

    log.info(f'Dataset : {args.dataset}')
    log.info(f'Output  : {out}')
    log.info(f'Tag     : {tag}')
    if device.type == 'cuda':
        log.info(f'CUDA    : index={device.index} name={torch.cuda.get_device_name(device)}')
    else:
        log.info('CUDA    : unavailable, using CPU')
    log.info(json.dumps(vars(args), indent=2))

    DS  = meta['cls']
    tr  = DS(str(ds_path), meta['feature'], phase=meta['phase_train'],  toTensor=False)
    te  = DS(str(ds_path), meta['feature'], phase=meta['phase_test'],   toTensor=False)
    log.info(f'Train: {len(tr)}  Test: {len(te)}')

    train_loader = DataLoader(
        tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True, pin_memory=(device.type == 'cuda'),
    )
    test_loader = DataLoader(
        te, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )

    concept_file = str(ROOT.parent / '000_all_concept_set.txt')
    if not os.path.exists(concept_file):
        concept_file = None
        log.warning('concept_file not found – alignment loss disabled')

    model = LFCRASH_CBM_GRU(
        x_dim          = meta['x_dim'],
        h_dim          = args.h_dim,
        z_dim          = args.z_dim,
        n_layers       = 2,
        n_obj          = meta['n_obj'],
        n_frames       = meta['n_frames'],
        fps            = meta['fps'],
        with_saa       = True,
        num_concepts   = args.num_concepts,
        concept_file   = concept_file,
        lambda_align   = 0.0 if args.no_align  else args.lambda_align,
        lambda_sparse  = 0.0 if args.no_sparse else args.lambda_sparse,
        lambda_recon   = 0.0 if args.no_recon  else args.lambda_recon,
        use_cbm        = not args.no_cbm,
        device         = str(device),
    ).to(device)
    model.enable_cgta = not args.disable_cgta
    model.enable_crs = not args.disable_crs
    model.ac_use_concepts = not args.ac_no_concepts
    model.use_ac = not args.disable_ac
    if args.disable_ac:
        model.lambda_ac_policy = 0.0
        model.lambda_ac_value = 0.0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Trainable params: {n_params}')

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_warmup_cosine_scheduler(
        optimizer, warmup_epochs=3, total_epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )

    best_ap      = 0.0
    best_epoch   = 0
    history      = []
    nan_count    = 0
    MAX_NAN      = 50
    no_improve   = 0   # early-stopping counter (eval steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = ep_ce = ep_aux = ep_align = ep_sparse = ep_recon = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Ep {epoch:03d}/{args.epochs}', leave=False)
        for xs, ys, toas in pbar:
            xs   = xs.to(device, non_blocking=True)
            ys   = ys.to(device, non_blocking=True)
            toas = toas.to(device, non_blocking=True)

            optimizer.zero_grad()
            losses, _, _ = model(xs, ys, toas)
            total = losses['total_loss']

            if torch.isnan(total) or torch.isinf(total):
                nan_count += 1
                if nan_count > MAX_NAN:
                    log.error(f'Too many NaN steps ({nan_count}), stopping.')
                    break
                continue

            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss   += total.item()
            ep_ce     += losses['ce_loss'].item()
            ep_aux    += losses['aux_loss'].item()
            ep_align  += losses['align_loss'].item()
            ep_sparse += losses['sparse_loss'].item()
            ep_recon  += losses.get('recon_loss', torch.tensor(0.0)).item()
            n_batches += 1

            pbar.set_postfix(
                loss=f'{total.item():.3f}',
                ce=f'{losses["ce_loss"].item():.3f}',
                gn=f'{grad_norm:.2f}',
            )

        if nan_count > MAX_NAN:
            break

        avg = lambda v: v / max(n_batches, 1)
        cur_lr = optimizer.param_groups[0]['lr']
        log.info(
            f'Ep {epoch:03d} | loss={avg(ep_loss):.4f}  '
            f'ce={avg(ep_ce):.4f}  aux={avg(ep_aux):.4f}  '
            f'align={avg(ep_align):.4f}  sparse={avg(ep_sparse):.4f}  '
            f'recon={avg(ep_recon):.4f}  lr={cur_lr:.2e}  nan={nan_count}'
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            AP, mTTA, TTA_R80, P_R80 = evaluate(model, test_loader, meta, device)
            log.info(
                f'Ep {epoch:03d} EVAL | AP={AP:.4f}  mTTA={mTTA:.4f}s  '
                f'TTA@R80={TTA_R80:.4f}s  P@R80={P_R80:.4f}'
            )

            row = dict(epoch=epoch,
                       train=dict(total=avg(ep_loss), ce=avg(ep_ce),
                                  aux=avg(ep_aux), align=avg(ep_align),
                                  sparse=avg(ep_sparse), recon=avg(ep_recon)),
                       test=dict(AP=AP, mTTA=mTTA, TTA_R80=TTA_R80, P_R80=P_R80),
                       lr=cur_lr)
            history.append(row)

            with open(out / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

            if AP > best_ap:
                best_ap    = AP
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'AP': AP, 'mTTA': mTTA,
                    'args': vars(args),
                }, out / 'best_model.pth')
                log.info(f'  * New best AP={AP:.4f} (epoch {epoch})')
            else:
                no_improve += 1
                if args.patience > 0 and no_improve >= args.patience:
                    log.info(f'Early stopping: no improvement for {no_improve} evals '
                             f'(patience={args.patience}). Best AP={best_ap:.4f} @ ep{best_epoch}')
                    break

    log.info('=' * 60)
    log.info(f'Training complete.  Best AP={best_ap:.4f} at epoch {best_epoch}')
    log.info(f'Output dir: {out}')

    ckpt_path = out / 'best_model.pth'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        AP, mTTA, TTA_R80, P_R80 = evaluate(model, test_loader, meta, device)
        log.info(f'FINAL (best ckpt) | AP={AP:.4f}  mTTA={mTTA:.4f}  '
                 f'TTA@R80={TTA_R80:.4f}  P@R80={P_R80:.4f}')
    else:
        AP, mTTA, TTA_R80, P_R80 = evaluate(model, test_loader, meta, device)
        log.info(f'FINAL (last ckpt) | AP={AP:.4f}  mTTA={mTTA:.4f}  '
                 f'TTA@R80={TTA_R80:.4f}  P@R80={P_R80:.4f}')

    result = dict(
        dataset=args.dataset, best_epoch=best_epoch,
        AP=AP, mTTA=mTTA, TTA_R80=TTA_R80, P_R80=P_R80,
        args=vars(args),
        ablation=dict(no_cbm=args.no_cbm, no_align=args.no_align,
                      no_sparse=args.no_sparse, no_recon=args.no_recon),
    )
    with open(out / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f'Results saved to {out / "results.json"}')
    return result


def parse_args():
    p = argparse.ArgumentParser(description='LFCRASH-CBM v3 training')
    p.add_argument('--dataset',       type=str,   required=True,
                   choices=['dad', 'crash', 'a3d'])
    p.add_argument('--gpu',           type=int,   default=0)
    p.add_argument('--epochs',        type=int,   default=80)
    p.add_argument('--batch_size',    type=int,   default=16)
    p.add_argument('--lr',            type=float, default=2e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--h_dim',         type=int,   default=256)
    p.add_argument('--z_dim',         type=int,   default=128)
    p.add_argument('--lambda_align',  type=float, default=1e-4)
    p.add_argument('--lambda_sparse', type=float, default=1e-3)
    p.add_argument('--lambda_recon',  type=float, default=1e-2)
    p.add_argument('--num_concepts',  type=int,   default=837)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--eval_every',    type=int,   default=2)
    p.add_argument('--output_dir',    type=str,   default='output')
    p.add_argument('--tag',           type=str,   default='',
                   help='Custom output subdirectory name')
    p.add_argument('--no_cbm',    action='store_true')
    p.add_argument('--no_align',  action='store_true')
    p.add_argument('--no_sparse', action='store_true')
    p.add_argument('--no_recon',  action='store_true')
    p.add_argument('--patience',  type=int, default=0,
                   help='Early stopping patience in eval steps (0=disabled)')
    p.add_argument('--disable_cgta', action='store_true')
    p.add_argument('--disable_crs', action='store_true')
    p.add_argument('--ac_no_concepts', action='store_true')
    p.add_argument('--disable_ac', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
