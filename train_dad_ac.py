#!/usr/bin/env python3
"""
train_dad_ac.py
===============
CG-CRASH v4: Actor-Critic + Curriculum Training on DAD

新增特性:
  1. Actor-Critic loss (policy + value + entropy) 直接优化 mTTA
  2. Concept-Aware Reward: 概念越早激活奖励越高
  3. 三阶段 curriculum:
     Phase 1 (warmup_epochs):      纯 GRU 预热, AC warmup
     Phase 2 (warmup~cbm_start):   引入 CBM, AC 继续训练
     Phase 3 (cbm_start~end):      CBM + AC 联合优化
  4. 更激进的 lr schedule (OneCycleLR)

Usage:
  python train_dad_ac.py --gpu 0 --tag dad_ac_v1
"""
import os, sys, json, logging, argparse, math
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT  = CRASH_ROOT / 'data'
DEFAULT_CONCEPT_FILE = str(ROOT.parent / '000_all_concept_set.txt')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru  import LFCRASH_CBM_GRU
from src.data_loader import DADDataset
from src.eval_tools  import evaluation


def setup_logging(log_file):
    logger = logging.getLogger('DAD_AC')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def collate_fn(batch):
    xs, ys, toas = zip(*batch)
    xs = torch.from_numpy(np.stack(xs, axis=0)).float()
    ys = torch.from_numpy(np.stack(ys, axis=0)).float()
    toa_flat = []
    for t in toas:
        if isinstance(t, (list, tuple, np.ndarray)):
            toa_flat.append(float(t[0]) if len(t) > 0 else float(t))
        elif isinstance(t, torch.Tensor):
            toa_flat.append(t.item() if t.numel() == 1 else t[0].item())
        else:
            toa_flat.append(float(t))
    return xs, ys, torch.tensor(toa_flat, dtype=torch.float32)


def make_loader(dataset, batch_size, shuffle, num_workers, collate_fn, pin_memory, drop_last=False):
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4
    return DataLoader(dataset, **loader_kwargs)


@torch.no_grad()
def evaluate(model, loader, n_frames, fps, device, logger):
    model.eval()
    all_pred, all_labels, all_toas = [], [], []
    for xs, ys, toas in loader:
        xs = xs.to(device)
        try:
            result = model(xs, None, None)
            if isinstance(result, tuple) and len(result) == 3:
                _, outputs, _ = result
            else:
                outputs = result
            if isinstance(outputs, torch.Tensor):
                if outputs.dim() == 3:
                    outputs = [outputs[:, t] for t in range(outputs.shape[1])]
                else:
                    outputs = [outputs]
            T = len(outputs)
            if T == 0:
                continue
            fp = np.zeros((xs.size(0), n_frames), dtype=np.float32)
            for t, out_t in enumerate(outputs):
                if out_t.dim() == 1:
                    out_t = out_t.unsqueeze(0)
                if out_t.shape[-1] == 2:
                    p = torch.softmax(out_t, dim=-1)[:, 1].cpu().numpy()
                else:
                    p = torch.sigmoid(out_t[:, 0]).cpu().numpy()
                fp[:, t] = p
            if T < n_frames:
                fp[:, T:] = fp[:, T-1:T]
            all_pred.append(fp)
            all_labels.append(ys[:, 1].numpy())
            all_toas.append(toas.numpy())
        except Exception as e:
            import traceback
            logger.error(f'Eval error: {e}\n{traceback.format_exc()}')
            continue
    if not all_pred:
        return 0., 0., 0., 0.
    all_pred   = np.concatenate(all_pred,   axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_toas   = np.concatenate(all_toas,   axis=0)
    return evaluation(all_pred, all_labels, all_toas, fps=fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu',             type=int,   default=0)
    p.add_argument('--tag',             type=str,   default='dad_ac_v1')
    p.add_argument('--epochs',          type=int,   default=150)
    p.add_argument('--warmup_epochs',   type=int,   default=15,
                   help='Epochs with CBM disabled (GRU + AC warmup)')
    p.add_argument('--cbm_ramp_epochs', type=int,   default=20,
                   help='Epochs to linearly ramp in CBM after warmup')
    p.add_argument('--batch_size',      type=int,   default=16)
    p.add_argument('--lr',              type=float, default=3e-4)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
    p.add_argument('--h_dim',           type=int,   default=256)
    p.add_argument('--z_dim',           type=int,   default=256)
    p.add_argument('--lambda_align',    type=float, default=1e-6)
    p.add_argument('--lambda_sparse',   type=float, default=5e-5)
    p.add_argument('--lambda_recon',    type=float, default=1e-4)
    p.add_argument('--lambda_ac_policy',type=float, default=0.5,
                   help='Actor-Critic policy loss weight')
    p.add_argument('--lambda_ac_value', type=float, default=0.5,
                   help='Actor-Critic value loss weight')
    p.add_argument('--ac_gamma',        type=float, default=0.95,
                   help='Actor-Critic discount factor')
    p.add_argument('--ac_entropy',      type=float, default=0.01,
                   help='Entropy regularization coefficient')
    p.add_argument('--num_concepts',    type=int,   default=837)
    p.add_argument('--eval_every',      type=int,   default=5)
    p.add_argument('--num_workers',     type=int,   default=8)
    p.add_argument('--resume',          type=str,   default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--use_rwkv',        action='store_true', default=False,
                   help='Use RWKV temporal module instead of GRU')
    p.add_argument('--disable_cgta',    action='store_true')
    p.add_argument('--disable_crs',     action='store_true')
    p.add_argument('--ac_no_concepts',  action='store_true')
    p.add_argument('--disable_ac',      action='store_true')
    p.add_argument('--concept_file',    type=str,   default=DEFAULT_CONCEPT_FILE)
    args = p.parse_args()

    out_dir = ROOT / 'output' / 'dad_ac' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir / 'train.log'))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'=== CG-CRASH v4: Actor-Critic Training ===')
    logger.info(f'Device: {device} | Tag: {args.tag}')
    logger.info(f'Epochs: {args.epochs} | Warmup: {args.warmup_epochs} | CBM ramp: {args.cbm_ramp_epochs}')
    logger.info(f'lambda_ac_policy={args.lambda_ac_policy} lambda_ac_value={args.lambda_ac_value}')
    logger.info(f'ac_gamma={args.ac_gamma} ac_entropy={args.ac_entropy}')

    # Data
    meta = {'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0}
    tr_ds = DADDataset(str(DATA_ROOT / 'dad'), 'vgg16', phase='training', toTensor=False)
    te_ds = DADDataset(str(DATA_ROOT / 'dad'), 'vgg16', phase='testing',  toTensor=False)
    tr_ld = make_loader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    te_ld = make_loader(
        te_ds,
        batch_size=32,
        shuffle=False,
        num_workers=max(2, min(4, args.num_workers // 2)),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    logger.info(f'Train: {len(tr_ds)} | Test: {len(te_ds)}')

    # Build model
    model = LFCRASH_CBM_GRU(
        x_dim=meta['x_dim'], h_dim=args.h_dim, z_dim=args.z_dim,
        n_layers=2, n_obj=meta['n_obj'], n_frames=meta['n_frames'],
        fps=meta['fps'], with_saa=True,
        num_concepts=args.num_concepts,
        concept_file=args.concept_file if args.concept_file and os.path.exists(args.concept_file) else None,
        lambda_align=args.lambda_align,
        lambda_sparse=args.lambda_sparse,
        lambda_recon=args.lambda_recon,
        use_cbm=True,
        device=str(device),
        legacy=False,
        use_rwkv=args.use_rwkv,
    ).to(device)
    model.enable_cgta = not args.disable_cgta
    model.enable_crs = not args.disable_crs
    model.ac_use_concepts = not args.ac_no_concepts
    model.use_ac = not args.disable_ac

    # Set AC hyperparams
    model.ac_module.gamma        = args.ac_gamma
    model.ac_module.entropy_coef = args.ac_entropy
    model.lambda_ac_policy       = 0.0 if args.disable_ac else args.lambda_ac_policy
    model.lambda_ac_value        = 0.0 if args.disable_ac else args.lambda_ac_value

    start_epoch = 1
    best_ap = 0.0
    best_epoch = 0

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # T_0 starts AFTER warmup so CBM gets full lr from the beginning of CBM phase
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6)
    _scheduler_reset = False  # track if we've reset after warmup

    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info(f'Resumed from {args.resume} (AP={ckpt.get("AP", "?"):.4f})')
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_ap = ckpt.get('AP', 0.0)

    results_path = out_dir / 'results.json'
    loss_log = []

    for epoch in range(start_epoch, args.epochs + 1):
        # ── Curriculum scheduling ──────────────────────────────────────────
        if epoch <= args.warmup_epochs:
            # Phase 1: GRU + AC warmup, CBM off
            cbm_scale = 0.0
            ac_scale  = min(1.0, epoch / max(args.warmup_epochs, 1))
            model.use_cbm = False
            phase_str = f'WARMUP(ac={ac_scale:.2f})'
        else:
            # Phase 2-3: linearly ramp in CBM
            ramp = min(args.cbm_ramp_epochs, args.epochs - args.warmup_epochs)
            cbm_scale = min(1.0, (epoch - args.warmup_epochs) / max(ramp, 1))
            ac_scale  = 1.0
            model.use_cbm = True
            phase_str = f'CBM({cbm_scale:.2f})+AC'

        model.lambda_align  = args.lambda_align  * cbm_scale
        model.lambda_sparse = args.lambda_sparse * cbm_scale
        model.lambda_recon  = args.lambda_recon  * cbm_scale
        model.lambda_ac_policy = args.lambda_ac_policy * ac_scale
        model.lambda_ac_value  = args.lambda_ac_value  * ac_scale

        # Reset lr scheduler at start of CBM phase so CBM gets full lr
        _skip_scheduler_step = False
        if epoch == args.warmup_epochs + 1 and not _scheduler_reset:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
            remaining = max(args.epochs - args.warmup_epochs, 50)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=remaining, T_mult=1, eta_min=1e-6)
            _scheduler_reset = True
            _skip_scheduler_step = True  # don't step this epoch, keep full lr
            logger.info(f'LR reset to {args.lr:.2e} for CBM phase (T_0={remaining})')

        # ── Training loop ─────────────────────────────────────────────────
        model.train()
        epoch_losses = {k: 0.0 for k in
            ['total_loss','ce_loss','aux_loss','ac_policy_loss','ac_value_loss']}
        n_batches = 0

        for xs, ys, toas in tr_ld:
            xs   = xs.to(device)
            ys   = ys.to(device)
            toas = toas.to(device)

            optimizer.zero_grad()
            losses, _, _ = model(xs, ys, toas)
            loss = losses['total_loss']

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f'NaN/Inf loss at epoch {epoch}, skipping batch')
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k in epoch_losses:
                if k in losses:
                    v = losses[k]
                    epoch_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
            n_batches += 1

        if not _skip_scheduler_step:
            scheduler.step()
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        loss_log.append({'epoch': epoch, **avg})

        logger.info(
            f'Ep {epoch:3d}/{args.epochs} [{phase_str}] '
            f'loss={avg["total_loss"]:.4f} '
            f'ce={avg["ce_loss"]:.4f} '
            f'aux={avg["aux_loss"]:.4f} '
            f'pol={avg["ac_policy_loss"]:.4f} '
            f'val={avg["ac_value_loss"]:.4f} '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}'
        )

        # ── Evaluation ────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            AP, mTTA, TTA_R80, P_R80 = evaluate(
                model, te_ld, meta['n_frames'], meta['fps'], device, logger)
            logger.info(
                f'  EVAL | AP={AP:.4f} mTTA={mTTA:.4f} '
                f'TTA_R80={TTA_R80:.4f} P_R80={P_R80:.4f}'
            )

            if AP > best_ap:
                best_ap    = AP
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'AP': AP, 'mTTA': mTTA,
                    'TTA_R80': TTA_R80, 'P_R80': P_R80,
                    'args': vars(args),
                }, out_dir / 'best_model.pt')
                with open(results_path, 'w') as f:
                    json.dump({'AP': AP, 'mTTA': mTTA, 'TTA_R80': TTA_R80,
                               'P_R80': P_R80, 'epoch': epoch,
                               'dataset': 'dad', 'tag': args.tag}, f, indent=2)
                logger.info(f'  *** New best AP={AP:.4f} at epoch {epoch} ***')

    # Save loss log
    with open(out_dir / 'loss_log.json', 'w') as f:
        json.dump(loss_log, f, indent=2)

    logger.info(f'\n=== DONE === Best AP={best_ap:.4f} at epoch {best_epoch}')
    logger.info(f'Output: {out_dir}')


if __name__ == '__main__':
    main()