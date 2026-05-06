#!/usr/bin/env python3
"""
train_dad_curriculum.py
=======================
DAD 攻坚：Concept Curriculum Training
策略：
  1. 前 warmup_epochs epoch：use_cbm=False，纯 GRU 预热
  2. 之后：逐渐引入 CBM，lambda 从 0 线性增长到目标值
  3. 低 lambda 配置（对齐 dad_lowlambda 最佳结果）
  4. 更长训练（150 epoch）+ 更好的 lr schedule

Usage:
  python train_dad_curriculum.py --gpu 1 --tag dad_curriculum_v1
"""
import os, sys, json, logging, argparse, math
import random
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru  import LFCRASH_CBM_GRU
from src.data_loader import DADDataset
from src.eval_tools  import evaluation


def setup_logging(log_file):
    logger = logging.getLogger('DAD_CURRICULUM')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
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


@torch.no_grad()
def evaluate(model, loader, n_frames, fps, device, logger):
    model.eval()
    all_pred, all_labels, all_toas = [], [], []
    for xs, ys, toas in loader:
        xs = xs.to(device)
        try:
            result = model(xs, None, None)
            # Handle different return formats
            if isinstance(result, tuple) and len(result) == 3:
                _, outputs, _ = result
            elif isinstance(result, tuple) and len(result) == 2:
                _, outputs = result
            else:
                outputs = result
            # outputs may be a list of (B,2) tensors or a single (B,T,2) tensor
            if isinstance(outputs, torch.Tensor):
                # shape (B, T, 2) or (B, T)
                if outputs.dim() == 3:
                    outputs = [outputs[:, t] for t in range(outputs.shape[1])]
                else:
                    outputs = [outputs]
            T = len(outputs)
            if T == 0:
                logger.warning('Empty outputs from model, skipping batch')
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
        logger.warning('No predictions collected during eval!')
        return 0., 0., 0., 0.
    all_pred   = np.concatenate(all_pred,   axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_toas   = np.concatenate(all_toas,   axis=0)
    return evaluation(all_pred, all_labels, all_toas, fps=fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu',           type=int,   default=1)
    p.add_argument('--tag',           type=str,   default='dad_curriculum_v1')
    p.add_argument('--epochs',        type=int,   default=150)
    p.add_argument('--warmup_epochs', type=int,   default=20,
                   help='Epochs with CBM disabled (GRU warmup)')
    p.add_argument('--batch_size',    type=int,   default=16)
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--h_dim',         type=int,   default=256)
    p.add_argument('--z_dim',         type=int,   default=256,
                   help='Larger z_dim=256 for DAD (more CBM capacity)')
    p.add_argument('--lambda_align',  type=float, default=1e-6)
    p.add_argument('--lambda_sparse', type=float, default=5e-5)
    p.add_argument('--lambda_recon',  type=float, default=1e-4)
    p.add_argument('--num_concepts',  type=int,   default=837)
    p.add_argument('--eval_every',    type=int,   default=5)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--seed',          type=int,   default=42)
    args = p.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_dir = ROOT / 'output' / 'dad_curriculum' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'train.log'
    logger = setup_logging(str(log_file))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    logger.info(f'Tag: {args.tag}')
    logger.info(f'Warmup epochs (no CBM): {args.warmup_epochs}')
    logger.info(f'Total epochs: {args.epochs}')
    logger.info(f'h_dim={args.h_dim} z_dim={args.z_dim}')
    logger.info(f'lambda_align={args.lambda_align} lambda_sparse={args.lambda_sparse} lambda_recon={args.lambda_recon}')

    # Data
    meta = {'x_dim':4096,'n_obj':19,'n_frames':100,'fps':20.0}
    tr_ds = DADDataset(str(DATA_ROOT/'dad'), 'vgg16', phase='training', toTensor=False)
    te_ds = DADDataset(str(DATA_ROOT/'dad'), 'vgg16', phase='testing',  toTensor=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, collate_fn=collate_fn)
    te_ld = DataLoader(te_ds, batch_size=32, shuffle=False,
                       num_workers=2, collate_fn=collate_fn)
    logger.info(f'Train: {len(tr_ds)} | Test: {len(te_ds)}')

    # Skip CLIP encoding during training to save GPU memory and startup time.
    # lambda_align is already very small (1e-6), so alignment loss has negligible effect.
    concept_file = None
    logger.info('CLIP concept encoding disabled (lambda_align is negligible for DAD)')

    # Build model (start with use_cbm=False for warmup)
    model = LFCRASH_CBM_GRU(
        x_dim=meta['x_dim'], h_dim=args.h_dim, z_dim=args.z_dim,
        n_layers=2, n_obj=meta['n_obj'], n_frames=meta['n_frames'],
        fps=meta['fps'], with_saa=True,
        num_concepts=args.num_concepts,
        concept_file=concept_file,
        lambda_align=args.lambda_align,
        lambda_sparse=args.lambda_sparse,
        lambda_recon=args.lambda_recon,
        use_cbm=True,   # always True in model; control via lambda scaling
        device=str(device),
        legacy=False,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    best_ap   = 0.0
    best_epoch = 0
    results_path = out_dir / 'results.json'

    for epoch in range(1, args.epochs + 1):
        # ── Curriculum: scale CBM lambdas ──────────────────────────────────
        if epoch <= args.warmup_epochs:
            # Phase 1: CBM off (zero lambdas)
            cbm_scale = 0.0
            model.use_cbm = False
        else:
            # Phase 2: linearly ramp CBM from 0 → 1 over next 30 epochs
            ramp_epochs = min(30, args.epochs - args.warmup_epochs)
            cbm_scale   = min(1.0, (epoch - args.warmup_epochs) / ramp_epochs)
            model.use_cbm = True

        model.lambda_align  = args.lambda_align  * cbm_scale
        model.lambda_sparse = args.lambda_sparse * cbm_scale
        model.lambda_recon  = args.lambda_recon  * cbm_scale

        model.train()
        total_loss = 0.0
        n_batches  = 0
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
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        phase_str = 'WARMUP' if epoch <= args.warmup_epochs else f'CBM({cbm_scale:.2f})'
        logger.info(f'Epoch {epoch:3d}/{args.epochs} [{phase_str}] | Loss: {avg_loss:.4f} | '
                    f'LR: {scheduler.get_last_lr()[0]:.2e}')

        # ── Evaluation ────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            AP, mTTA, TTA_R80, P_R80 = evaluate(
                model, te_ld, meta['n_frames'], meta['fps'], device, logger)
            logger.info(f'  EVAL | AP={AP:.4f} mTTA={mTTA:.4f} '
                        f'TTA_R80={TTA_R80:.4f} P_R80={P_R80:.4f}')

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
                logger.info(f'  *** New best AP={AP:.4f} at epoch {epoch} ***')

                # Update results.json
                with open(results_path, 'w') as f:
                    json.dump({'AP': AP, 'mTTA': mTTA, 'TTA_R80': TTA_R80,
                               'P_R80': P_R80, 'epoch': epoch,
                               'dataset': 'dad', 'tag': args.tag}, f, indent=2)

    logger.info(f'\n=== DONE === Best AP={best_ap:.4f} at epoch {best_epoch}')
    logger.info(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
