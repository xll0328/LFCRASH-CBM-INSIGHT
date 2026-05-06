#!/usr/bin/env python3
"""
train_dad_ac_distill.py
=======================
CG-CRASH v4 FULL: Actor-Critic + Temporally Shifted Distillation
Usage:
  python train_dad_ac_distill.py --gpu 6 --tag dad_ac_distill_v1
"""
import os, sys, json, logging, argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT       = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT  = CRASH_ROOT / 'data'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru  import LFCRASH_CBM_GRU
from src.data_loader import DADDataset
from src.eval_tools  import evaluation
from src.distillation import TemporallyShiftedDistillation, CLIPTeacher


def setup_logging(log_file):
    logger = logging.getLogger('DAD_AC_DISTILL')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(log_file); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def collate_fn(batch):
    xs, ys, toas = zip(*batch)
    xs = torch.from_numpy(np.stack(xs)).float()
    ys = torch.from_numpy(np.stack(ys)).float()
    toa_flat = []
    for t in toas:
        if isinstance(t, (list, tuple, np.ndarray)): toa_flat.append(float(t[0]))
        elif isinstance(t, torch.Tensor): toa_flat.append(t.item())
        else: toa_flat.append(float(t))
    return xs, ys, torch.tensor(toa_flat, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, n_frames, fps, device, logger):
    model.eval()
    all_pred, all_labels, all_toas = [], [], []
    for xs, ys, toas in loader:
        xs = xs.to(device)
        try:
            _, outputs, _ = model(xs, None, None)
            T = len(outputs)
            if T == 0: continue
            fp = np.zeros((xs.size(0), n_frames), dtype=np.float32)
            for t, out_t in enumerate(outputs):
                if out_t.dim() == 1: out_t = out_t.unsqueeze(0)
                p = torch.softmax(out_t, dim=-1)[:, 1].cpu().numpy()
                fp[:, t] = p
            if T < n_frames: fp[:, T:] = fp[:, T-1:T]
            all_pred.append(fp); all_labels.append(ys[:, 1].numpy()); all_toas.append(toas.numpy())
        except Exception as e:
            logger.error(f'Eval error: {e}'); continue
    if not all_pred: return 0., 0., 0., 0.
    return evaluation(np.concatenate(all_pred), np.concatenate(all_labels), np.concatenate(all_toas), fps=fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu',              type=int,   default=6)
    p.add_argument('--tag',              type=str,   default='dad_ac_distill_v1')
    p.add_argument('--epochs',           type=int,   default=150)
    p.add_argument('--warmup_epochs',    type=int,   default=15)
    p.add_argument('--cbm_ramp_epochs',  type=int,   default=20)
    p.add_argument('--distill_start',    type=int,   default=10)
    p.add_argument('--batch_size',       type=int,   default=16)
    p.add_argument('--lr',               type=float, default=3e-4)
    p.add_argument('--weight_decay',     type=float, default=1e-4)
    p.add_argument('--h_dim',            type=int,   default=256)
    p.add_argument('--lambda_align',     type=float, default=1e-6)
    p.add_argument('--lambda_sparse',    type=float, default=5e-5)
    p.add_argument('--lambda_recon',     type=float, default=1e-4)
    p.add_argument('--lambda_ac_policy', type=float, default=0.5)
    p.add_argument('--lambda_ac_value',  type=float, default=0.5)
    p.add_argument('--lambda_distill',   type=float, default=0.3)
    p.add_argument('--num_concepts',     type=int,   default=837)
    p.add_argument('--eval_every',       type=int,   default=5)
    p.add_argument('--num_workers',      type=int,   default=4)
    args = p.parse_args()

    out_dir = ROOT / 'output' / 'dad_ac_distill' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir / 'train.log'))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'=== CG-CRASH v4: AC + TSD | Device: {device} | Tag: {args.tag} ===')

    meta = {'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0}
    tr_ds = DADDataset(str(DATA_ROOT / 'dad'), 'vgg16', phase='training', toTensor=False)
    te_ds = DADDataset(str(DATA_ROOT / 'dad'), 'vgg16', phase='testing',  toTensor=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_fn)
    logger.info(f'Train: {len(tr_ds)} | Test: {len(te_ds)}')

    model = LFCRASH_CBM_GRU(
        x_dim=meta['x_dim'], h_dim=args.h_dim, z_dim=args.h_dim,
        n_layers=2, n_obj=meta['n_obj'], n_frames=meta['n_frames'],
        fps=meta['fps'], with_saa=True, num_concepts=args.num_concepts,
        concept_file=None, lambda_align=args.lambda_align,
        lambda_sparse=args.lambda_sparse, lambda_recon=args.lambda_recon,
        use_cbm=True, device=str(device), legacy=False,
    ).to(device)
    model.lambda_ac_policy = args.lambda_ac_policy
    model.lambda_ac_value  = args.lambda_ac_value

    tsd = TemporallyShiftedDistillation(
        student_dim=args.h_dim, teacher_dim=512, num_concepts=args.num_concepts,
        lambda_spatial=1.0, lambda_temporal=2.0, lambda_contrast=0.5,
    ).to(device)
    teacher_net = CLIPTeacher(vgg_dim=meta['x_dim'], teacher_dim=512, freeze=True).to(device)

    torch.manual_seed(42)
    concept_text_embs = torch.nn.functional.normalize(
        torch.randn(args.num_concepts, 512, device=device), dim=-1).detach()

    optimizer = optim.AdamW(
        list(model.parameters()) + list(tsd.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    best_ap, best_epoch = 0.0, 0
    results_path = out_dir / 'results.json'

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            cbm_scale = 0.0; ac_scale = min(1.0, epoch / max(args.warmup_epochs, 1))
            model.use_cbm = False; phase_str = f'WARMUP(ac={ac_scale:.2f})'
        else:
            ramp = min(args.cbm_ramp_epochs, args.epochs - args.warmup_epochs)
            cbm_scale = min(1.0, (epoch - args.warmup_epochs) / max(ramp, 1))
            ac_scale = 1.0; model.use_cbm = True; phase_str = f'CBM+AC+TSD({cbm_scale:.2f})'

        model.lambda_align   = args.lambda_align  * cbm_scale
        model.lambda_sparse  = args.lambda_sparse * cbm_scale
        model.lambda_recon   = args.lambda_recon  * cbm_scale
        model.lambda_ac_policy = args.lambda_ac_policy * ac_scale
        model.lambda_ac_value  = args.lambda_ac_value  * ac_scale
        use_distill = (epoch >= args.distill_start)

        if epoch == args.warmup_epochs + 1:
            teacher_net.unfreeze()
            optimizer.add_param_group({'params': list(teacher_net.parameters()), 'lr': args.lr * 0.1})
            logger.info('Teacher adapter unfrozen')

        model.train(); tsd.train()
        tot_loss = tot_distill = 0.0; n_batches = 0

        for xs, ys, toas in tr_ld:
            xs = xs.to(device); ys = ys.to(device); toas = toas.to(device)
            optimizer.zero_grad()

            losses, _, all_hidden = model(xs, ys, toas)
            loss = losses['total_loss']

            if use_distill and len(all_hidden) > 0 and model.use_cbm:
                try:
                    h_seq = torch.stack(all_hidden, dim=1)          # (B, T, h_dim)
                    B2, T2 = xs.shape[0], xs.shape[1]
                    with torch.no_grad():
                        t_feats = teacher_net(
                            xs[:, :, 0, :].reshape(B2*T2, -1)
                        ).reshape(B2, T2, 512)
                    with torch.no_grad():
                        c_seq = torch.stack(
                            [model.cbm.encode(model.phi_x(xs[:, t, 0, :]))
                             for t in range(T2)], dim=1)
                    d_loss, _ = tsd(h_seq, t_feats, c_seq, concept_text_embs)
                    d_loss = torch.clamp(d_loss, 0.0, 10.0)
                    loss = loss + args.lambda_distill * d_loss
                    tot_distill += d_loss.item()
                except Exception as e:
                    logger.warning(f'Distill error: {e}')

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f'NaN/Inf loss ep {epoch}'); continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tot_loss += loss.item(); n_batches += 1

        scheduler.step()
        avg_loss    = tot_loss    / max(n_batches, 1)
        avg_distill = tot_distill / max(n_batches, 1)
        logger.info(f'Ep {epoch:3d}/{args.epochs} [{phase_str}] '
                    f'loss={avg_loss:.4f} distill={avg_distill:.4f} '
                    f'lr={scheduler.get_last_lr()[0]:.2e}')

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            AP, mTTA, TTA_R80, P_R80 = evaluate(
                model, te_ld, meta['n_frames'], meta['fps'], device, logger)
            logger.info(f'  EVAL | AP={AP:.4f} mTTA={mTTA:.4f} TTA_R80={TTA_R80:.4f}')
            if AP > best_ap:
                best_ap = AP; best_epoch = epoch
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'AP': AP, 'mTTA': mTTA, 'args': vars(args)},
                           out_dir / 'best_model.pt')
                with open(results_path, 'w') as f:
                    json.dump({'AP': AP, 'mTTA': mTTA, 'TTA_R80': TTA_R80,
                               'P_R80': P_R80, 'epoch': epoch,
                               'dataset': 'dad', 'tag': args.tag}, f, indent=2)
                logger.info(f'  *** New best AP={AP:.4f} at epoch {epoch} ***')

    logger.info(f'\n=== DONE === Best AP={best_ap:.4f} at epoch {best_epoch}')


if __name__ == '__main__':
    main()
