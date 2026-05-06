#!/usr/bin/env python3
"""
train_multi.py — CG-CRASH v4 多数据集训练
支持 DAD / A3D / CCD (crash) 数据集
Usage:
  python train_multi.py --dataset a3d --gpu 5 --tag a3d_ac_v1
  python train_multi.py --dataset dad --gpu 4 --tag dad_ac_v3
  python train_multi.py --dataset crash --gpu 6 --tag crash_ac_v1
"""
import os, sys, json, logging, argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as torch_mp

ROOT       = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT  = CRASH_ROOT / 'data'
DEFAULT_CONCEPT_FILE = str(ROOT.parent / '000_all_concept_set.txt')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru  import LFCRASH_CBM_GRU
from src.eval_tools  import evaluation

try:
    torch_mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass

DATASET_CFG = {
    'dad':   {'x_dim':4096,'n_obj':19,'n_frames':100,'fps':20.0,'data_dir':'dad'},
    'a3d':   {'x_dim':4096,'n_obj':19,'n_frames':100,'fps':10.0,'data_dir':'a3d'},
    'crash': {'x_dim':4096,'n_obj':19,'n_frames':50, 'fps':10.0,'data_dir':'crash'},
}


def get_dataset(dataset_name, phase):
    from src.DataLoader import DADDataset, A3DDataset, CrashDataset
    data_dir = str(DATA_ROOT / DATASET_CFG[dataset_name]['data_dir'])
    if dataset_name == 'dad':
        # DAD uses 'training'/'testing'
        return DADDataset(data_dir, 'vgg16', phase=phase, toTensor=False)
    elif dataset_name == 'a3d':
        # A3D uses 'train'/'test'
        phase_map = {'training': 'train', 'testing': 'test'}
        return A3DDataset(data_dir, 'vgg16', phase=phase_map.get(phase, phase), toTensor=False)
    elif dataset_name == 'crash':
        # CrashDataset uses 'train'/'test'
        phase_map = {'training': 'train', 'testing': 'test'}
        return CrashDataset(data_dir, 'vgg16', phase=phase_map.get(phase, phase), toTensor=False)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')


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


def setup_logging(log_file):
    logger = logging.getLogger('MULTI')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(log_file); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


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
            _, outputs, _ = model(xs, None, None)
            T = len(outputs)
            if T == 0: continue
            fp = np.zeros((xs.size(0), n_frames), dtype=np.float32)
            for t, out_t in enumerate(outputs):
                if out_t.dim() == 1: out_t = out_t.unsqueeze(0)
                p = torch.softmax(out_t, dim=-1)[:, 1].cpu().numpy()
                fp[:, t] = p
            if T < n_frames: fp[:, T:] = fp[:, T-1:T]
            all_pred.append(fp)
            all_labels.append(ys[:, 1].numpy())
            all_toas.append(toas.numpy())
        except Exception as e:
            logger.error(f'Eval error: {e}'); continue
    if not all_pred: return 0., 0., 0., 0.
    return evaluation(np.concatenate(all_pred), np.concatenate(all_labels),
                      np.concatenate(all_toas), fps=fps)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--dataset',         type=str,   default='a3d',
                    choices=['dad','a3d','crash'])
    pa.add_argument('--gpu',             type=int,   default=5)
    pa.add_argument('--tag',             type=str,   default='a3d_ac_v1')
    pa.add_argument('--epochs',          type=int,   default=150)
    pa.add_argument('--warmup_epochs',   type=int,   default=15)
    pa.add_argument('--cbm_ramp_epochs', type=int,   default=20)
    pa.add_argument('--batch_size',      type=int,   default=16)
    pa.add_argument('--lr',              type=float, default=3e-4)
    pa.add_argument('--weight_decay',    type=float, default=1e-4)
    pa.add_argument('--h_dim',           type=int,   default=256)
    pa.add_argument('--lambda_align',    type=float, default=1e-6)
    pa.add_argument('--lambda_sparse',   type=float, default=5e-5)
    pa.add_argument('--lambda_recon',    type=float, default=1e-4)
    pa.add_argument('--lambda_ac_policy',type=float, default=0.3)
    pa.add_argument('--lambda_ac_value', type=float, default=0.3)
    pa.add_argument('--num_concepts',    type=int,   default=837)
    pa.add_argument('--eval_every',      type=int,   default=5)
    pa.add_argument('--num_workers',     type=int,   default=8)
    pa.add_argument('--seed',            type=int,   default=42)
    pa.add_argument('--disable_cgta',    action='store_true')
    pa.add_argument('--disable_crs',     action='store_true')
    pa.add_argument('--ac_no_concepts',  action='store_true')
    pa.add_argument('--disable_ac',      action='store_true')
    pa.add_argument('--concept_file',    type=str,   default=DEFAULT_CONCEPT_FILE)
    args = pa.parse_args()

    cfg = DATASET_CFG[args.dataset]
    out_dir = ROOT / f'output/{args.dataset}_ac' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir / 'train.log'))

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'=== CG-CRASH v4 | Dataset: {args.dataset.upper()} | Tag: {args.tag} ===')
    logger.info(f'Device: {device} | n_frames={cfg["n_frames"]} fps={cfg["fps"]}')
    logger.info(f'Seed: {args.seed}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tr_ds = get_dataset(args.dataset, 'training')
    te_ds = get_dataset(args.dataset, 'testing')
    pin_memory = torch.cuda.is_available() and args.num_workers > 0

    tr_ld = make_loader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    te_ld = make_loader(
        te_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0 if args.num_workers == 0 else max(2, min(4, args.num_workers // 2)),
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    logger.info(f'Train: {len(tr_ds)} | Test: {len(te_ds)}')

    model = LFCRASH_CBM_GRU(
        x_dim=cfg['x_dim'], h_dim=args.h_dim, z_dim=args.h_dim,
        n_layers=2, n_obj=cfg['n_obj'], n_frames=cfg['n_frames'],
        fps=cfg['fps'], with_saa=True, num_concepts=args.num_concepts,
        concept_file=args.concept_file if args.concept_file and os.path.exists(args.concept_file) else None,
        lambda_align=args.lambda_align,
        lambda_sparse=args.lambda_sparse, lambda_recon=args.lambda_recon,
        use_cbm=True, device=str(device), legacy=False,
    ).to(device)
    model.enable_cgta = not args.disable_cgta
    model.enable_crs = not args.disable_crs
    model.ac_use_concepts = not args.ac_no_concepts
    model.use_ac = not args.disable_ac
    model.lambda_ac_policy = 0.0 if args.disable_ac else args.lambda_ac_policy
    model.lambda_ac_value  = 0.0 if args.disable_ac else args.lambda_ac_value

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6)
    _scheduler_reset = False

    best_ap, best_epoch = 0.0, 0
    results_path = out_dir / 'results.json'

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            cbm_scale = 0.0; ac_scale = min(1.0, epoch / max(args.warmup_epochs, 1))
            model.use_cbm = False; phase_str = f'WARMUP(ac={ac_scale:.2f})'
        else:
            ramp = min(args.cbm_ramp_epochs, args.epochs - args.warmup_epochs)
            cbm_scale = min(1.0, (epoch - args.warmup_epochs) / max(ramp, 1))
            ac_scale = 1.0; model.use_cbm = True
            phase_str = f'CBM({cbm_scale:.2f})+AC'

        model.lambda_align   = args.lambda_align  * cbm_scale
        model.lambda_sparse  = args.lambda_sparse * cbm_scale
        model.lambda_recon   = args.lambda_recon  * cbm_scale
        model.lambda_ac_policy = args.lambda_ac_policy * ac_scale
        model.lambda_ac_value  = args.lambda_ac_value  * ac_scale

        # Reset lr at CBM phase start
        _skip_sched = False
        if epoch == args.warmup_epochs + 1 and not _scheduler_reset:
            for pg in optimizer.param_groups: pg['lr'] = args.lr
            remaining = max(args.epochs - args.warmup_epochs, 50)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=remaining, T_mult=1, eta_min=1e-6)
            _scheduler_reset = True
            _skip_sched = True
            logger.info(f'LR reset to {args.lr:.2e} for CBM phase')

        model.train()
        tot_loss = 0.0; n_batches = 0
        for xs, ys, toas in tr_ld:
            xs = xs.to(device); ys = ys.to(device); toas = toas.to(device)
            optimizer.zero_grad()
            losses, _, _ = model(xs, ys, toas)
            loss = losses['total_loss']
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tot_loss += loss.item(); n_batches += 1
        if not _skip_sched:
            scheduler.step()

        avg_loss = tot_loss / max(n_batches, 1)
        logger.info(f'Ep {epoch:3d}/{args.epochs} [{phase_str}] '
                    f'loss={avg_loss:.4f} lr={optimizer.param_groups[0]["lr"]:.2e}')

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            AP, mTTA, TTA_R80, P_R80 = evaluate(
                model, te_ld, cfg['n_frames'], cfg['fps'], device, logger)
            logger.info(f'  EVAL | AP={AP:.4f} mTTA={mTTA:.4f} TTA_R80={TTA_R80:.4f}')
            if AP > best_ap:
                best_ap = AP; best_epoch = epoch
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'AP': AP, 'mTTA': mTTA, 'args': vars(args)},
                           out_dir / 'best_model.pt')
                with open(results_path, 'w') as f:
                    json.dump({'AP': AP, 'mTTA': mTTA, 'TTA_R80': TTA_R80,
                               'P_R80': P_R80, 'epoch': epoch,
                               'dataset': args.dataset, 'tag': args.tag}, f, indent=2)
                logger.info(f'  *** New best AP={AP:.4f} at epoch {epoch} ***')

    logger.info(f'\n=== DONE === Best AP={best_ap:.4f} at epoch {best_epoch}')


if __name__ == '__main__':
    main()
