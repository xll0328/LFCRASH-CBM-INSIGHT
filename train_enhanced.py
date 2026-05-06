#!/usr/bin/env python3
"""LFCRASH-CBM v4 Enhanced Training with Logging & Stability"""
import os, sys, json, logging, argparse, math
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.multiprocessing as torch_mp

ROOT = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT = CRASH_ROOT / 'data'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru import LFCRASH_CBM_GRU
from src.data_loader import DADDataset, CrashDataset, A3DDataset
from src.eval_tools import evaluation

try:
    torch_mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass

DS_META = {
    'dad': {'cls': DADDataset, 'feature': 'vgg16', 'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0, 'phase_train': 'training', 'phase_test': 'testing'},
    'crash': {'cls': CrashDataset, 'feature': 'vgg16', 'x_dim': 4096, 'n_obj': 19, 'n_frames': 50, 'fps': 10.0, 'phase_train': 'train', 'phase_test': 'test'},
    'a3d': {'cls': A3DDataset, 'feature': 'vgg16', 'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0, 'phase_train': 'train', 'phase_test': 'test'},
}

def setup_logging(log_file):
    logger = logging.getLogger('LFCRASH')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
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
def evaluate(model, loader, meta, device, logger):
    model.eval()
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
                frame_probs[:, T:] = frame_probs[:, T - 1:T]
            all_pred.append(frame_probs)
            all_labels.append(ys[:, 1].numpy())
            all_toas.append(toas.numpy())
        except Exception as e:
            logger.error(f"Eval error: {e}")
            return 0.0, 0.0, 0.0, 0.0
    all_pred = np.concatenate(all_pred, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_toas = np.concatenate(all_toas, axis=0)
    AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toas, fps=meta['fps'])
    return AP, mTTA, TTA_R80, P_R80

def get_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    meta = DS_META[args.dataset]
    ds_path = DATA_ROOT / args.dataset
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = args.tag if args.tag else f'{args.dataset}_{ts}'
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(out / 'train.log')
    logger.info(f"LFCRASH-CBM v4 Enhanced | Dataset: {args.dataset} | Output: {out}")
    logger.info(json.dumps(vars(args), indent=2))
    
    DS = meta['cls']
    tr = DS(str(ds_path), meta['feature'], phase=meta['phase_train'], toTensor=False)
    te = DS(str(ds_path), meta['feature'], phase=meta['phase_test'], toTensor=False)
    logger.info(f"Train: {len(tr)}, Test: {len(te)}")

    # Subset training data for data efficiency experiments
    if hasattr(args, 'train_fraction') and args.train_fraction < 1.0:
        n_sub = max(1, int(len(tr) * args.train_fraction))
        import random; random.seed(getattr(args,'seed',42))
        sub_idx = random.sample(range(len(tr)), n_sub)
        from torch.utils.data import Subset
        tr = Subset(tr, sub_idx)
        logger.info(f"Using {n_sub}/{len(te)+n_sub} training samples ({args.train_fraction*100:.0f}%)")

    # Handle eval_every as alias for eval_interval
    if hasattr(args, 'eval_every') and args.eval_every is not None:
        args.eval_interval = args.eval_every
    
    pin_memory = device.type == 'cuda' and args.num_workers > 0
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True, pin_memory=pin_memory)
    test_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    
    concept_file = str(ROOT.parent / '000_all_concept_set.txt')
    if not os.path.exists(concept_file):
        concept_file = None
        logger.warning('concept_file not found')
    
    model = LFCRASH_CBM_GRU(
        x_dim=meta['x_dim'], h_dim=args.h_dim, z_dim=args.z_dim, n_layers=2, n_obj=meta['n_obj'], n_frames=meta['n_frames'], fps=meta['fps'], with_saa=True,
        num_concepts=args.num_concepts, concept_file=concept_file,
        lambda_align=0.0 if args.no_align else args.lambda_align,
        lambda_sparse=0.0 if args.no_sparse else args.lambda_sparse,
        lambda_recon=0.0 if args.no_recon else args.lambda_recon,
        use_cbm=not args.no_cbm, device=str(device),
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, warmup_epochs=3, total_epochs=args.epochs, steps_per_epoch=len(train_loader))
    
    best_ap, best_epoch, nan_count, inf_count = 0.0, 0, 0, 0
    best_metrics = None
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        epoch_losses = []
        
        for xs, ys, toas in pbar:
            xs, ys, toas = xs.to(device), ys.to(device), toas.to(device)
            optimizer.zero_grad()
            
            try:
                # model returns (losses_dict, all_outputs, all_hidden)
                losses_dict, outputs, all_hidden = model(xs, ys, toas)
                
                total_loss = losses_dict['total_loss']
                
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    if torch.isnan(total_loss):
                        nan_count += 1
                    else:
                        inf_count += 1
                    logger.warning(f"Invalid loss: {total_loss.item()}")
                    continue
                
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"Invalid grad_norm: {grad_norm}")
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
                scheduler.step()
                
                epoch_losses.append(total_loss.item())
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'ce': f"{losses_dict['ce_loss'].item():.4f}",
                    'align': f"{losses_dict['align_loss'].item():.4f}",
                    'grad': f"{grad_norm:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                })
                
            except Exception as e:
                logger.error(f"Batch error: {e}")
                optimizer.zero_grad()
                continue
        
        if epoch_losses:
            logger.info(f"Epoch {epoch} | Loss: {np.mean(epoch_losses):.4f}")
        
        if epoch % args.eval_interval == 0:
            logger.info("Evaluating...")
            AP, mTTA, TTA_R80, P_R80 = evaluate(model, test_loader, meta, device, logger)
            logger.info(f"AP={AP:.4f}, mTTA={mTTA:.4f}, TTA_R80={TTA_R80:.4f}, P_R80={P_R80:.4f}")
            
            if AP > best_ap:
                best_ap, best_epoch = AP, epoch
                best_metrics = {
                    'AP': float(AP),
                    'mTTA': float(mTTA),
                    'TTA_R80': float(TTA_R80),
                    'P_R80': float(P_R80),
                    'epoch': int(epoch),
                    'dataset': args.dataset,
                    'tag': tag,
                    'seed': int(args.seed),
                    'nan_count': int(nan_count),
                    'inf_count': int(inf_count),
                }
                ckpt_path = out / 'best_model.pt'
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'AP': AP}, ckpt_path)
                logger.info(f"Saved best model (AP={AP:.4f})")

    logger.info(f"Training Complete! Best AP: {best_ap:.4f} at epoch {best_epoch}")
    results = best_metrics or {
        'best_ap': float(best_ap),
        'best_epoch': int(best_epoch),
        'dataset': args.dataset,
        'tag': tag,
        'seed': int(args.seed),
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
    }
    with open(out / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='crash', choices=['dad', 'crash', 'a3d'])
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--num_concepts', type=int, default=837)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_align', type=float, default=1e-4)
    parser.add_argument('--lambda_sparse', type=float, default=1e-3)
    parser.add_argument('--lambda_recon', type=float, default=1e-2)
    parser.add_argument('--no_cbm', action='store_true')
    parser.add_argument('--no_align', action='store_true')
    parser.add_argument('--no_sparse', action='store_true')
    parser.add_argument('--no_recon', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='output/v4_enhanced')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--eval_every', type=int, default=None,
                        help='Alias for eval_interval')
    parser.add_argument('--train_fraction', type=float, default=1.0,
                        help='Fraction of training data to use (0.0-1.0)')
    args = parser.parse_args()
    train(args)
