#!/usr/bin/env python3
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT = CRASH_ROOT / 'data'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru import LFCRASH_CBM_GRU
from src.data_loader import DADDataset, CrashDataset, A3DDataset

DS_META = {
    'dad':   {'cls': DADDataset, 'feature': 'vgg16', 'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0, 'phase_test': 'testing'},
    'crash': {'cls': CrashDataset, 'feature': 'vgg16', 'x_dim': 4096, 'n_obj': 19, 'n_frames': 50,  'fps': 10.0, 'phase_test': 'test'},
    'a3d':   {'cls': A3DDataset, 'feature': 'vgg16', 'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0, 'phase_test': 'test'},
}
DEFAULT_CONCEPT_FILE = str(ROOT.parent / '000_all_concept_set.txt')


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


def load_model(checkpoint_path, dataset, device, concept_file):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get('args', {})
    state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    meta = DS_META[dataset]
    h_dim = args.get('h_dim', 256)
    gru_w = state_dict.get('gru.gru.weight_ih_l0')
    legacy = bool(gru_w is not None and gru_w.shape[1] == 3 * h_dim)
    model = LFCRASH_CBM_GRU(
        x_dim=meta['x_dim'],
        h_dim=h_dim,
        z_dim=args.get('z_dim', 128),
        n_layers=args.get('n_layers', 2),
        n_obj=meta['n_obj'],
        n_frames=meta['n_frames'],
        fps=meta['fps'],
        with_saa=True,
        num_concepts=args.get('num_concepts', 837),
        concept_file=concept_file if os.path.exists(concept_file) else None,
        lambda_align=0.0,
        lambda_sparse=0.0,
        lambda_recon=0.0,
        use_cbm=True,
        device=str(device),
        legacy=legacy,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def first_crossing(arr, threshold):
    idx = np.where(arr >= threshold)[0]
    return int(idx[0]) if len(idx) else None


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--checkpoint', required=True)
    pa.add_argument('--dataset', required=True, choices=['dad', 'crash', 'a3d'])
    pa.add_argument('--gpu', type=int, default=0)
    pa.add_argument('--concept_file', default=DEFAULT_CONCEPT_FILE)
    pa.add_argument('--output_dir', required=True)
    pa.add_argument('--max_samples', type=int, default=64)
    pa.add_argument('--surge_quantile', type=float, default=0.95)
    pa.add_argument('--alert_threshold', type=float, default=0.5)
    args = pa.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    meta = DS_META[args.dataset]
    model = load_model(args.checkpoint, args.dataset, device, args.concept_file)
    risk_weights = torch.sigmoid(model.concept_risk_w).detach().cpu().numpy()

    ds = meta['cls'](str(DATA_ROOT / args.dataset), meta['feature'], phase=meta['phase_test'], toTensor=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    rows = []
    processed = 0
    for xs, ys, toas in loader:
        if int(ys[0, 1].item()) != 1:
            continue
        x = xs.to(device)
        acts = model.get_concept_activations(x)[0].detach().cpu().numpy()  # (T, C)
        risk = (acts * risk_weights[None, :]).sum(axis=1)

        _, outputs, _ = model(x, None, None)
        pred = np.stack([torch.softmax(o, dim=-1)[0, 1].detach().cpu().numpy() for o in outputs], axis=0)

        actor = None
        if getattr(model, 'use_ac', True):
            actor_vals = []
            h = torch.zeros(model.n_layers, 1, model.h_dim, device=device)
            model._rwkv_state = None
            prev_c_act = None
            all_hidden = []
            for t in range(x.shape[1]):
                frame = x[:, t]
                feats = model.phi_x(frame)
                img_emb = feats[:, 0]
                obj_emb = feats[:, 1:]
                c_act, c_embed = model.cbm(img_emb) if model.use_cbm else (img_emb.new_zeros(1, model.cbm.num_concepts), img_emb)
                obj_ctx = model.ofa(obj_emb, h).squeeze(1)
                fft_in = model.fft_in(img_emb) if model.fft_in is not None else img_emb
                fft_out = model.fft_block(fft_in.unsqueeze(-1))
                fft_vec = model.fft_out(fft_out.mean(dim=1))
                if prev_c_act is not None and model.enable_cgta and len(all_hidden) > 0:
                    delta_c = c_act - prev_c_act
                    h_stack = torch.stack(all_hidden, dim=1)
                    q = model.cgta_q(delta_c).unsqueeze(1)
                    k = model.cgta_k(h_stack)
                    v = model.cgta_v(h_stack)
                    attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(model.h_dim), dim=-1)
                    cgta_ctx = torch.bmm(attn, v).squeeze(1)
                    cgta_ctx = torch.tanh(model.cgta_gate) * cgta_ctx
                else:
                    cgta_ctx = img_emb.new_zeros(1, model.h_dim)
                prev_c_act = c_act.detach()
                if model.enable_crs:
                    risk_score = (c_act * torch.sigmoid(model.concept_risk_w)).sum(dim=1, keepdim=True)
                    risk_feat = model.crs_proj(risk_score)
                else:
                    risk_feat = img_emb.new_zeros(1, model.h_dim)
                gru_in = torch.cat([obj_ctx, c_embed, fft_vec, cgta_ctx + risk_feat], dim=1).unsqueeze(1)
                out_t, h = model.gru(gru_in, h)
                all_hidden.append(h[-1])
                ac_c = c_act if model.ac_use_concepts else torch.zeros_like(c_act)
                logits, _, _ = model.ac_module(h[-1], ac_c)
                actor_vals.append(torch.softmax(logits, dim=-1)[0, 1].item())
            actor = np.array(actor_vals)

        delta = np.diff(acts, axis=0)
        surge_score = np.maximum(delta, 0).max(axis=1) if delta.size else np.array([])
        surge_thr = np.quantile(surge_score, args.surge_quantile) if surge_score.size else None
        surge_idx = first_crossing(surge_score, surge_thr) if surge_thr is not None else None
        risk_thr = np.quantile(risk, args.surge_quantile)
        risk_idx = first_crossing(risk, risk_thr)
        pred_idx = first_crossing(pred, args.alert_threshold)
        actor_idx = first_crossing(actor, args.alert_threshold) if actor is not None else None

        rows.append({
            'sample_index': processed,
            'toa': float(toas[0].item()),
            'surge_frame': surge_idx,
            'risk_frame': risk_idx,
            'pred_frame': pred_idx,
            'actor_frame': actor_idx,
            'surge_to_actor': None if surge_idx is None or actor_idx is None else int(actor_idx - surge_idx),
            'risk_to_actor': None if risk_idx is None or actor_idx is None else int(actor_idx - risk_idx),
            'actor_to_toa': None if actor_idx is None else float(toas[0].item() - actor_idx),
            'pred_to_toa': None if pred_idx is None else float(toas[0].item() - pred_idx),
            'actor_peak': None if actor is None else float(actor.max()),
            'actor_mean': None if actor is None else float(actor.mean()),
        })
        processed += 1
        if processed >= args.max_samples:
            break

    def summarize(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return None if not vals else {'mean': float(np.mean(vals)), 'median': float(np.median(vals)), 'count': len(vals)}

    actor_peak_vals = [r['actor_peak'] for r in rows if r['actor_peak'] is not None]
    actor_mean_vals = [r['actor_mean'] for r in rows if r['actor_mean'] is not None]
    summary = {
        'dataset': args.dataset,
        'checkpoint': args.checkpoint,
        'num_samples': len(rows),
        'surge_to_actor': summarize('surge_to_actor'),
        'risk_to_actor': summarize('risk_to_actor'),
        'actor_to_toa': summarize('actor_to_toa'),
        'pred_to_toa': summarize('pred_to_toa'),
        'actor_peak': None if not actor_peak_vals else {'mean': float(np.mean(actor_peak_vals)), 'median': float(np.median(actor_peak_vals))},
        'actor_mean': None if not actor_mean_vals else {'mean': float(np.mean(actor_mean_vals)), 'median': float(np.median(actor_mean_vals))},
        'rows': rows,
    }

    with open(out_dir / 'timing_faithfulness_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
