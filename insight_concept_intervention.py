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
from src.concept_utils import load_concept_names, concept_intervention


def extract_actor_probs(model, x, device):
    x = x.to(device)
    B = x.shape[0]
    h = torch.zeros(model.n_layers, B, model.h_dim, device=device)
    model._rwkv_state = None
    prev_c_act = None
    all_hidden = []
    actor_vals = []
    for t in range(x.shape[1]):
        frame = x[:, t]
        feats = model.phi_x(frame)
        img_emb = feats[:, 0]
        obj_emb = feats[:, 1:]
        if model.use_cbm:
            c_act, c_embed = model.cbm(img_emb)
        else:
            c_act = img_emb.new_zeros(B, model.cbm.num_concepts)
            c_embed = img_emb
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
            cgta_ctx = img_emb.new_zeros(B, model.h_dim)
        prev_c_act = c_act.detach()
        if model.enable_crs:
            risk_score = (c_act * torch.sigmoid(model.concept_risk_w)).sum(dim=1, keepdim=True)
            risk_feat = model.crs_proj(risk_score)
        else:
            risk_feat = img_emb.new_zeros(B, model.h_dim)
        gru_in = torch.cat([obj_ctx, c_embed, fft_vec, cgta_ctx + risk_feat], dim=1).unsqueeze(1)
        _, h = model.gru(gru_in, h)
        all_hidden.append(h[-1])
        ac_c = c_act if model.ac_use_concepts else torch.zeros_like(c_act)
        logits, _, _ = model.ac_module(h[-1], ac_c)
        actor_vals.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu())
    return torch.stack(actor_vals, dim=1).squeeze(0).numpy()

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


def first_alert_frame(probs, threshold=0.5):
    idx = np.where(probs >= threshold)[0]
    return int(idx[0]) if len(idx) else None


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--checkpoint', required=True)
    pa.add_argument('--dataset', required=True, choices=['dad', 'crash', 'a3d'])
    pa.add_argument('--gpu', type=int, default=0)
    pa.add_argument('--concept_file', default=DEFAULT_CONCEPT_FILE)
    pa.add_argument('--output_dir', required=True)
    pa.add_argument('--topk', type=int, default=3)
    pa.add_argument('--max_samples', type=int, default=32)
    pa.add_argument('--threshold', type=float, default=0.5)
    pa.add_argument('--concept_source', choices=['activation','risk_weight','hybrid'], default='hybrid')
    pa.add_argument('--intervention_value', type=float, default=0.0)
    args = pa.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    meta = DS_META[args.dataset]

    concept_names = load_concept_names(args.concept_file) if os.path.exists(args.concept_file) else None
    model = load_model(args.checkpoint, args.dataset, device, args.concept_file)

    ds = meta['cls'](str(DATA_ROOT / args.dataset), meta['feature'], phase=meta['phase_test'], toTensor=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    rows = []
    processed = 0
    for xs, ys, toas in loader:
        label = int(ys[0, 1].item())
        if label != 1:
            continue
        x = xs.to(device)
        acts = model.get_concept_activations(x)[0].mean(dim=0)
        risk_w = torch.sigmoid(model.concept_risk_w).detach().cpu()
        if args.concept_source == 'activation':
            select_score = acts.detach().cpu()
        elif args.concept_source == 'risk_weight':
            select_score = risk_w
        else:
            select_score = acts.detach().cpu() * risk_w
        top_idx = torch.topk(select_score, k=min(args.topk, select_score.numel())).indices.tolist()

        orig_probs, intv_probs = concept_intervention(model, x, top_idx, intervention_value=args.intervention_value, device=str(device))
        orig_acc = orig_probs[:, 1].numpy()
        intv_acc = intv_probs[:, 1].numpy()
        orig_actor = extract_actor_probs(model, x, device)
        original_encode = model.cbm.encode
        def patched_encode(img_embed):
            patched = original_encode(img_embed)
            patched = patched.clone()
            patched[:, top_idx] = args.intervention_value
            return patched
        model.cbm.encode = patched_encode
        try:
            intv_actor = extract_actor_probs(model, x, device)
        finally:
            model.cbm.encode = original_encode

        orig_alert = first_alert_frame(orig_acc, args.threshold)
        intv_alert = first_alert_frame(intv_acc, args.threshold)
        orig_actor_alert = first_alert_frame(orig_actor, args.threshold)
        intv_actor_alert = first_alert_frame(intv_actor, args.threshold)
        rows.append({
            'sample_index': processed,
            'toa': float(toas[0].item()),
            'top_concepts': top_idx,
            'top_concept_names': [concept_names[i] if concept_names else str(i) for i in top_idx],
            'concept_source': args.concept_source,
            'orig_alert_frame': orig_alert,
            'intv_alert_frame': intv_alert,
            'alert_shift': None if orig_alert is None or intv_alert is None else int(intv_alert - orig_alert),
            'orig_actor_alert_frame': orig_actor_alert,
            'intv_actor_alert_frame': intv_actor_alert,
            'actor_alert_shift': None if orig_actor_alert is None or intv_actor_alert is None else int(intv_actor_alert - orig_actor_alert),
            'orig_peak_prob': float(orig_acc.max()),
            'intv_peak_prob': float(intv_acc.max()),
            'peak_prob_delta': float(intv_acc.max() - orig_acc.max()),
            'orig_actor_peak': float(orig_actor.max()),
            'intv_actor_peak': float(intv_actor.max()),
            'actor_peak_delta': float(intv_actor.max() - orig_actor.max()),
        })
        processed += 1
        if processed >= args.max_samples:
            break

    valid_shifts = [r['alert_shift'] for r in rows if r['alert_shift'] is not None]
    valid_actor_shifts = [r['actor_alert_shift'] for r in rows if r['actor_alert_shift'] is not None]
    summary = {
        'dataset': args.dataset,
        'checkpoint': args.checkpoint,
        'num_samples': len(rows),
        'concept_source': args.concept_source,
        'intervention_value': args.intervention_value,
        'mean_alert_shift': float(np.mean(valid_shifts)) if valid_shifts else None,
        'median_alert_shift': float(np.median(valid_shifts)) if valid_shifts else None,
        'mean_actor_alert_shift': float(np.mean(valid_actor_shifts)) if valid_actor_shifts else None,
        'median_actor_alert_shift': float(np.median(valid_actor_shifts)) if valid_actor_shifts else None,
        'mean_peak_prob_delta': float(np.mean([r['peak_prob_delta'] for r in rows])) if rows else None,
        'mean_actor_peak_delta': float(np.mean([r['actor_peak_delta'] for r in rows])) if rows else None,
        'rows': rows,
    }

    with open(out_dir / 'concept_intervention_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
