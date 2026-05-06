#!/usr/bin/env python3
import os
import sys
import json
import math
import argparse
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
from src.eval_tools import evaluation

DS_META = {
    'dad': {
        'cls': DADDataset, 'feature': 'vgg16',
        'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0,
        'phase_test': 'testing',
    },
    'crash': {
        'cls': CrashDataset, 'feature': 'vgg16',
        'x_dim': 4096, 'n_obj': 19, 'n_frames': 50, 'fps': 10.0,
        'phase_test': 'test',
    },
    'a3d': {
        'cls': A3DDataset, 'feature': 'vgg16',
        'x_dim': 4096, 'n_obj': 19, 'n_frames': 100, 'fps': 20.0,
        'phase_test': 'test',
    },
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
    meta = DS_META[dataset]
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'args' in ckpt:
        saved_args = ckpt['args']
        if isinstance(saved_args, argparse.Namespace):
            saved_args = vars(saved_args)
    elif 'model_args' in ckpt:
        saved_args = ckpt['model_args']
    else:
        saved_args = {}

    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    inferred_num_concepts = None
    if 'concept_risk_w' in state:
        inferred_num_concepts = int(state['concept_risk_w'].shape[0])
    elif 'cbm.concept_proj.3.weight' in state:
        inferred_num_concepts = int(state['cbm.concept_proj.3.weight'].shape[0])

    h_dim = saved_args.get('h_dim', 256)
    z_dim = saved_args.get('z_dim', 128)
    num_concepts = inferred_num_concepts or saved_args.get('num_concepts', 837)
    no_cbm = saved_args.get('no_cbm', False)
    no_align = saved_args.get('no_align', False)
    no_sparse = saved_args.get('no_sparse', False)
    no_recon = saved_args.get('no_recon', False)
    lambda_align = 0.0 if no_align else saved_args.get('lambda_align', 1e-4)
    lambda_sparse = 0.0 if no_sparse else saved_args.get('lambda_sparse', 1e-3)
    lambda_recon = 0.0 if no_recon else saved_args.get('lambda_recon', 1e-2)

    has_cgta = any('cgta' in k for k in state.keys())
    legacy = not has_cgta

    # Avoid forcing a mismatched text-embedding cache for non-837 ontologies.
    concept_file_to_use = concept_file if (concept_file and os.path.exists(concept_file) and num_concepts == 837) else None

    model = LFCRASH_CBM_GRU(
        x_dim=meta['x_dim'], h_dim=h_dim, z_dim=z_dim,
        n_layers=saved_args.get('n_layers', 2), n_obj=meta['n_obj'],
        n_frames=meta['n_frames'], fps=meta['fps'], with_saa=True,
        num_concepts=num_concepts,
        concept_file=concept_file_to_use,
        lambda_align=lambda_align, lambda_sparse=lambda_sparse,
        lambda_recon=lambda_recon, use_cbm=not no_cbm,
        device=str(device), legacy=legacy,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_frame_scores(model, xs):
    losses, outputs, _ = model(xs, None, None)
    del losses
    T = len(outputs)
    B = xs.size(0)
    classifier = np.zeros((B, T), dtype=np.float32)
    for t, out_t in enumerate(outputs):
        classifier[:, t] = torch.softmax(out_t, dim=-1)[:, 1].cpu().numpy()

    actor = np.zeros((B, T), dtype=np.float32)
    if not getattr(model, 'use_ac', False):
        return classifier, None

    h = torch.zeros(model.n_layers, B, model.h_dim, device=xs.device)
    model._rwkv_state = None
    all_hidden = []
    prev_c_act = None

    for t in range(xs.shape[1]):
        frame = xs[:, t]
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

        if prev_c_act is not None and getattr(model, 'enable_cgta', False) and len(all_hidden) > 0 and not getattr(model, '_legacy', False):
            delta_c = c_act - prev_c_act
            h_stack = torch.stack(all_hidden, dim=1)
            q = model.cgta_q(delta_c).unsqueeze(1)
            k = model.cgta_k(h_stack)
            v = model.cgta_v(h_stack)
            attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(model.h_dim), dim=-1)
            cgta_ctx = torch.bmm(attn, v).squeeze(1)
            cgta_ctx = torch.tanh(model.cgta_gate) * cgta_ctx
        else:
            cgta_ctx = img_emb.new_zeros(B, model.h_dim)
        prev_c_act = c_act.detach()

        if getattr(model, 'enable_crs', False) and not getattr(model, '_legacy', False):
            risk_score = (c_act * torch.sigmoid(model.concept_risk_w)).sum(dim=1, keepdim=True)
            risk_feat = model.crs_proj(risk_score)
            gru_in = torch.cat([obj_ctx, c_embed, fft_vec, cgta_ctx + risk_feat], dim=1).unsqueeze(1)
        else:
            gru_in = torch.cat([obj_ctx, c_embed, fft_vec], dim=1).unsqueeze(1)

        if model.use_rwkv:
            out_t, h_last = model.gru.forward_step(gru_in.squeeze(1), state=model._rwkv_state)
            model._rwkv_state = h_last.unsqueeze(0) if h_last.dim() == 2 else h_last
            h = h_last.unsqueeze(0).expand(model.n_layers, -1, -1).contiguous()
        else:
            out_t, h = model.gru(gru_in, h)
        del out_t
        all_hidden.append(h[-1])

        ac_c = c_act if getattr(model, 'ac_use_concepts', True) else torch.zeros_like(c_act)
        logits, _, _ = model.ac_module(h[-1], ac_c)
        actor[:, t] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    return classifier, actor


def pad_to_nframes(arr, n_frames):
    if arr is None:
        return None
    B, T = arr.shape
    if T == n_frames:
        return arr
    out = np.zeros((B, n_frames), dtype=np.float32)
    out[:, :T] = arr
    if T > 0 and T < n_frames:
        out[:, T:] = arr[:, T-1:T]
    return out


def summarize_actor(actor_scores, threshold=0.5):
    peaks = actor_scores.max(axis=1)
    means = actor_scores.mean(axis=1)
    crossings = (actor_scores >= threshold).any(axis=1)
    return {
        'peak_mean': float(peaks.mean()),
        'peak_median': float(np.median(peaks)),
        'mean_mean': float(means.mean()),
        'crossing_rate_at_threshold': float(crossings.mean()),
    }


def evaluate_trigger(scores, labels, toas, fps):
    AP, mTTA, TTA_R80, P_R80 = evaluation(scores, labels, toas, fps=fps)
    return {
        'AP': float(AP),
        'mTTA': float(mTTA),
        'TTA_R80': float(TTA_R80),
        'P_R80': float(P_R80),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True, choices=['dad', 'crash', 'a3d'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--concept_file', default=DEFAULT_CONCEPT_FILE)
    parser.add_argument('--output_json', required=True)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    meta = DS_META[args.dataset]
    model, ckpt = load_model(args.checkpoint, args.dataset, device, args.concept_file)

    ds = meta['cls'](str(DATA_ROOT / args.dataset), meta['feature'], phase=meta['phase_test'], toTensor=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    all_cls, all_actor, all_labels, all_toas = [], [], [], []
    for xs, ys, toas in loader:
        xs = xs.to(device)
        cls_scores, actor_scores = extract_frame_scores(model, xs)
        cls_scores = pad_to_nframes(cls_scores, meta['n_frames'])
        actor_scores = pad_to_nframes(actor_scores, meta['n_frames']) if actor_scores is not None else None
        all_cls.append(cls_scores)
        if actor_scores is not None:
            all_actor.append(actor_scores)
        all_labels.append(ys[:, 1].numpy())
        all_toas.append(toas.numpy())

    cls_scores = np.concatenate(all_cls, axis=0)
    actor_scores = np.concatenate(all_actor, axis=0) if all_actor else None
    labels = np.concatenate(all_labels, axis=0)
    toas = np.concatenate(all_toas, axis=0)

    result = {
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'epoch': ckpt.get('epoch', '?'),
        'classifier_trigger': evaluate_trigger(cls_scores, labels, toas, meta['fps']),
    }
    if actor_scores is not None:
        result['actor_trigger'] = evaluate_trigger(actor_scores, labels, toas, meta['fps'])
        result['actor_diagnostics'] = summarize_actor(actor_scores)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
