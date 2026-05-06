#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent

VARIANTS = {
    'pedestrian crossing risk': ['risk of pedestrian crossing', 'pedestrian crossing hazard'],
    'oncoming traffic conflict': ['conflict with oncoming traffic', 'oncoming-traffic hazard'],
    'visibility reduction': ['reduced visibility', 'poor visibility conditions'],
    'merge conflict': ['merging conflict', 'merge-related conflict'],
    'unsafe following distance': ['dangerous following distance', 'insufficient following gap'],
    'lane change maneuver': ['lane-change maneuver', 'vehicle changing lanes'],
    'wet road surface': ['wet pavement', 'rain-slick road surface'],
    'limited lateral clearance': ['low lateral clearance', 'narrow side clearance'],
    'heavy vehicle blind spot': ['blind spot near heavy vehicle', 'heavy-truck blind spot'],
    'rear-end collision risk': ['risk of rear-end collision', 'rear-end crash hazard'],
    'motorcycle proximity': ['nearby motorcycle', 'motorcycle close by'],
    'red-light compliance risk': ['red-light violation risk', 'risk of running red light'],
}


def load_clip(device):
    import clip
    model, preprocess = clip.load('ViT-B/16', device=device)
    model.eval()
    return clip, model, preprocess


def encode_texts(clip_mod, model, texts, device):
    with torch.no_grad():
        tokens = clip_mod.tokenize(texts).to(device)
        emb = model.encode_text(tokens).float()
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb


def encode_image(model, preprocess, image_path, device):
    img = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img).float()
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb


def safe_corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def topk_overlap(a, b, k=10):
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())
    return float(len(ia & ib) / k)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_manifest', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_frames', type=int, default=120)
    parser.add_argument('--output_json', required=True)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    clip_mod, model, preprocess = load_clip(device)

    manifest = json.loads(Path(args.frame_manifest).read_text())
    manifest = [m for m in manifest if os.path.exists(m['frame_path'])][:args.max_frames]

    all_texts = []
    index = {}
    for canonical, variants in VARIANTS.items():
        index[canonical] = {'canonical': len(all_texts), 'variants': []}
        all_texts.append(canonical)
        for v in variants:
            index[canonical]['variants'].append(len(all_texts))
            all_texts.append(v)

    text_emb = encode_texts(clip_mod, model, all_texts, device)

    image_embs = []
    for row in manifest:
        image_embs.append(encode_image(model, preprocess, row['frame_path'], device))
    image_embs = torch.cat(image_embs, dim=0)
    sims = (image_embs @ text_emb.T).detach().cpu().numpy()

    results = []
    for canonical, idx_info in index.items():
        c_idx = idx_info['canonical']
        c_scores = sims[:, c_idx]
        c_emb = text_emb[c_idx:c_idx+1]
        for variant, v_idx in zip(VARIANTS[canonical], idx_info['variants']):
            v_scores = sims[:, v_idx]
            v_emb = text_emb[v_idx:v_idx+1]
            results.append({
                'canonical': canonical,
                'variant': variant,
                'text_cosine': float((c_emb @ v_emb.T).item()),
                'frame_score_correlation': safe_corr(c_scores, v_scores),
                'top10_frame_overlap': topk_overlap(c_scores, v_scores, k=min(10, len(c_scores))),
                'mean_abs_score_diff': float(np.mean(np.abs(c_scores - v_scores))),
            })

    agg = {
        'mean_text_cosine': float(np.mean([r['text_cosine'] for r in results])) if results else 0.0,
        'mean_frame_score_correlation': float(np.mean([r['frame_score_correlation'] for r in results])) if results else 0.0,
        'mean_top10_frame_overlap': float(np.mean([r['top10_frame_overlap'] for r in results])) if results else 0.0,
        'mean_abs_score_diff': float(np.mean([r['mean_abs_score_diff'] for r in results])) if results else 0.0,
    }

    out = {
        'num_frames': len(manifest),
        'num_canonical_concepts': len(VARIANTS),
        'variants_per_concept': 2,
        'aggregate': agg,
        'per_variant': results,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out['aggregate'], indent=2))


if __name__ == '__main__':
    main()
