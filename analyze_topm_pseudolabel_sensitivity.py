#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent


def load_clip(device):
    import clip
    model, preprocess = clip.load('ViT-B/16', device=device)
    model.eval()
    return clip, model, preprocess


def load_concepts(concept_meta_path):
    meta = json.loads(Path(concept_meta_path).read_text())
    families = meta['families']
    concepts = []
    concept_to_family = {}
    for family, items in families.items():
        for c in items:
            concepts.append(c)
            concept_to_family[c] = family
    return concepts, concept_to_family


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


def summarize_counts(counter):
    total = sum(counter.values())
    return {k: {'count': int(v), 'frac': float(v / total) if total else 0.0} for k, v in counter.most_common()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_manifest', required=True)
    parser.add_argument('--concept_meta', required=True)
    parser.add_argument('--topms', nargs='+', type=int, default=[1, 3, 5, 10])
    parser.add_argument('--max_frames', type=int, default=120)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_json', required=True)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    clip_mod, model, preprocess = load_clip(device)

    manifest = json.loads(Path(args.frame_manifest).read_text())
    manifest = [m for m in manifest if os.path.exists(m['frame_path'])][:args.max_frames]

    concepts, concept_to_family = load_concepts(args.concept_meta)
    text_emb = encode_texts(clip_mod, model, concepts, device)

    topms = sorted(set(args.topms))
    concept_counts = {m: Counter() for m in topms}
    family_counts = {m: Counter() for m in topms}
    frame_family_diversity = {m: [] for m in topms}
    frame_score_mass = {m: [] for m in topms}
    overlap_vs_top3 = defaultdict(list)

    top3_sets = []
    scored_frames = 0
    for row in manifest:
        image_emb = encode_image(model, preprocess, row['frame_path'], device)
        sims = (image_emb @ text_emb.T).squeeze(0).detach().cpu().numpy()
        order = np.argsort(-sims)
        top3 = set(order[:3].tolist())
        top3_sets.append(top3)
        for m in topms:
            idx = order[:m]
            fams = set()
            score_mass = float(sims[idx].sum() / max(1e-8, sims[order[:min(len(order), 20)]].sum()))
            frame_score_mass[m].append(score_mass)
            for i in idx:
                c = concepts[int(i)]
                f = concept_to_family[c]
                concept_counts[m][c] += 1
                family_counts[m][f] += 1
                fams.add(f)
            frame_family_diversity[m].append(len(fams))
            if m != 3:
                overlap_vs_top3[m].append(len(set(idx.tolist()) & top3) / max(1, len(top3)))
        scored_frames += 1

    summary = {
        'frame_manifest': args.frame_manifest,
        'concept_meta': args.concept_meta,
        'num_frames': scored_frames,
        'topm_summary': {}
    }
    for m in topms:
        summary['topm_summary'][str(m)] = {
            'avg_family_diversity_per_frame': float(np.mean(frame_family_diversity[m])) if frame_family_diversity[m] else 0.0,
            'avg_relative_score_mass_vs_top20': float(np.mean(frame_score_mass[m])) if frame_score_mass[m] else 0.0,
            'family_distribution': summarize_counts(family_counts[m]),
            'top_concepts': summarize_counts(concept_counts[m]),
        }
        if m != 3:
            summary['topm_summary'][str(m)]['mean_overlap_with_top3'] = float(np.mean(overlap_vs_top3[m])) if overlap_vs_top3[m] else 0.0

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        'num_frames': scored_frames,
        'topms': topms,
        'avg_family_diversity': {str(m): summary['topm_summary'][str(m)]['avg_family_diversity_per_frame'] for m in topms}
    }, indent=2))


if __name__ == '__main__':
    main()
