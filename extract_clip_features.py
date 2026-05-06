#!/usr/bin/env python3
"""
extract_clip_features.py — 用 CLIP ViT-B/16 重新提取 DAD/A3D 视频特征
替换 VGG16 特征，预期 AP 提升 10-15 个百分点

Usage:
  python extract_clip_features.py --dataset dad --split training
  python extract_clip_features.py --dataset dad --split testing
"""
import os, sys, json, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import clip
from PIL import Image
import cv2

ROOT       = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT  = CRASH_ROOT / 'data'

DATASET_CFG = {
    'dad': {
        'video_dir': DATA_ROOT / 'dad' / 'video',
        'out_dir':   DATA_ROOT / 'dad' / 'clip_vit_features',
        'n_frames':  100,
        'fps':       20.0,
        'splits':    ['training', 'testing'],
    },
    'a3d': {
        'video_dir': DATA_ROOT / 'a3d' / 'video',
        'out_dir':   DATA_ROOT / 'a3d' / 'clip_vit_features',
        'n_frames':  100,
        'fps':       10.0,
        'splits':    ['train', 'test'],
    },
}


def extract_frames(video_path: str, n_frames: int) -> list:
    """均匀采样 n_frames 帧."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        else:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
    cap.release()
    return frames


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--dataset', type=str, default='dad', choices=['dad', 'a3d'])
    pa.add_argument('--split',   type=str, default='training')
    pa.add_argument('--gpu',     type=int, default=0)
    pa.add_argument('--model',   type=str, default='ViT-B/16',
                    choices=['ViT-B/16', 'ViT-L/14', 'ViT-B/32'])
    pa.add_argument('--batch',   type=int, default=32)
    args = pa.parse_args()

    cfg    = DATASET_CFG[args.dataset]
    device = torch.device(f'cuda:{args.gpu}')

    print(f'Loading CLIP {args.model}...')
    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    feat_dim = 512 if 'B' in args.model else 768  # ViT-B→512, ViT-L→768
    n_frames = cfg['n_frames']

    # Output dirs
    out_dir = cfg['out_dir'] / args.split
    (out_dir / 'positive').mkdir(parents=True, exist_ok=True)
    (out_dir / 'negative').mkdir(parents=True, exist_ok=True)

    # Find videos
    video_dir = cfg['video_dir'] / args.split
    if not video_dir.exists():
        print(f'Video dir not found: {video_dir}')
        return

    for label in ['positive', 'negative']:
        label_dir = video_dir / label
        if not label_dir.exists():
            continue
        videos = sorted(label_dir.glob('*.mp4'))
        print(f'\n[{label}] {len(videos)} videos')

        for vpath in tqdm(videos, desc=f'{args.dataset}/{args.split}/{label}'):
            out_path = out_dir / label / (vpath.stem + '.npy')
            if out_path.exists():
                continue

            frames = extract_frames(str(vpath), n_frames)
            if not frames:
                print(f'  Skip empty: {vpath.name}')
                continue

            # Batch encode
            all_feats = []
            for i in range(0, len(frames), args.batch):
                batch = torch.stack([
                    preprocess(f) for f in frames[i:i+args.batch]
                ]).to(device)
                with torch.no_grad():
                    feats = model.encode_image(batch)  # (B, feat_dim)
                    feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize
                all_feats.append(feats.cpu().numpy())

            feat_arr = np.concatenate(all_feats, axis=0)  # (n_frames, feat_dim)
            np.save(str(out_path), feat_arr)

        # Write txt list
        txt_path = cfg['out_dir'] / f'{args.split}.txt'
        with open(txt_path, 'w') as f:
            for label in ['positive', 'negative']:
                label_dir = video_dir / label
                if not label_dir.exists(): continue
                for vpath in sorted(label_dir.glob('*.mp4')):
                    feat_path = out_dir / label / (vpath.stem + '.npy')
                    lbl = 1 if label == 'positive' else 0
                    f.write(f'{feat_path} {lbl}\n')

    print(f'\nDone! Features saved to {cfg["out_dir"]}')
    print(f'Feature dim: {feat_dim} (vs VGG16: 4096)')
    print(f'To use: change x_dim={feat_dim} in train_dad_ac.py')


if __name__ == '__main__':
    main()
