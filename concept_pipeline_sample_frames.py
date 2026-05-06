#!/usr/bin/env python3
import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

import cv2

ROOT = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT = CRASH_ROOT / 'data'

VIDEO_SOURCES = {
    'dad': [
        DATA_ROOT / 'dad' / 'video' / 'training' / 'positive',
        DATA_ROOT / 'dad' / 'video' / 'training' / 'negative',
        DATA_ROOT / 'dad' / 'video' / 'testing' / 'positive',
        DATA_ROOT / 'dad' / 'video' / 'testing' / 'negative',
    ],
    'crash': [
        DATA_ROOT / 'crash' / 'videos' / 'Crash-1500',
        DATA_ROOT / 'crash' / 'videos' / 'Normal',
    ],
    'a3d': [
        DATA_ROOT / 'a3d',
    ],
}

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}


def list_videos(dataset: str) -> List[Path]:
    roots = VIDEO_SOURCES[dataset]
    videos = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if p.suffix.lower() in VIDEO_EXTS:
                videos.append(p)
    return sorted(videos)


def sample_frames(video_path: Path, num_frames: int) -> List[Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = sorted(set(int(i) for i in [total * (k + 1) / (num_frames + 1) for k in range(num_frames)]))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    return frames


def main():
    p = argparse.ArgumentParser(description='Sample videos and extract frames for concept discovery')
    p.add_argument('--datasets', nargs='+', default=['dad', 'crash'])
    p.add_argument('--max_videos_per_dataset', type=int, default=50)
    p.add_argument('--frames_per_video', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for ds in args.datasets:
        videos = list_videos(ds)
        if len(videos) > args.max_videos_per_dataset:
            videos = random.sample(videos, args.max_videos_per_dataset)
        for video_path in videos:
            sampled = sample_frames(video_path, args.frames_per_video)
            for frame_idx, frame in sampled:
                rel_name = f"{ds}__{video_path.stem}__f{frame_idx:06d}.jpg"
                save_path = frames_dir / rel_name
                cv2.imwrite(str(save_path), frame)
                manifest.append({
                    'dataset': ds,
                    'video_path': str(video_path),
                    'frame_index': frame_idx,
                    'frame_path': str(save_path),
                })

    with open(out_dir / 'frame_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({'num_frames': len(manifest), 'output_dir': str(out_dir)}, indent=2))


if __name__ == '__main__':
    main()
