#!/usr/bin/env python3
from pathlib import Path
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cv2

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
DATA = Path('/data/sony/LFCRASH/CRASH/data/dad')
FEAT_DIR = DATA / 'vgg16_features' / 'testing'
VID_DIR = DATA / 'videos' / 'testing'
SHORTLIST = ROOT / 'paper' / 'neurips2026' / 'v4_case_bank_shortlist.json'
OUT = ROOT / 'paper' / 'figures' / 'insight_fig_appendix_v4_casebank_selected.png'
FEAT_FILES = sorted([p for p in FEAT_DIR.iterdir() if p.suffix == '.npz'])

RED='#D62728'; GOLD='#F0C040'; FG='#111111'; SUB='#555555'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9})

def sample_meta(idx):
    d = np.load(str(FEAT_FILES[idx]), allow_pickle=True)
    vid = str(d['ID']).split('_')[-1]
    return {'vid': vid, 'video': VID_DIR / f'{vid}.mp4', 'det': d['det']}

def read_frames(vpath, fi_arr, boxes, size=(300,168)):
    cap = cv2.VideoCapture(str(vpath)); out = []
    for fi in fi_arr:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
        if not ok:
            fr = np.zeros((size[1], size[0], 3), np.uint8)
        else:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            sx, sy = size[0] / fr.shape[1], size[1] / fr.shape[0]
            fr = cv2.resize(fr, size)
            b = boxes[min(int(fi), len(boxes)-1)]
            if len(b):
                areas = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
                for j in np.argsort(areas)[::-1][:4]:
                    x1,y1,x2,y2 = b[j,:4]
                    if x2>x1 and y2>y1:
                        cv2.rectangle(fr,(int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),(255,210,0),2)
        out.append(fr)
    cap.release(); return out

if __name__ == '__main__':
    data = json.loads(SHORTLIST.read_text())['selected']
    picks = [
        ('strong_primary', 'Strong early warning'),
        ('strong_secondary', 'Secondary strong case'),
        ('borderline_positive', 'Borderline but positive'),
        ('borderline_negative', 'Near-threshold miss'),
        ('near_threshold_alt', 'Near-threshold alternative'),
        ('night_visibility_alt', 'Night / low-visibility case'),
    ]

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    outer = gridspec.GridSpec(3, 2, figure=fig, wspace=0.18, hspace=0.28)
    fig.suptitle('Selected v4 DAD Case Bank for Appendix-Level Qualitative Evidence', fontsize=15, fontweight='bold', y=0.985)

    for slot, (key, label) in enumerate(picks):
        item = data[key]
        idx, toa = item['idx'], item['toa']
        meta = sample_meta(idx)
        fi = np.unique(np.clip(np.linspace(max(0, toa-55), min(99, toa+3), 4, dtype=int), 0, 99))
        frames = read_frames(meta['video'], fi, meta['det'])
        strip = np.concatenate(frames, axis=1)
        H, W = strip.shape[:2]; wf = W / len(frames)

        sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[slot], height_ratios=[3.2, 1.1], hspace=0.05)
        ax_img = fig.add_subplot(sub[0])
        ax_txt = fig.add_subplot(sub[1])

        ax_img.imshow(strip)
        for j, ff in enumerate(fi):
            near = abs(ff - toa) <= 7
            if near:
                ax_img.add_patch(Rectangle((j*wf, 0), wf, H, fc=RED, alpha=0.16, ec=RED, lw=2.0))
            ax_img.text((j+0.5)*wf, H+7, f'{ff/20.0:.1f}s', ha='center', va='top', fontsize=8, color=RED if near else SUB, fontweight='bold' if near else 'normal')
        ax_img.set_xlim(0, W); ax_img.set_ylim(H+18, -2)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.set_title(f'{label}  |  idx={idx}  |  pred TTA={item["pred_tta"]:.2f}s  |  max={item["pred_max"]:.3f}', fontsize=10.2, fontweight='bold', pad=4)
        for sp in ax_img.spines.values(): sp.set_visible(False)

        ax_txt.axis('off')
        txt = 'Top concepts: ' + '; '.join(item['top_names'][:4])
        ax_txt.text(0.0, 0.72, txt, fontsize=8.7, color=FG, wrap=True)
        ax_txt.text(0.0, 0.18, f'Video {meta["vid"]}  |  Accident @ {toa/20.0:.1f}s', fontsize=8.3, color=SUB)

    plt.savefig(str(OUT), dpi=220, bbox_inches='tight')
    print(json.dumps({'saved': str(OUT)}, indent=2))
