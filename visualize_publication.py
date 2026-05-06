#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_publication.py  –  Publication-Quality Figures for CG-CRASH
======================================================================
Produces figures suitable for CVPR/ICCV/AAAI/IJCAI submission:

  pub_fig1_casestudy.png   – Video frames + concept activations (THE hero figure)
  pub_fig2_concepts.png    – Semantic concept importance (named concepts)
  pub_fig3_timeline.png    – Frame strip + prediction curve + concept heatmap
  pub_fig4_ablation.png    – Ablation results (clean grouped bar + numbers)
  pub_fig5_tsne.png        – Concept space (publication style)

Usage:
  python visualize_publication.py --dataset crash --gpu 0
"""
import os, sys, json, argparse, re, warnings
from pathlib import Path
from typing import List, Tuple, Optional
from io import BytesIO

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from sklearn.manifold import TSNE
import imageio.v2 as imageio

warnings.filterwarnings('ignore')
ROOT      = Path(__file__).resolve().parent
CRASH_DIR = ROOT.parent / 'CRASH'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_DIR))
from src.models_gru  import LFCRASH_CBM_GRU
from src.data_loader import DADDataset, CrashDataset, A3DDataset

# ── Publication color palette (inspired by Nature / CVPR style) ─────────────
# Clean white background for paper figures
PAPER_BG    = '#FFFFFF'
PAPER_TEXT  = '#1A1A1A'
PAPER_GRID  = '#E8E8E8'
RED         = '#D62728'   # accident
BLUE        = '#1F77B4'   # normal / safe
ORANGE      = '#FF7F0E'   # highlight / warning
GREEN       = '#2CA02C'   # positive metric
PURPLE      = '#9467BD'
GRAY        = '#7F7F7F'
LIGHT_RED   = '#FFCCCC'
LIGHT_BLUE  = '#CCE5FF'

# Heatmap: white → yellow → red  (clean for paper)
CMAP_ACT = LinearSegmentedColormap.from_list(
    'act', ['#FFFFFF', '#FFF3CD', '#FF7F0E', '#D62728'])
# For overlay on dark frames
CMAP_OVERLAY = LinearSegmentedColormap.from_list(
    'overlay', [(0,'#00000000'),(0.3,'#FF7F0E88'),(1.0,'#D62728EE')])

FONT_TITLE  = dict(fontsize=13, fontweight='bold', color=PAPER_TEXT)
FONT_LABEL  = dict(fontsize=11, color=PAPER_TEXT)
FONT_TICK   = dict(fontsize=9,  color=PAPER_TEXT)
FONT_ANNOT  = dict(fontsize=8,  color=PAPER_TEXT)

plt.rcParams.update({
    'font.family'       : 'DejaVu Serif',
    'font.size'         : 10,
    'axes.facecolor'    : PAPER_BG,
    'figure.facecolor'  : PAPER_BG,
    'axes.edgecolor'    : '#BBBBBB',
    'axes.labelcolor'   : PAPER_TEXT,
    'xtick.color'       : PAPER_TEXT,
    'ytick.color'       : PAPER_TEXT,
    'text.color'        : PAPER_TEXT,
    'grid.color'        : PAPER_GRID,
    'grid.linestyle'    : '-',
    'grid.alpha'        : 0.7,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'legend.framealpha' : 0.9,
    'legend.edgecolor'  : '#CCCCCC',
})

DS_META = {
    'crash': dict(cls=CrashDataset, feature='vgg16', x_dim=4096, n_obj=19,
        n_frames=50, fps=10.0, phase='test',
        data_path=str(CRASH_DIR/'data'/'crash'),
        video_dir=str(CRASH_DIR/'data'/'crash'/'videos'/'Crash-1500'),
        normal_dir=str(CRASH_DIR/'data'/'crash'/'videos'/'Normal'),
        anno_file=str(CRASH_DIR/'data'/'crash'/'videos'/'Crash-1500.txt'),
        h_dim=256, lambda_align=1e-4, lambda_sparse=1e-3, lambda_recon=1e-3),
    'dad': dict(cls=DADDataset, feature='vgg16', x_dim=4096, n_obj=19,
        n_frames=100, fps=20.0, phase='testing',
        data_path=str(CRASH_DIR/'data'/'dad'),
        video_dir=str(CRASH_DIR/'data'/'dad'/'video'/'testing'),
        h_dim=256, lambda_align=1e-4, lambda_sparse=5e-4, lambda_recon=5e-3),
    'a3d': dict(cls=A3DDataset, feature='vgg16', x_dim=4096, n_obj=19,
        n_frames=100, fps=20.0, phase='test',
        data_path=str(CRASH_DIR/'data'/'a3d'),
        video_dir=str(CRASH_DIR/'data'/'a3d'/'videos'),
        h_dim=256, lambda_align=5e-4, lambda_sparse=3e-3, lambda_recon=1e-2),
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_concepts(path: str) -> List[str]:
    lines = [l.strip() for l in open(path) if l.strip()]
    cleaned = []
    for s in lines:
        for pre in ('A photo of a ','A photo of ','Photo of a ','Photo of '):
            if s.lower().startswith(pre.lower()):
                s = s[len(pre):]; s = s[0].upper()+s[1:]; break
        cleaned.append(s.rstrip('.'))
    return cleaned

def _collate(batch):
    xs,ys,toas = zip(*batch)
    xs = torch.from_numpy(np.stack(xs)).float()
    ys = torch.from_numpy(np.stack(ys)).float()
    tf = []
    for t in toas:
        if isinstance(t,(list,tuple,np.ndarray)): tf.append(float(t[0]))
        elif isinstance(t,torch.Tensor): tf.append(t.item() if t.numel()==1 else t[0].item())
        else: tf.append(float(t))
    return xs, ys, torch.tensor(tf, dtype=torch.float32)

def load_model(ckpt_path: str, device: str) -> LFCRASH_CBM_GRU:
    ck = torch.load(ckpt_path, map_location='cpu'); args = ck['args']; ds = args['dataset']
    m = DS_META[ds]
    model = LFCRASH_CBM_GRU(
        x_dim=m['x_dim'], h_dim=args.get('h_dim',m['h_dim']),
        z_dim=args.get('z_dim',128), n_layers=2,
        n_obj=m['n_obj'], n_frames=m['n_frames'], fps=m['fps'],
        num_concepts=args.get('num_concepts',837),
        lambda_align=args.get('lambda_align',m['lambda_align']),
        lambda_sparse=args.get('lambda_sparse',m['lambda_sparse']),
        lambda_recon=args.get('lambda_recon',m['lambda_recon']),
        use_cbm=not args.get('no_cbm',False), device=device)
    model.load_state_dict(ck['state_dict'], strict=False)
    model.to(device).eval()
    return model

@torch.no_grad()
def extract_one_sample(model, x_np, device):
    """Extract concept acts + probs for a single sample. x_np: (T,N+1,D)"""
    x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)  # (1,T,N+1,D)
    B,T = 1, x.shape[1]
    h = torch.zeros(2,B,model.h_dim,device=device)
    acts_buf, probs_buf, hh = [], [], []
    for t in range(T):
        frame=x[:,t]; feats=model.phi_x(frame)
        img_emb=feats[:,0]; obj_emb=feats[:,1:]
        c_act,c_embed=(model.cbm(img_emb) if model.use_cbm
                       else (img_emb.new_zeros(B,model.cbm.num_concepts),img_emb))
        obj_vec=model.ofa(obj_emb,h).squeeze(1)
        fi=model.fft_in(img_emb) if model.fft_in is not None else img_emb
        fv=model.fft_out(model.fft_block(fi.unsqueeze(-1)).mean(1))
        out_t,h=model.gru(torch.cat([obj_vec,c_embed,fv],1).unsqueeze(1),h)
        acts_buf.append(c_act.cpu().numpy()[0])   # (C,)
        probs_buf.append(F.softmax(out_t,-1)[0,1].item())
        hh.append(h.detach())
        if len(hh)>=10 and (t+1)%10==0: h=model._apply_rsd(hh[-10:])
    return np.stack(acts_buf,0), np.array(probs_buf)  # (T,C), (T,)

@torch.no_grad()
def extract_all(model, dataset, device, max_samples=200):
    from torch.utils.data import DataLoader
    loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0,collate_fn=_collate)
    ba,bp,bl,bt=[],[],[],[]
    for x,y,toa in loader:
        if sum(len(a) for a in ba)>=max_samples: break
        x=x.to(device); B,T=x.shape[0],x.shape[1]
        h=torch.zeros(2,B,model.h_dim,device=device)
        ab,pb,hh=[],[],[]
        for t in range(T):
            frame=x[:,t]; feats=model.phi_x(frame)
            img_emb=feats[:,0]; obj_emb=feats[:,1:]
            c_act,c_embed=(model.cbm(img_emb) if model.use_cbm
                           else (img_emb.new_zeros(B,model.cbm.num_concepts),img_emb))
            obj_vec=model.ofa(obj_emb,h).squeeze(1)
            fi=model.fft_in(img_emb) if model.fft_in is not None else img_emb
            fv=model.fft_out(model.fft_block(fi.unsqueeze(-1)).mean(1))
            out_t,h=model.gru(torch.cat([obj_vec,c_embed,fv],1).unsqueeze(1),h)
            ab.append(c_act.cpu()); pb.append(F.softmax(out_t,-1)[:,1].cpu())
            hh.append(h.detach())
            if len(hh)>=10 and (t+1)%10==0: h=model._apply_rsd(hh[-10:])
        ba.append(torch.stack(ab,1).numpy()); bp.append(torch.stack(pb,1).numpy())
        bl.append(y[:,1].numpy()); bt.append(toa.numpy())
    return (np.concatenate(ba,0)[:max_samples], np.concatenate(bp,0)[:max_samples],
            np.concatenate(bl,0)[:max_samples], np.concatenate(bt,0)[:max_samples])

def get_video_frames(video_path: str, n_frames: int, target_w=160, target_h=90) -> Optional[np.ndarray]:
    """Extract evenly-spaced frames. Returns (T, H, W, 3) RGB or None."""
    if not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0: cap.release(); return None
    indices = np.linspace(0, total-1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if not ret: fr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        fr = cv2.resize(fr, (target_w, target_h))
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, 0)  # (T, H, W, 3)

def read_crash_anno(anno_file: str):
    """Returns dict: video_id -> (toa_frame, weather, time_of_day)"""
    anno = {}
    for line in open(anno_file):
        parts = line.strip().split(',')
        if len(parts) < 6: continue
        vid = parts[0].strip()
        labels = list(map(int, re.findall(r'\d+', parts[1])))
        toa = next((i for i,l in enumerate(labels) if l==1), len(labels))
        time_of_day = parts[4].strip() if len(parts)>4 else ''
        weather = parts[5].strip() if len(parts)>5 else ''
        anno[vid] = dict(toa=toa, time=time_of_day, weather=weather)
    return anno

# ─────────────────────────────────────────────────────────────────────────────
# PUB FIG 1: Hero Case-Study Figure
# Video frames + prediction curve + concept heatmap + concept bar
# ─────────────────────────────────────────────────────────────────────────────
def pub_fig1_casestudy(model, dataset, cnames, fps, video_dir, out,
                       anno=None, ds_name='crash', device='cuda'):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    best = None
    for i,(x,y,toa) in enumerate(loader):
        if i > 80: break
        if y[0,1].item() < 0.5: continue
        acts_s, probs_s = extract_one_sample(model, x[0].numpy(), device)
        peak = probs_s.max()
        if best is None or peak > best['peak']:
            best = dict(idx=i, acts=acts_s, probs=probs_s,
                        toa=int(toa[0].item()), peak=peak)
        if best['peak'] > 0.92: break
    if best is None: print('  [!] No positive sample'); return

    acts, probs, toa_f = best['acts'], best['probs'], best['toa']
    T, C = acts.shape
    t_ax = np.arange(T) / fps
    vid_id = f"{best['idx']+1:06d}"
    # For datasets where video filenames don't match sequential index,
    # use sorted file list from video_dir
    vid_path = None
    if os.path.isdir(video_dir):
        vid_files = sorted([f for f in os.listdir(video_dir)
                            if f.endswith(('.mp4','.avi','.MP4'))])
        if best['idx'] < len(vid_files):
            vid_path = os.path.join(video_dir, vid_files[best['idx']])
            vid_id = vid_files[best['idx']].replace('.mp4','').replace('.avi','')
        else:
            vid_path = os.path.join(video_dir, f'{vid_id}.mp4')
    else:
        vid_path = os.path.join(video_dir, f'{vid_id}.mp4')
    SHOW_N = 8
    frames = get_video_frames(vid_path, SHOW_N, target_w=200, target_h=112)
    has_video = frames is not None

    peak_t = min(toa_f, T-1)
    pre_acts = acts[max(0,peak_t-5):peak_t+1].mean(0)
    top_k = 8
    # Use discriminability (accident vs normal) not raw activation
    ma_all = acts.mean(1)  # reuse dataset-level if passed, else approximate
    # For single sample: rank by deviation from temporal mean (most dynamic concepts)
    temporal_std = acts.std(0)   # (C,) - most time-varying concepts
    disc_proxy = pre_acts * temporal_std  # high at crash AND temporally dynamic
    top_idx = np.argsort(disc_proxy)[::-1][:top_k]
    short_c = [c[:52]+'…' if len(c)>52 else c for c in [cnames[i] for i in top_idx]]

    fig = plt.figure(figsize=(16, 12), facecolor=PAPER_BG)
    gs = gridspec.GridSpec(4, 1, figure=fig,
                           height_ratios=[2.0,1.8,1.8,2.4], hspace=0.50)

    # Row 0: frames strip
    ax0 = fig.add_subplot(gs[0]); ax0.set_axis_off()
    if has_video:
        strip = np.concatenate(list(frames), axis=1)
        ax0.imshow(strip)
        W_f = strip.shape[1] / SHOW_N
        fi_arr = np.linspace(0, T-1, SHOW_N, dtype=int)
        crash_x = (toa_f / T) * strip.shape[1]
        ax0.axvline(crash_x, color=RED, lw=2.5, ls='--', alpha=0.9)
        for j,fi in enumerate(fi_arr):
            col = RED if abs(fi-toa_f)<=3 else '#555555'
            ax0.text((j+0.5)*W_f, strip.shape[0]+3, f'{fi/fps:.1f}s',
                     ha='center', va='top', fontsize=8, color=col)
        meta = ''
        if anno and vid_id in anno:
            meta = f"  |  {anno[vid_id]['time']} / {anno[vid_id]['weather']}"
        ax0.set_title(f'Input Frames  (#{vid_id}{meta}  —  Crash onset @ {toa_f/fps:.1f}s)',
                      **FONT_TITLE, pad=7)
    else:
        ax0.text(0.5,0.5,f'Video #{vid_id}',ha='center',va='center',
                 fontsize=12,color=GRAY)
        ax0.set_title('Input Video Frames', **FONT_TITLE)

    # Row 1: prediction curve
    ax1 = fig.add_subplot(gs[1])
    ax1.fill_between(t_ax, probs, alpha=0.15, color=RED)
    ax1.plot(t_ax, probs, color=RED, lw=2.5, label='P(accident)', zorder=3)
    ax1.axhline(0.5, color=GRAY, lw=1.2, ls='--', label='Decision boundary')
    if toa_f < T:
        ax1.axvline(toa_f/fps, color=ORANGE, lw=2.0, ls=':', zorder=4,
                    label=f'Crash onset @ {toa_f/fps:.1f}s')
    ax1.set_ylim(0, 1.08); ax1.set_xlim(0, t_ax[-1])
    ax1.set_ylabel('P(accident)', **FONT_LABEL)
    ax1.set_xlabel('Time (s)', **FONT_LABEL)
    ax1.legend(fontsize=9, loc='upper left', ncol=3)
    ax1.set_title('Accident Prediction Confidence', **FONT_TITLE)
    ax1.grid(True)

    # Row 2: concept heatmap — normalize each concept independently for visibility
    ax2 = fig.add_subplot(gs[2])
    heat_raw = acts[:, top_idx].T   # (K, T)
    # Per-row normalization: each concept scaled to [0,1]
    row_min = heat_raw.min(1, keepdims=True)
    row_max = heat_raw.max(1, keepdims=True)
    heat = (heat_raw - row_min) / (row_max - row_min + 1e-8)
    im = ax2.imshow(heat, aspect='auto', cmap=CMAP_ACT,
                    extent=[0, t_ax[-1], top_k-0.5, -0.5],
                    vmin=0, vmax=1.0)
    ax2.set_yticks(range(top_k)); ax2.set_yticklabels(short_c, fontsize=8)
    if toa_f < T: ax2.axvline(toa_f/fps, color=ORANGE, lw=2.0, ls=':', zorder=5)
    ax2.set_xlabel('Time (s)', **FONT_LABEL)
    ax2.set_title('Top Concept Activations over Time', **FONT_TITLE)
    plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.01, label='Normalized Activation')

    # Row 3: concept bar at crash moment
    ax3 = fig.add_subplot(gs[3])
    vals = pre_acts[top_idx]
    mu, sg = vals.mean(), vals.std()
    col_bars = [RED if v>mu+sg else ORANGE if v>mu else BLUE for v in vals]
    bars = ax3.barh(range(top_k), vals, color=col_bars, alpha=0.85,
                    height=0.6, edgecolor='white', lw=0.5)
    ax3.set_yticks(range(top_k)); ax3.set_yticklabels(short_c, fontsize=8.5)
    ax3.invert_yaxis()
    for bar,v in zip(bars,vals):
        ax3.text(v+0.002, bar.get_y()+bar.get_height()/2,
                 f'{v:.3f}', va='center', fontsize=8)
    ax3.set_xlabel('Mean Activation (pre-crash window)', **FONT_LABEL)
    ax3.set_title('Safety-Critical Concepts Active at Crash Moment', **FONT_TITLE)
    ax3.grid(True, axis='x')
    from matplotlib.patches import Patch
    ax3.legend(handles=[Patch(color=RED,label='High'),
                        Patch(color=ORANGE,label='Medium'),
                        Patch(color=BLUE,label='Low')],
               fontsize=8, loc='lower right')

    plt.suptitle(f'CG-CRASH: Interpretable Accident Prediction via Concept Bottleneck [{ds_name.upper()}]',
                 fontsize=14, fontweight='bold', color=PAPER_TEXT, y=1.01)
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=PAPER_BG)
    plt.close(fig)
    print(f'  [✓] {Path(out).name}')

# ─────────────────────────────────────────────────────────────────────────────
# PUB FIG 2: Semantic Concept Importance
# ─────────────────────────────────────────────────────────────────────────────
def pub_fig2_concepts(acts, labels, cnames, out, top_k=15, ds_name='crash'):
    import textwrap
    ma = acts.mean(1)
    pos = ma[labels==1]; neg = ma[labels==0]
    if len(pos)==0: return
    pm=pos.mean(0); nm=neg.mean(0) if len(neg) else np.zeros_like(pm)
    # Use absolute difference normalized by pooled std for discriminability
    pooled_std = np.sqrt((pos.var(0)+neg.var(0))/2+1e-8) if len(neg) else pos.std(0)+1e-8
    disc = np.abs(pm-nm) / pooled_std
    top_idx=np.argsort(disc)[::-1][:top_k]
    wrapped=['\n'.join(textwrap.wrap(cnames[i],38)) for i in top_idx]
    pv=pm[top_idx]; nv=nm[top_idx]; dv=disc[top_idx]
    y=np.arange(top_k); bh=0.38
    fig,axes=plt.subplots(1,2,figsize=(16,top_k*0.55+2),facecolor=PAPER_BG,
        gridspec_kw={'width_ratios':[2.8,1]})
    ax=axes[0]
    ax.barh(y+bh/2,pv,height=bh,color=RED, alpha=0.80,label='Accident')
    ax.barh(y-bh/2,nv,height=bh,color=BLUE,alpha=0.80,label='Normal')
    ax.set_yticks(y); ax.set_yticklabels(wrapped,fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Concept Activation',**FONT_LABEL)
    ax.set_title(f'Top-{top_k} Safety-Critical Concepts [{ds_name.upper()}]',**FONT_TITLE)
    ax.legend(fontsize=10,loc='lower right'); ax.grid(True,axis='x')
    ax2=axes[1]
    norm=plt.Normalize(dv.min(),dv.max())
    cols=[plt.cm.RdYlGn_r(norm(v)) for v in dv]
    ax2.barh(y,dv,color=cols,alpha=0.90,height=0.55,edgecolor='white',lw=0.4)
    ax2.set_yticks([]); ax2.invert_yaxis()
    ax2.set_xlabel('Discriminability Score',**FONT_LABEL)
    ax2.set_title('Score',**FONT_TITLE)
    for i,v in enumerate(dv): ax2.text(v+dv.max()*0.02,i,f'{v:.2f}',va='center',fontsize=8.5)
    ax2.grid(True,axis='x')
    sm=plt.cm.ScalarMappable(cmap='RdYlGn_r',norm=norm)
    plt.colorbar(sm,ax=ax2,shrink=0.6,label='Low→High')
    plt.suptitle('Concept-Level Interpretability Analysis',fontsize=14,
        fontweight='bold',color=PAPER_TEXT,y=1.01)
    plt.tight_layout()
    fig.savefig(out,dpi=200,bbox_inches='tight',facecolor=PAPER_BG)
    plt.close(fig); print(f'  [✓] {Path(out).name}')

# ─────────────────────────────────────────────────────────────────────────────
# PUB FIG 3: Multi-sample timeline (frame strip + prob + heatmap)
# ─────────────────────────────────────────────────────────────────────────────
def pub_fig3_timeline(acts, probs, labels, toas, cnames, fps, video_dir,
                      out, ds_name='crash', n_cases=3):
    pos_idx=np.where(labels==1)[0]
    if len(pos_idx)==0: return
    chosen=pos_idx[np.argsort(probs[pos_idx].max(1))[::-1][:n_cases]]
    top_k=6; T=acts.shape[1]; t_ax=np.arange(T)/fps
    ma=acts.mean(1)
    pos_m=ma[labels==1].mean(0)
    neg_m=ma[labels==0].mean(0) if (labels==0).any() else np.zeros_like(pos_m)
    # Cohen's d style discriminability
    pos_std=ma[labels==1].std(0) if (labels==1).any() else np.ones_like(pos_m)
    neg_std=ma[labels==0].std(0) if (labels==0).any() else np.ones_like(pos_m)
    pooled=np.sqrt((pos_std**2+neg_std**2)/2+1e-8)
    disc=np.abs(pos_m-neg_m)/pooled
    gtop=np.argsort(disc)[::-1][:top_k]
    short_c=[c[:33]+'…' if len(c)>33 else c for c in [cnames[i] for i in gtop]]
    fig,ag=plt.subplots(n_cases,3,figsize=(18,3.8*n_cases),facecolor=PAPER_BG,
        gridspec_kw={'width_ratios':[1.8,2.2,1.8],'wspace':0.35,'hspace':0.55})
    if n_cases==1: ag=[ag]
    # Pre-build sorted video file list for index-based lookup
    vid_files_sorted = []
    if os.path.isdir(video_dir):
        vid_files_sorted = sorted([f for f in os.listdir(video_dir)
                                   if f.endswith(('.mp4','.avi','.MP4'))])
    for row,si in enumerate(chosen):
        vid_id=f'{si+1:06d}'
        if si < len(vid_files_sorted):
            vpath = os.path.join(video_dir, vid_files_sorted[si])
            vid_id = vid_files_sorted[si].replace('.mp4','').replace('.avi','')
        else:
            vpath = os.path.join(video_dir, f'{vid_id}.mp4')
        frames=get_video_frames(vpath, 5, target_w=180, target_h=100)
        toa_f=int(toas[si])
        # frames
        ax_f=ag[row][0]
        if frames is not None:
            strip=np.concatenate(list(frames),axis=1)
            ax_f.imshow(strip); ax_f.set_axis_off()
            fi_arr=np.linspace(0,T-1,5,dtype=int); W_f=strip.shape[1]/5
            for j,fi in enumerate(fi_arr):
                col=RED if abs(fi-toa_f)<=4 else '#555555'
                ax_f.text((j+0.5)*W_f,strip.shape[0]+2,f'{fi/fps:.1f}s',
                    ha='center',va='top',fontsize=7.5,color=col)
        else:
            ax_f.text(0.5,0.5,f'#{vid_id}',ha='center',va='center',color=GRAY)
            ax_f.set_axis_off()
        ax_f.set_title(f'Sample #{si+1}  TOA={toa_f/fps:.1f}s',
            fontsize=9,fontweight='bold',color=PAPER_TEXT,pad=4)
        # prob
        ax_p=ag[row][1]
        ax_p.fill_between(t_ax,probs[si],alpha=0.15,color=RED)
        ax_p.plot(t_ax,probs[si],color=RED,lw=2.2)
        ax_p.axhline(0.5,color=GRAY,lw=1.0,ls='--')
        if toa_f<T: ax_p.axvline(toa_f/fps,color=ORANGE,lw=1.8,ls=':')
        ax_p.set_ylim(0,1.08); ax_p.set_xlim(0,t_ax[-1])
        ax_p.set_ylabel('P(accident)',fontsize=9); ax_p.set_xlabel('Time (s)',fontsize=9)
        ax_p.grid(True)
        if row==0: ax_p.set_title('Prediction Confidence',**FONT_TITLE)
        # heatmap — per-concept normalization
        ax_h=ag[row][2]
        heat_raw=acts[si][:,gtop].T
        rmin=heat_raw.min(1,keepdims=True); rmax=heat_raw.max(1,keepdims=True)
        heat=(heat_raw-rmin)/(rmax-rmin+1e-8)
        im=ax_h.imshow(heat,aspect='auto',cmap=CMAP_ACT,
            extent=[0,t_ax[-1],top_k-0.5,-0.5],vmin=0,vmax=1.0)
        ax_h.set_yticks(range(top_k)); ax_h.set_yticklabels(short_c,fontsize=7.5)
        if toa_f<T: ax_h.axvline(toa_f/fps,color=ORANGE,lw=1.8,ls=':')
        ax_h.set_xlabel('Time (s)',fontsize=9)
        if row==0: ax_h.set_title('Concept Activations',**FONT_TITLE)
        plt.colorbar(im,ax=ax_h,shrink=0.7)
    plt.suptitle(f'CG-CRASH: Prediction & Concept Analysis [{ds_name.upper()}]',
        fontsize=14,fontweight='bold',y=1.01)
    fig.savefig(out,dpi=200,bbox_inches='tight',facecolor=PAPER_BG)
    plt.close(fig); print(f'  [✓] {Path(out).name}')

# ─────────────────────────────────────────────────────────────────────────────
# PUB FIG 4: Ablation study (clean publication bar chart)
# ─────────────────────────────────────────────────────────────────────────────
def pub_fig4_ablation(results_dir, out):
    base=Path(results_dir)
    datasets=['crash','a3d','dad']
    variants=[('full','Full Model'),('no_cbm','w/o CBM'),
               ('no_align','w/o Align'),('no_sparse','w/o Sparse'),
               ('no_recon','w/o Recon')]
    metrics=[('AP','AP'),('mTTA','mTTA (s)')]
    ds_colors={'crash':RED,'a3d':BLUE,'dad':GREEN}
    n_var=len(variants); n_met=len(metrics)
    fig,axes=plt.subplots(1,n_met,figsize=(14,5),facecolor=PAPER_BG)
    x=np.arange(n_var); bw=0.22
    for mi,(mk,ml) in enumerate(metrics):
        ax=axes[mi]
        for di,ds in enumerate(datasets):
            vals=[]
            for vk,_ in variants:
                f=base/f'{ds}_{vk}'/'results.json'
                if f.exists():
                    d=json.load(open(f)); vals.append(d.get(mk,0))
                else:
                    vals.append(0)
            offset=(di-1)*bw
            bars=ax.bar(x+offset,vals,width=bw,color=ds_colors[ds],
                        alpha=0.85,label=ds.upper(),edgecolor='white',lw=0.5)
            # value labels
            for bar,v in zip(bars,vals):
                if v>0:
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()+0.005*ax.get_ylim()[1] if mi==0
                            else bar.get_height()+0.02,
                            f'{v:.3f}' if mi==0 else f'{v:.2f}',
                            ha='center',va='bottom',fontsize=6.5,rotation=45)
        ax.set_xticks(x)
        ax.set_xticklabels([v[1] for v in variants],fontsize=9,rotation=15,ha='right')
        ax.set_ylabel(ml,**FONT_LABEL)
        ax.set_title(f'Ablation: {ml}',**FONT_TITLE)
        ax.legend(fontsize=9)
        ax.grid(True,axis='y')
        # Highlight full model
        ax.axvspan(-0.5,0.5,color='#F0F0F0',zorder=0,alpha=0.8)
        ax.text(0,ax.get_ylim()[0],'Baseline',ha='center',fontsize=7.5,
                color=GRAY,style='italic',va='bottom')
    plt.suptitle('Ablation Study: Impact of Each CG-CRASH Component',
        fontsize=14,fontweight='bold',color=PAPER_TEXT,y=1.02)
    plt.tight_layout()
    fig.savefig(out,dpi=200,bbox_inches='tight',facecolor=PAPER_BG)
    plt.close(fig); print(f'  [✓] {Path(out).name}')

# ─────────────────────────────────────────────────────────────────────────────
# PUB FIG 5: t-SNE (publication style)
# ─────────────────────────────────────────────────────────────────────────────
def pub_fig5_tsne(acts, labels, cnames, out, ds_name='crash', max_pts=400):
    ma=acts.mean(1)
    N=min(len(ma),max_pts)
    idx=np.random.choice(len(ma),N,replace=False)
    X=ma[idx]; y=labels[idx]
    Xs=(X-X.mean(0))/(X.std(0)+1e-8)
    emb=TSNE(n_components=2,perplexity=min(30,N//3),
             random_state=42,n_iter=1000).fit_transform(Xs)
    # Top concept weights for coloring overlay
    pos_m=ma[labels==1].mean(0) if (labels==1).any() else np.zeros(ma.shape[1])
    neg_m=ma[labels==0].mean(0) if (labels==0).any() else np.zeros(ma.shape[1])
    disc=np.abs(pos_m-neg_m)/(ma.std(0)+1e-8)
    top3=np.argsort(disc)[::-1][:3]
    fig,axes=plt.subplots(1,2,figsize=(14,6),facecolor=PAPER_BG)
    # Left: accident vs normal
    ax=axes[0]
    for lv,color,marker,lab,sz in [(0,BLUE,'o','Normal',40),(1,RED,'*','Accident',70)]:
        m=y==lv
        ax.scatter(emb[m,0],emb[m,1],c=color,marker=marker,s=sz,
                   alpha=0.70,label=lab,edgecolors='none',zorder=3 if lv==1 else 2)
    ax.set_title('Concept Space: Accident vs Normal',**FONT_TITLE)
    ax.legend(fontsize=11,markerscale=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('t-SNE dim 1',**FONT_LABEL); ax.set_ylabel('t-SNE dim 2',**FONT_LABEL)
    ax.grid(True)
    # Right: color by top concept activation
    ax2=axes[1]
    top_c_vals=X[:,top3[0]]
    sc=ax2.scatter(emb[:,0],emb[:,1],c=top_c_vals,cmap='RdYlBu_r',
                   s=45,alpha=0.75,edgecolors='none')
    plt.colorbar(sc,ax=ax2,label='Activation')
    ax2.set_title(f'Colored by: {cnames[top3[0]][:45]}',**FONT_TITLE,wrap=True)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_xlabel('t-SNE dim 1',**FONT_LABEL); ax2.set_ylabel('t-SNE dim 2',**FONT_LABEL)
    ax2.grid(True)
    plt.suptitle(f't-SNE Projection of Concept Activation Space [{ds_name.upper()}]',
        fontsize=14,fontweight='bold',y=1.01)
    plt.tight_layout()
    fig.savefig(out,dpi=200,bbox_inches='tight',facecolor=PAPER_BG)
    plt.close(fig); print(f'  [✓] {Path(out).name}')

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(ds_name, ckpt_dir, out_dir, concept_file, device, max_samples=200):
    ckpt = Path(ckpt_dir)/f'{ds_name}_full'/'best_model.pth'
    if not ckpt.exists():
        print(f'  [!] Missing: {ckpt}'); return
    print(f'\n=== {ds_name.upper()} ===')
    model = load_model(str(ckpt), device)
    m = DS_META[ds_name]
    dataset = m['cls'](m['data_path'], m['feature'], m['phase'])
    cnames = load_concepts(concept_file)
    out = Path(out_dir)/ds_name; out.mkdir(parents=True, exist_ok=True)

    # Load cached activations if available, else extract
    cache = Path('output/visualizations')/ds_name/'activations.npz'
    if cache.exists():
        print('  Using cached activations...')
        c = np.load(str(cache))
        acts,probs,labels,toas = c['acts'],c['probs'],c['labels'],c['toas']
    else:
        print('  Extracting activations...')
        acts,probs,labels,toas = extract_all(model,dataset,device,max_samples)
        np.savez_compressed(str(out/'activations.npz'),
            acts=acts,probs=probs,labels=labels,toas=toas)

    fps = m['fps']
    video_dir = m.get('video_dir','')
    anno = None
    if ds_name=='crash' and 'anno_file' in m:
        try: anno = read_crash_anno(m['anno_file'])
        except: pass

    print('  Generating publication figures...')
    pub_fig1_casestudy(model, dataset, cnames, fps, video_dir,
                       str(out/'pub_fig1_casestudy.png'),
                       anno=anno, ds_name=ds_name, device=device)
    pub_fig2_concepts(acts, labels, cnames,
                      str(out/'pub_fig2_concepts.png'), top_k=15, ds_name=ds_name)
    pub_fig3_timeline(acts, probs, labels, toas, cnames, fps, video_dir,
                      str(out/'pub_fig3_timeline.png'), ds_name=ds_name, n_cases=3)
    pub_fig5_tsne(acts, labels, cnames,
                  str(out/'pub_fig5_tsne.png'), ds_name=ds_name)
    print(f'  Done -> {out}/')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='crash', choices=['crash','dad','a3d','all'])
    ap.add_argument('--gpu',     default='0')
    ap.add_argument('--ckpt_dir',     default='output/v3_final')
    ap.add_argument('--out_dir',      default='output/pub_figures')
    ap.add_argument('--concept_file', default='/data/sony/LFCRASH/000_all_concept_set.txt')
    ap.add_argument('--max_samples',  type=int, default=200)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Fig 4: ablation (all datasets, run once)
    out_root = ROOT/args.out_dir; out_root.mkdir(parents=True, exist_ok=True)
    pub_fig4_ablation(str(ROOT/args.ckpt_dir), str(out_root/'pub_fig4_ablation.png'))

    datasets = ['crash','dad','a3d'] if args.dataset=='all' else [args.dataset]
    for ds in datasets:
        run(ds, str(ROOT/args.ckpt_dir), str(out_root),
            args.concept_file, device, args.max_samples)

    print('\n=== ALL PUBLICATION FIGURES DONE ===')
    print(f'Output: {out_root}/')


if __name__ == '__main__':
    main()


