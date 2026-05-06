#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_v3.py  -  Publication + Video Animation Visualization Suite v3
Outputs per dataset:
  best_case_study.png     Hero figure with large frames
  concept_importance.png  Named concept discriminability
  multi_case.png          Top-3 best samples
  animation.mp4           Synchronized video animation
  interactive.html        Plotly HCI dashboard
"""
import os,sys,json,argparse,warnings,textwrap,re
from pathlib import Path
from typing import List,Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib import cm
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import imageio.v2 as imageio
import io

warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parent
CRASH_DIR=ROOT.parent/'CRASH'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(CRASH_DIR))
from src.models_gru import LFCRASH_CBM_GRU
from src.data_loader import DADDataset,CrashDataset,A3DDataset

# ── Palette (Nature / CVPR) ───────────────────────────────────────────────────
# ── Muted Morandi palette (paper-ready) ─────────────────────────────────
BG_PAGE  = '#FFFFFF'
BG_CARD  = '#FAFAFA'
BG_STRIP = '#F5F5F5'
W        = '#FFFFFF'
BK       = '#2A2A2A'
G1       = '#444444'
G2       = '#888888'
G3       = '#CCCCCC'
RED      = '#DF908A'   # accident / crash  — dusty rose
RED_L    = '#F7E5E4'   # accident fill
ORG      = '#EDBF97'   # warning / alert   — warm peach
GRN      = '#81C1C3'   # safe / normal     — muted teal
BLUE     = '#598EBB'   # concept / info    — steel blue
BLUE_L   = '#D6E8F5'   # concept fill
PRP      = '#A5A3C3'   # accent            — lavender grey

CMAP_ACT = LinearSegmentedColormap.from_list(
    'act', ['#FFFFFF','#ECD0CB','#E4A3A1','#DF908A','#C07070'])

plt.rcParams.update({
    'font.family'      : 'DejaVu Serif',
    'font.size'        : 13,
    'axes.facecolor'   : '#FAFAFA',
    'figure.facecolor' : '#FFFFFF',
    'axes.edgecolor'   : '#CCCCCC',
    'axes.labelcolor'  : '#444444',
    'xtick.color'      : '#444444',
    'ytick.color'      : '#444444',
    'text.color'       : '#2A2A2A',
    'grid.color'       : '#E8E8E8',
    'grid.linestyle'   : '-',
    'grid.alpha'       : 0.5,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'legend.framealpha': 0.95,
    'legend.edgecolor' : '#CCCCCC',
    'legend.facecolor' : '#FFFFFF',
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 12,
    'xtick.labelsize'  : 11,
    'ytick.labelsize'  : 11,
    'legend.fontsize'  : 11,
})

DS_META={
    'crash':dict(cls=CrashDataset,feature='vgg16',x_dim=4096,n_obj=19,
        n_frames=50,fps=10.0,phase='test',
        data_path=str(CRASH_DIR/'data'/'crash'),
        video_dir=str(CRASH_DIR/'data'/'crash'/'videos'/'Crash-1500'),
        anno_file=str(CRASH_DIR/'data'/'crash'/'videos'/'Crash-1500.txt'),
        h_dim=256,lambda_align=1e-4,lambda_sparse=1e-3,lambda_recon=1e-3),
    'dad':dict(cls=DADDataset,feature='vgg16',x_dim=4096,n_obj=19,
        n_frames=100,fps=20.0,phase='testing',
        data_path=str(CRASH_DIR/'data'/'dad'),
        video_dir=str(CRASH_DIR/'data'/'dad'/'video'/'testing'),
        h_dim=256,lambda_align=1e-4,lambda_sparse=5e-4,lambda_recon=5e-3),
    'a3d':dict(cls=A3DDataset,feature='vgg16',x_dim=4096,n_obj=19,
        n_frames=100,fps=20.0,phase='test',
        data_path=str(CRASH_DIR/'data'/'a3d'),
        video_dir=None,
        h_dim=256,lambda_align=5e-4,lambda_sparse=3e-3,lambda_recon=1e-2),
}

def load_concepts(path):
    out=[]
    for s in open(path):
        s=s.strip()
        if not s: continue
        for pre in ('A photo of a ','A photo of ','Photo of a ','Photo of '):
            if s.lower().startswith(pre.lower()):
                s=s[len(pre):]; s=s[0].upper()+s[1:]; break
        out.append(s.rstrip('.'))
    return out

def short(s,n=44): return s[:n]+('\u2026' if len(s)>n else '')
def wrapn(s,n=36): return '\n'.join(textwrap.wrap(s,n))

def _collate(batch):
    xs,ys,toas=zip(*batch)
    xs=torch.from_numpy(np.stack(xs)).float()
    ys=torch.from_numpy(np.stack(ys)).float()
    tf=[]
    for t in toas:
        if isinstance(t,(list,tuple,np.ndarray)): tf.append(float(t[0]))
        elif isinstance(t,torch.Tensor): tf.append(t.item() if t.numel()==1 else t[0].item())
        else: tf.append(float(t))
    return xs,ys,torch.tensor(tf,dtype=torch.float32)

# ── Model & extraction ───────────────────────────────────────────────────────
def load_model(ckpt,device):
    ck=torch.load(ckpt,map_location='cpu'); a=ck['args']; ds=a['dataset']; m=DS_META[ds]
    model=LFCRASH_CBM_GRU(x_dim=m['x_dim'],h_dim=a.get('h_dim',m['h_dim']),
        z_dim=a.get('z_dim',128),n_layers=2,n_obj=m['n_obj'],
        n_frames=m['n_frames'],fps=m['fps'],num_concepts=a.get('num_concepts',837),
        lambda_align=a.get('lambda_align',m['lambda_align']),
        lambda_sparse=a.get('lambda_sparse',m['lambda_sparse']),
        lambda_recon=a.get('lambda_recon',m['lambda_recon']),
        use_cbm=not a.get('no_cbm',False),device=device)
    model.load_state_dict(ck['state_dict'],strict=False)
    return model.to(device).eval()

@torch.no_grad()
def extract_all(model,dataset,device,max_s=300):
    from torch.utils.data import DataLoader
    loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0,collate_fn=_collate)
    ba,bp,bl,bt=[],[],[],[]
    for x,y,toa in loader:
        if sum(len(a) for a in ba)>=max_s: break
        x=x.to(device); B,T=x.shape[0],x.shape[1]
        h=torch.zeros(2,B,model.h_dim,device=device)
        ab,pb,hh=[],[],[]
        for t in range(T):
            frame=x[:,t]; feats=model.phi_x(frame)
            img_emb=feats[:,0]; obj_emb=feats[:,1:]
            c_act,c_embed=(model.cbm(img_emb) if model.use_cbm
                else (img_emb.new_zeros(B,model.cbm.num_concepts),img_emb))
            ov=model.ofa(obj_emb,h).squeeze(1)
            fi=model.fft_in(img_emb) if model.fft_in is not None else img_emb
            fv=model.fft_out(model.fft_block(fi.unsqueeze(-1)).mean(1))
            out_t,h=model.gru(torch.cat([ov,c_embed,fv],1).unsqueeze(1),h)
            ab.append(c_act.cpu()); pb.append(F.softmax(out_t,-1)[:,1].cpu())
            hh.append(h.detach())
            if len(hh)>=10 and (t+1)%10==0: h=model._apply_rsd(hh[-10:])
        ba.append(torch.stack(ab,1).numpy()); bp.append(torch.stack(pb,1).numpy())
        bl.append(y[:,1].numpy()); bt.append(toa.numpy())
    return (np.concatenate(ba,0)[:max_s],np.concatenate(bp,0)[:max_s],
            np.concatenate(bl,0)[:max_s],np.concatenate(bt,0)[:max_s])

# ── Smart sample selection ────────────────────────────────────────────────────
def select_best_samples(acts,probs,labels,toas,n=3):
    """Select samples with clear rising prediction curve:
    low initial prob, strong rise before crash, early warning."""
    pos=np.where(labels==1)[0]
    if len(pos)==0: return pos[:n]
    T=acts.shape[1]; results=[]
    for i in pos:
        toa_f=int(toas[i]); p=probs[i]
        p0=p[0]  # should be LOW
        p_toa=p[min(toa_f,T-1)]  # should be HIGH
        cross=np.where(p>0.5)[0]
        warn=(toa_f-cross[0]) if len(cross) and cross[0]<toa_f else 0
        rise=p_toa-p0
        # Penalize samples that start high (boring, no story)
        low_start_bonus=(1-p0)**2
        score=low_start_bonus*0.5 + rise*0.3 + min(warn/T,1)*0.2
        results.append((score,i))
    results.sort(reverse=True)
    return np.array([i for _,i in results[:n]])

# ── Video frame extraction ────────────────────────────────────────────────────
def get_frames(video_path,n,W=320,H=180,start_f=None,end_f=None,total_model_frames=None):
    """Extract n frames. start_f/end_f are model-space indices, mapped to video frames."""
    if not video_path or not os.path.exists(video_path): return None
    import signal
    def _timeout_handler(signum,frame): raise TimeoutError('video open timeout')
    try:
        signal.signal(signal.SIGALRM,_timeout_handler)
        signal.alarm(15)  # 15s timeout for video open (NFS can be slow)
        cap=cv2.VideoCapture(video_path)
        signal.alarm(0)
    except TimeoutError:
        return None
    total_vid=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_vid==0: cap.release(); return None
    ratio=(total_vid/total_model_frames) if total_model_frames else 1.0
    f0=int((start_f or 0)*ratio); f1=int(((end_f or total_vid/ratio)-1)*ratio)
    f0=max(0,min(f0,total_vid-1)); f1=max(f0,min(f1,total_vid-1))
    idx=np.linspace(f0,f1,n,dtype=int) if n>1 else np.array([f0])
    frames=[]
    for i in idx:
        try:
            signal.alarm(8)  # 8s per frame read
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(i))
            ret,fr=cap.read()
            signal.alarm(0)
        except TimeoutError:
            ret=False; fr=None
        if not ret or fr is None: fr=np.zeros((H,W,3),np.uint8)
        frames.append(cv2.cvtColor(cv2.resize(fr,(W,H)),cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)

def get_vpath(video_dir,sample_idx):
    if not video_dir or not os.path.isdir(video_dir): return None
    vf=sorted([f for f in os.listdir(video_dir) if f.endswith(('.mp4','.avi'))])
    return os.path.join(video_dir,vf[sample_idx]) if sample_idx<len(vf) else None

def read_anno(anno_file):
    anno={}
    for line in open(anno_file):
        p=line.strip().split(',')
        if len(p)<4: continue
        vid=p[0].strip()
        labels=list(map(int,re.findall(r'\d+',p[1])))
        toa=next((i for i,l in enumerate(labels) if l==1),len(labels))
        anno[vid]=dict(toa=toa,
            time=p[4].strip() if len(p)>4 else '',
            weather=p[5].strip() if len(p)>5 else '')
    return anno

# ── Concept discriminability ──────────────────────────────────────────────────
def top_disc(acts,labels,k):
    ma=acts.mean(1)
    pos=ma[labels==1]; neg=ma[labels==0]
    if len(pos)==0: return np.arange(k)
    pm=pos.mean(0); nm=neg.mean(0) if len(neg) else np.zeros_like(pm)
    ps=pos.std(0)+1e-8; ns=neg.std(0)+1e-8 if len(neg) else np.ones_like(pm)
    disc=np.abs(pm-nm)/np.sqrt((ps**2+ns**2)/2)
    idx=np.argsort(disc)[::-1][:k]
    return idx, disc[idx]

def top_delta_concepts(acts_si, toas_si, k=8):
    """Select concepts with largest Safe→Crash delta for a single sample.
    Safe window: first 30% of frames before alert.
    Crash window: last 8 frames before/at TOA.
    Returns (top_idx, safe_vals, crash_vals, delta_vals) sorted by delta descending.
    """
    T=acts_si.shape[0]; toa_f=int(toas_si)
    safe_end=max(1,int(toa_f*0.25))  # first 25% = truly safe window
    safe_vals=acts_si[:safe_end].mean(0)
    crash_start=max(0,toa_f-8)
    crash_vals=acts_si[crash_start:toa_f+1].mean(0)
    delta=crash_vals-safe_vals  # positive = rises before crash
    top_idx=np.argsort(delta)[::-1][:k]
    return top_idx, safe_vals[top_idx], crash_vals[top_idx], delta[top_idx]

def norm_rows(M):
    """Normalize each row to [0,1] for heatmap visibility."""
    mn=M.min(1,keepdims=True); mx=M.max(1,keepdims=True)
    return (M-mn)/(mx-mn+1e-8)

# ── Fig 1: Hero case study — Publication Quality ────────────────────────────
def fig_hero(si, acts, probs, toas, cnames, fps, video_dir, out, ds_name, anno=None):
    T=acts.shape[1]; t_ax=np.arange(T)/fps
    toa_f=int(toas[si]); p=probs[si]
    top_k=8
    top_idx,safe_v,crash_v,delta_v=top_delta_concepts(acts[si],toas[si],k=top_k)
    short_c=[short(cnames[i],44) for i in top_idx]

    vpath=get_vpath(video_dir,si)
    N_FRAMES=8
    frames=get_frames(vpath,N_FRAMES,W=360,H=202) if vpath else None
    has_v=frames is not None
    cross_idx=np.where(p>0.5)[0]
    warn_t=cross_idx[0]/fps if len(cross_idx) and cross_idx[0]<toa_f else t_ax[-1]

    # Layout: row0=frames, row1=prob curve, row2=heatmap+bar
    fig=plt.figure(figsize=(22,10),facecolor=W)
    gs=gridspec.GridSpec(3,1,figure=fig,
        height_ratios=[2.8,1.1,2.4],hspace=0.18)

    # ── Row 0: wide cinema strip ──────────────────────────────────────────────
    ax0=fig.add_subplot(gs[0]); ax0.set_axis_off()
    if has_v:
        pad=6; H_f,W_f_=frames[0].shape[:2]
        total_W=W_f_*N_FRAMES+pad*(N_FRAMES-1)
        canvas=np.ones((H_f,total_W,3),np.uint8)*230
        fi_arr=np.linspace(0,T-1,N_FRAMES,dtype=int)
        for j,fr in enumerate(frames):
            x0=j*(W_f_+pad)
            canvas[:,x0:x0+W_f_]=fr
        ax0.imshow(canvas)
        for j,fi in enumerate(fi_arr):
            xc=j*(W_f_+pad)+W_f_//2
            x0=j*(W_f_+pad)
            is_crash=toa_f<T and fi>=toa_f
            is_warn=not is_crash and p[fi]>0.5
            bc=RED if is_crash else ORG if is_warn else '#888888'
            bw=4.5 if is_crash else 2.5 if is_warn else 1.0
            ax0.add_patch(plt.Rectangle((x0-1,-1),W_f_+1,H_f+1,
                lw=bw,edgecolor=bc,facecolor='none',zorder=5))
            # Time label below
            ax0.text(xc,H_f+8,f'{fi/fps:.1f}s',ha='center',va='top',
                fontsize=9,color=bc)
            # Prob badge above
            ax0.text(xc,-8,f'P={p[fi]:.2f}',ha='center',va='bottom',
                fontsize=9.5,color=bc,fontweight='bold')
        ax0.set_xlim(-10,total_W+10); ax0.set_ylim(H_f+30,-30)
        warn_frames=(toa_f-cross_idx[0]) if len(cross_idx) and cross_idx[0]<toa_f else 0
        ax0.set_title(
            f'[{ds_name.upper()}]  Sample #{si+1}  ·  '
            f'P₀={p[0]:.3f} → Peak={p.max():.3f}  ·  '
            f'Crash @ {toa_f/fps:.1f}s  ·  Early warning: {warn_frames/fps:.1f}s',
            fontsize=11.5,fontweight='bold',color=BK,pad=16,loc='left')
        ax0.legend(handles=[
            plt.Line2D([0],[0],color=RED,lw=3,label='Crash'),
            plt.Line2D([0],[0],color=ORG,lw=2,label='Alert (P>0.5)'),
            plt.Line2D([0],[0],color='#888888',lw=1.5,label='Normal'),
        ],loc='upper right',fontsize=9,framealpha=0.9,ncol=3)
    else:
        ax0.set_facecolor('#F5F5F5')
        ax0.text(0.5,0.5,f'[{ds_name.upper()}]  No video — feature-based analysis',
            ha='center',va='center',fontsize=14,color=G2,
            transform=ax0.transAxes,style='italic')
        ax0.set_title(f'[{ds_name.upper()}]  Sample #{si+1}  ·  Crash @ {toa_f/fps:.1f}s',
            fontsize=12,fontweight='bold',pad=10)

    # ── Row 1: Prediction confidence ──────────────────────────────────────────
    ax1=fig.add_subplot(gs[1])
    ax1.fill_between(t_ax,p,alpha=0.18,color=GRN,where=t_ax<warn_t,interpolate=True)
    ax1.fill_between(t_ax,p,alpha=0.18,color=ORG,
        where=(t_ax>=warn_t)&(t_ax<toa_f/fps),interpolate=True)
    ax1.fill_between(t_ax,p,alpha=0.22,color=RED,where=t_ax>=toa_f/fps,interpolate=True)
    ax1.plot(t_ax,p,color='#1A1A2E',lw=2.2,zorder=3)
    ax1.axhline(0.5,color=G2,lw=1.0,ls='--',alpha=0.7)
    if toa_f<T:
        ax1.axvline(toa_f/fps,color=RED,lw=1.8,ls=':',zorder=4,
            label=f'Crash @ {toa_f/fps:.1f}s')
    if len(cross_idx) and cross_idx[0]<toa_f:
        ax1.axvline(cross_idx[0]/fps,color=ORG,lw=1.5,ls=':',
            label=f'Alert @ {cross_idx[0]/fps:.1f}s')
    ax1.set_ylim(-0.02,1.08); ax1.set_xlim(0,t_ax[-1])
    ax1.set_ylabel('P(accident)',fontsize=11)
    ax1.set_xlabel('Time (s)',fontsize=11)
    ax1.legend(fontsize=9.5,loc='upper left',ncol=3,framealpha=0.9)
    ax1.set_title('Accident Prediction Confidence',fontsize=12,fontweight='bold',pad=5)
    ax1.grid(True,alpha=0.3,color='#EEEEEE')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # ── Row 2: Heatmap (wider) + Concept bar (narrower, no overlap) ───────────
    gs2=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[2],
        width_ratios=[1.6,1.2],wspace=0.22)

    # Heatmap
    ax2=fig.add_subplot(gs2[0])
    heat=norm_rows(acts[si][:,top_idx].T)
    im=ax2.imshow(heat,aspect='auto',cmap=CMAP_ACT,
        extent=[0,t_ax[-1],top_k-0.5,-0.5],vmin=0,vmax=1,interpolation='bilinear')
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(short_c,fontsize=9.5)
    if toa_f<T:
        ax2.axvline(toa_f/fps,color='white',lw=2.0,ls=':',zorder=5)
        ax2.text(toa_f/fps+0.05,top_k-0.2,'Crash',color='white',fontsize=8,
            fontweight='bold',va='bottom')
    if len(cross_idx) and cross_idx[0]<toa_f:
        ax2.axvline(cross_idx[0]/fps,color=ORG,lw=1.5,ls=':',zorder=4)
    ax2.set_xlabel('Time (s)',fontsize=11)
    ax2.set_title('Concept Activation Timeline (top-8 by Safe→Crash Δ)',
        fontsize=11,fontweight='bold',pad=6)
    cb=plt.colorbar(im,ax=ax2,shrink=0.80,pad=0.01,aspect=20)
    cb.set_label('Norm. Activation',fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Concept bar — clean, no overlap
    ax3=fig.add_subplot(gs2[1])
    # Show both safe (background) and crash (foreground) values
    y=np.arange(top_k)
    bar_max=max(crash_v.max(),safe_v.max())*1.15+0.05
    ax3.barh(y,safe_v,height=0.55,color='#B5DCDF',alpha=0.75,
        edgecolor='white',lw=0.5,label='Safe baseline',zorder=2)
    crash_colors=[RED if v>safe_v[i]*1.3 else ORG if v>safe_v[i]*1.1
                  else BLUE for i,v in enumerate(crash_v)]
    ax3.barh(y,crash_v,height=0.55,color=crash_colors,alpha=0.82,
        edgecolor='white',lw=0.4,label='At crash',zorder=3)
    ax3.set_yticks(y)
    ax3.set_yticklabels(['']*len(y))  # hide — shown in ax2 heatmap already
    ax3.invert_yaxis()
    ax3.set_xlim(0,bar_max*1.35)  # extra room for delta labels
    # Delta labels — placed OUTSIDE bars at consistent position
    for i,(sv,cv) in enumerate(zip(safe_v,crash_v)):
        d=cv-sv
        dlabel=f'\u0394{d:+.2f}'
        ax3.text(bar_max*1.05,i,dlabel,ha='left',va='center',
            fontsize=8.5,fontweight='bold',
            color=RED if d>0.05 else GRN if d<-0.05 else G2,
            bbox=dict(boxstyle='round,pad=0.1',facecolor='white',
                      edgecolor='none',alpha=0.7))
    ax3.set_xlabel('Mean Activation',fontsize=10)
    ax3.set_title('Safe vs Crash\nConcept Activation',fontsize=11,fontweight='bold',pad=6)
    ax3.legend(fontsize=8.5,loc='upper left',framealpha=0.9)
    ax3.grid(True,axis='x',alpha=0.3,color='#EEEEEE')
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    plt.suptitle(
        'CG-CRASH: Concept-Guided Interpretable Traffic Accident Prediction',
        fontsize=14,fontweight='bold',color=BK,y=1.005)
    fig.savefig(out,dpi=180,bbox_inches='tight',facecolor=W)
    plt.close(fig); print(f'  [\u2713] {Path(out).name}')

# ── Fig 2: Concept importance ────────────────────────────────────────────────
def fig_concepts(acts,labels,cnames,out,ds_name,top_k=15):
    tidx,dv=top_disc(acts,labels,top_k)
    ma=acts.mean(1); pos=ma[labels==1]; neg=ma[labels==0]
    pm=pos.mean(0)[tidx]; nm=neg.mean(0)[tidx] if len(neg) else np.zeros(top_k)
    wrapped=[wrapn(cnames[i],38) for i in tidx]
    y=np.arange(top_k); bh=0.38
    fig,axes=plt.subplots(1,2,figsize=(17,top_k*0.56+2.2),facecolor=W,
        gridspec_kw={'width_ratios':[2.8,1],'wspace':0.10})
    ax=axes[0]
    ax.barh(y+bh/2,pm,height=bh,color=RED,alpha=0.82,label='Accident samples')
    ax.barh(y-bh/2,nm,height=bh,color=BLUE,alpha=0.82,label='Normal samples')
    ax.set_yticks(y); ax.set_yticklabels(wrapped,fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Concept Activation',fontsize=11)
    ax.set_title(f'Top-{top_k} Safety-Critical Concepts  [{ds_name.upper()}]',
        fontsize=13,fontweight='bold',pad=7)
    ax.legend(fontsize=10,loc='upper right')
    ax.grid(True,axis='x',alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # Only draw arrows for top-5 most discriminative (avoid clutter)
    top5_disc_set=set(tidx[:5])
    for i,(p_,n_) in enumerate(zip(pm,nm)):
        if tidx[i] in top5_disc_set and abs(p_-n_)>0.06:
            ax.annotate('',xy=(max(p_,n_)+0.008,i),xytext=(min(p_,n_),i),
                arrowprops=dict(arrowstyle='<->',color=ORG,lw=1.6))
    ax2=axes[1]
    norm_=plt.Normalize(dv.min(),dv.max())
    ax2.barh(y,dv,color=[plt.cm.RdPu(0.3+0.7*v/max(dv.max(),1e-9)) for v in dv],
        alpha=0.88,height=0.55,edgecolor=W,lw=0.4)
    ax2.set_yticks([]); ax2.invert_yaxis()
    ax2.set_xlabel("Cohen's d",fontsize=11)
    ax2.set_title('Discrimin.\nScore',fontsize=10,fontweight='bold',pad=7)
    xmax_d=dv.max()*1.35
    ax2.set_xlim(0,xmax_d)
    for i,v in enumerate(dv):
        ax2.text(min(v+dv.max()*0.03,xmax_d*0.95),i,f'{v:.2f}',
            va='center',fontsize=9,ha='left' if v<dv.max()*0.7 else 'right')
    ax2.grid(True,axis='x',alpha=0.3)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    sm=plt.cm.ScalarMappable(cmap='RdPu',norm=norm_)
    plt.colorbar(sm,ax=ax2,shrink=0.55,pad=0.15,label="Cohen's d")
    plt.suptitle('Concept-Level Interpretability Analysis',fontsize=14,
        fontweight='bold',color=BK,y=1.01)
    plt.tight_layout()
    fig.savefig(out,dpi=180,bbox_inches='tight',facecolor=W)
    plt.close(fig); print(f'  [\u2713] {Path(out).name}')

# ── Fig 3: Multi-case ────────────────────────────────────────────────────────
def fig_multi(best_idx,acts,probs,toas,cnames,fps,video_dir,out,ds_name):
    """Multi-case: n best samples, each row = 8-frame strip + prob curve + concept heatmap."""
    T=acts.shape[1]; t_ax=np.arange(T)/fps
    from collections import Counter
    all_delta=[top_delta_concepts(acts[si],toas[si],k=8)[0] for si in best_idx]
    cnt=Counter(idx for row in all_delta for idx in row)
    tidx=np.array([c[0] for c in cnt.most_common(8)])
    short_c=[short(cnames[i],32) for i in tidx]
    n=len(best_idx)
    row_h=8.5
    fig=plt.figure(figsize=(38,row_h*n+1.0),facecolor=W)
    gs_outer=gridspec.GridSpec(n,1,figure=fig,hspace=0.22)
    for row,si in enumerate(best_idx):
        toa_f=int(toas[si]); p=probs[si]
        cross_idx=np.where(p>0.5)[0]
        vpath=get_vpath(video_dir,si)
        gs_inner=gridspec.GridSpecFromSubplotSpec(1,3,
            subplot_spec=gs_outer[row],
            width_ratios=[6.0,1.8,1.4],wspace=0.10)
        ax_f=fig.add_subplot(gs_inner[0]); ax_f.set_axis_off()
        frames=get_frames(vpath,8,W=420,H=340) if vpath else None
        if frames is not None:
            pad=4; H_fr,W_fr=frames[0].shape[:2]
            total_W=W_fr*8+pad*7
            canvas=np.ones((H_fr,total_W,3),np.uint8)*215
            fi_arr=np.linspace(0,T-1,8,dtype=int)
            for j,fr in enumerate(frames):
                x0=j*(W_fr+pad); canvas[:,x0:x0+W_fr]=fr
            ax_f.imshow(canvas)
            for j,fi in enumerate(fi_arr):
                xc=j*(W_fr+pad)+W_fr//2; x0=j*(W_fr+pad)
                is_crash=toa_f<T and fi>=toa_f
                is_warn=not is_crash and p[fi]>0.5
                bc=RED if is_crash else ORG if is_warn else '#888'
                bw=4.0 if is_crash else 2.0 if is_warn else 0.5
                if is_crash or is_warn:
                    ax_f.add_patch(plt.Rectangle((x0-1,-1),W_fr+1,H_fr+1,
                        lw=bw,edgecolor=bc,facecolor='none',zorder=5))
                # Timestamp labels inside frame at bottom
                ax_f.text(xc,H_fr-8,f'{fi/fps:.1f}s',ha='center',va='bottom',
                    fontsize=8.5,color='white',fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15',facecolor=bc,
                              edgecolor='none',alpha=0.75))
            ax_f.set_xlim(-3,total_W+3)
            ax_f.set_ylim(H_fr+5,-5)
        else:
            ax_f.set_facecolor('#F0F0F0')
            ax_f.text(0.5,0.5,f'No video\nSample #{si+1}',ha='center',va='center',
                fontsize=10,color=G2,transform=ax_f.transAxes)
        warn_s=max(0,(toa_f-(cross_idx[0] if len(cross_idx) and cross_idx[0]<toa_f else toa_f)))/fps
        ax_f.set_title(
            f'Sample #{si+1}  \u00b7  TOA={toa_f/fps:.1f}s  \u00b7  Peak={p.max():.3f}  \u00b7  Warning={warn_s:.1f}s',
            fontsize=9.5,fontweight='bold',pad=8,
            color=RED if p.max()>0.9 else ORG)
        ax_p=fig.add_subplot(gs_inner[1])
        ax_p.fill_between(t_ax,p,alpha=0.12,color=GRN,where=p<=0.5,interpolate=True)
        ax_p.fill_between(t_ax,p,alpha=0.18,color=RED,where=p>0.5,interpolate=True)
        ax_p.plot(t_ax,p,color='#1A1A2E',lw=2.0)
        ax_p.axhline(0.5,color=G2,lw=0.9,ls='--',alpha=0.7)
        if toa_f<T: ax_p.axvline(toa_f/fps,color=RED,lw=1.8,ls=':',alpha=0.8)
        if len(cross_idx) and cross_idx[0]<toa_f:
            ax_p.axvline(cross_idx[0]/fps,color=ORG,lw=1.4,ls=':')
        ax_p.set_ylim(0,1.08); ax_p.set_xlim(0,t_ax[-1])
        ax_p.set_ylabel('P(accident)',fontsize=9.5); ax_p.set_xlabel('Time (s)',fontsize=9.5)
        ax_p.grid(True,alpha=0.3,color='#EEEEEE')
        ax_p.spines['top'].set_visible(False); ax_p.spines['right'].set_visible(False)
        if row==0: ax_p.set_title('Prediction Confidence',fontsize=11,fontweight='bold',pad=5)
        ax_h=fig.add_subplot(gs_inner[2])
        heat=norm_rows(acts[si][:,tidx].T)
        im=ax_h.imshow(heat,aspect='auto',cmap=CMAP_ACT,
            extent=[0,t_ax[-1],8-0.5,-0.5],vmin=0,vmax=1,interpolation='bilinear')
        ax_h.set_yticks(range(8)); ax_h.set_yticklabels(short_c,fontsize=8)
        if toa_f<T: ax_h.axvline(toa_f/fps,color='white',lw=1.8,ls=':',alpha=0.9)
        if len(cross_idx) and cross_idx[0]<toa_f:
            ax_h.axvline(cross_idx[0]/fps,color=ORG,lw=1.2,ls=':',alpha=0.8)
        ax_h.set_xlabel('Time (s)',fontsize=9.5)
        plt.colorbar(im,ax=ax_h,shrink=0.80,pad=0.01,aspect=15).ax.tick_params(labelsize=7)
        if row==0: ax_h.set_title('Concept Activations',fontsize=11,fontweight='bold',pad=5)
    plt.suptitle(f'CG-CRASH: Top-{n} Best Predicted Cases  [{ds_name.upper()}]',
        fontsize=14,fontweight='bold',y=1.005)
    fig.savefig(out,dpi=160,bbox_inches='tight',facecolor=W)
    plt.close(fig); print(f'  [\u2713] {Path(out).name}')

# ── Smooth GIF Animation ─────────────────────────────────────────────────────
def make_animation(si, acts, probs, toas, cnames, fps, video_dir, out, ds_name):
    """Smooth GIF: composite video frame + live prob curve + concept bar.
    Covers the full sequence (all T frames), sampled at ~8fps for smooth playback.
    """
    T=acts.shape[1]; toa_f=int(toas[si]); p=probs[si]
    t_ax=np.arange(T)/fps

    # Delta concepts for interpretable bars
    top_idx,safe_v,crash_v,delta_v=top_delta_concepts(acts[si],toas[si],k=6)
    short_c=[short(cnames[i],28) for i in top_idx]
    bar_max=max(acts[si][:,top_idx].max(),0.5)+0.05

    # Sample frames: cover full sequence, ~50 frames total for smooth GIF
    N_GIF=min(T,50)
    frame_indices=np.linspace(0,T-1,N_GIF,dtype=int)

    # Pre-extract video frames (mapped to model frame indices)
    vpath=get_vpath(video_dir,si)
    vid_frames=None
    if vpath:
        vid_frames=get_frames(vpath,N_GIF,W=360,H=202,
                              start_f=0,end_f=T,total_model_frames=T)

    # Light theme palette
    BG='#FFFFFF'; TXT='#2A2A2A'; GRID='#EEEEEE'
    C_SAFE=GRN; C_ALRT=ORG; C_CRSH=RED

    from PIL import Image as PILImage
    gif_frames=[]

    for fi,t in enumerate(frame_indices):
        fig=plt.figure(figsize=(14,5),facecolor=BG)
        plt.rcParams.update({
            'axes.facecolor':'#FAFAFA','figure.facecolor':BG,
            'text.color':TXT,'axes.labelcolor':TXT,
            'xtick.color':'#666','ytick.color':'#666',
            'axes.edgecolor':'#CCCCCC','grid.color':GRID,'grid.alpha':1.0,
            'axes.titlesize':10,'axes.labelsize':9,
            'xtick.labelsize':8,'ytick.labelsize':8,
        })
        gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.42,
            width_ratios=[1.6,2.2,1.2])

        # ── Panel 0: video frame ──────────────────────────────────────────
        ax0=fig.add_subplot(gs[0]); ax0.set_axis_off()
        if vid_frames is not None:
            ax0.imshow(vid_frames[fi])
        else:
            ax0.set_facecolor('#1A1A2E')
            ax0.text(0.5,0.5,f't={t/fps:.1f}s',ha='center',va='center',
                fontsize=13,color='#555',transform=ax0.transAxes)
        is_crash=toa_f<T and t>=toa_f
        is_alert=not is_crash and p[t]>0.5
        status='CRASH' if is_crash else ('ALERT' if is_alert else 'NORMAL')
        sc=C_CRSH if is_crash else (C_ALRT if is_alert else C_SAFE)
        if is_crash or is_alert:
            for sp in ax0.spines.values():
                sp.set_visible(True); sp.set_color(sc); sp.set_linewidth(4)
        ax0.set_title(f't = {t/fps:.1f}s   [{status}]   P = {p[t]:.3f}',
            color=sc,fontsize=10,fontweight='bold',pad=5)

        # ── Panel 1: growing prob curve ───────────────────────────────────
        ax1=fig.add_subplot(gs[1])
        ax1.fill_between(t_ax[:t+1],p[:t+1],
            where=p[:t+1]<=0.5,color=C_SAFE,alpha=0.20,interpolate=True)
        ax1.fill_between(t_ax[:t+1],p[:t+1],
            where=p[:t+1]>0.5,color=C_ALRT,alpha=0.25,interpolate=True)
        # Future (faded)
        ax1.plot(t_ax[t:],p[t:],color='#333344',lw=1.2,ls='--')
        # Past (solid)
        ax1.plot(t_ax[:t+1],p[:t+1],color=TXT,lw=2.0)
        ax1.axhline(0.5,color='#555',lw=0.8,ls='--')
        if toa_f<T:
            ax1.axvline(toa_f/fps,color=C_CRSH,lw=1.5,ls=':',alpha=0.7)
        # Current position dot
        ax1.scatter([t/fps],[p[t]],color=sc,s=60,zorder=6,edgecolors='white',lw=1)
        ax1.set_xlim(0,t_ax[-1]); ax1.set_ylim(-0.02,1.08)
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('P(accident)')
        ax1.set_title('Accident Probability',fontweight='bold',pad=4)
        ax1.grid(True)

        # ── Panel 2: live concept bar ──────────────────────────────────────
        ax2=fig.add_subplot(gs[2])
        vals=acts[si,t,top_idx]
        bar_cc=[C_CRSH if v>safe_v[i]*1.5 else C_ALRT if v>safe_v[i]*1.1 else C_SAFE
                for i,v in enumerate(vals)]
        ax2.barh(range(6),safe_v,height=0.55,color='#1A3550',
                 alpha=0.8,edgecolor='none',zorder=2)
        ax2.barh(range(6),vals,height=0.55,color=bar_cc,
                 alpha=0.85,edgecolor='none',zorder=3)
        ax2.set_xlim(0,bar_max)
        # Short concept labels on y-axis, enough left margin
        ax2.set_yticks(range(6))
        ax2.set_yticklabels([c[:22] for c in short_c],fontsize=7)
        ax2.invert_yaxis()
        ax2.set_xlabel('Activation')
        ax2.set_title('Live Concepts',fontweight='bold',pad=4)
        ax2.grid(True,axis='x')
        ax2.tick_params(axis='y',pad=2)

        fig.suptitle(
            f'CG-CRASH Interpretable Prediction  ·  {ds_name.upper()}  '
            f'·  Sample #{si+1}  ·  Early Warning: '
            f'{max(0,toa_f-(np.where(p>0.5)[0][0] if any(p>0.5) else toa_f))/fps:.1f}s',
            fontsize=10,color=TXT,y=1.01)

        buf=io.BytesIO()
        fig.savefig(buf,format='png',dpi=90,bbox_inches='tight',
                    facecolor=BG,edgecolor='none')
        plt.close(fig)
        # Restore light theme
        plt.rcParams.update({'axes.facecolor':W,'figure.facecolor':W,
            'text.color':BK,'axes.labelcolor':BK,'xtick.color':G1,
            'ytick.color':G1,'axes.edgecolor':G3,'grid.color':G3,
            'axes.titlesize':13,'axes.labelsize':12,
            'xtick.labelsize':11,'ytick.labelsize':11})
        buf.seek(0)
        gif_frames.append(PILImage.fromarray(imageio.imread(buf)))

    gif_out=out.replace('.mp4','.gif')
    try:
        # Save with imageio for reliable multi-frame GIF
        import imageio as iio
        with iio.get_writer(gif_out, mode='I', duration=80, loop=0) as writer:
            for frame in gif_frames:
                writer.append_data(np.array(frame))
        print(f'  [\u2713] {Path(gif_out).name}  ({N_GIF} frames @ 12fps)')
    except Exception as e:
        # Fallback to PIL
        try:
            gif_frames[0].save(
                gif_out,save_all=True,append_images=gif_frames[1:],
                duration=80,loop=0,optimize=False)
            print(f'  [\u2713] {Path(gif_out).name}  ({N_GIF} frames @ 12fps, PIL)')
        except Exception as e2:
            print(f'  [!] GIF failed: {e} / {e2}')

# ── Interactive HTML Dashboard (HCI) ─────────────────────────────────────────
def make_html(acts, probs, labels, toas, cnames, fps, video_dir, out, ds_name):
    T=acts.shape[1]; t_ax=(np.arange(T)/fps).tolist()
    best_idx=select_best_samples(acts,probs,labels,toas,n=5)
    tidx,dv=top_disc(acts,labels,20)
    ma=acts.mean(1); pos=ma[labels==1]; neg=ma[labels==0]
    pm=pos.mean(0)[tidx]; nm=neg.mean(0)[tidx] if len(neg) else np.zeros(20)
    top_names=[short(cnames[i],50) for i in tidx]

    # Use 4 rows, 1 col layout to avoid overlapping
    fig=make_subplots(
        rows=4,cols=1,
        subplot_titles=[
            f'[{ds_name.upper()}] Prediction Confidence — Top-5 Best Samples',
            f'Top-20 Discriminative Concepts (Accident vs Normal)',
            f'Concept Activation Timeline — Best Sample (TOA marked)',
            f'Concept Discriminability (Cohen\'s d)'],
        specs=[[{'type':'scatter'}],
               [{'type':'bar'}],
               [{'type':'heatmap'}],
               [{'type':'bar'}]],
        row_heights=[0.20,0.28,0.30,0.22],
        vertical_spacing=0.08)

    # P1: confidence curves for top-5 samples
    pal=[RED,'#E67E22','#27AE60','#2980B9','#8E44AD']
    for k,si in enumerate(best_idx):
        toa_f=int(toas[si])
        fig.add_trace(go.Scatter(x=t_ax,y=probs[si].tolist(),mode='lines',
            name=f'Sample #{si+1} (TOA={toa_f/fps:.1f}s)',
            line=dict(color=pal[k%5],width=2.2),
            hovertemplate='t=%{x:.1f}s  P=%{y:.3f}<br><extra></extra>'),row=1,col=1)
        if toa_f<T:
            fig.add_vline(x=toa_f/fps,line=dict(color=pal[k%5],dash='dot',width=1.5),row=1,col=1)
    fig.add_hline(y=0.5,line=dict(color='#888888',dash='dash',width=1.5),row=1,col=1)

    # P2: grouped concept bar chart (accident vs normal)
    fig.add_trace(go.Bar(name='Accident',x=pm,y=top_names,orientation='h',
        marker_color=RED,opacity=0.82,
        hovertemplate='%{y}<br>Accident: %{x:.3f}<extra></extra>'),row=2,col=1)
    fig.add_trace(go.Bar(name='Normal',x=nm,y=top_names,orientation='h',
        marker_color=BLUE,opacity=0.82,
        hovertemplate='%{y}<br>Normal: %{x:.3f}<extra></extra>'),row=2,col=1)

    # P3: heatmap (best sample)
    si0=best_idx[0]; toa0=int(toas[si0])
    heat=norm_rows(acts[si0][:,tidx[:15]].T)
    fig.add_trace(go.Heatmap(z=heat.tolist(),x=t_ax,
        y=[short(cnames[i],45) for i in tidx[:15]],
        colorscale=[[0,'#FFFFFF'],[0.4,'#ECD0CB'],[0.7,'#E4A3A1'],[1,'#DF908A']],
        colorbar=dict(title=dict(text='Norm.Act',side='right'),
                      thickness=15,len=0.28,y=0.35,yanchor='middle'),
        hovertemplate='t=%{x:.1f}s  act=%{z:.3f}<extra></extra>'),row=3,col=1)
    if toa0<T:
        fig.add_vline(x=toa0/fps,line=dict(color=ORG,dash='dot',width=2.5),row=3,col=1)

    # P4: discriminability scores
    disc_colors=[f'rgba({max(0,int(196*(v/max(dv.max(),1e-9))))},'
                 f'{max(0,int(40*(1-v/max(dv.max(),1e-9))))},'
                 f'{max(0,int(40*(1-v/max(dv.max(),1e-9))))},0.85)'
                 for v in dv]
    fig.add_trace(go.Bar(x=dv.tolist(),y=top_names,orientation='h',
        marker_color=disc_colors,name="Cohen's d",showlegend=False,
        hovertemplate='%{y}<br>d=%{x:.3f}<extra></extra>'),row=4,col=1)

    # Layout
    fig.update_layout(
        height=1800,
        paper_bgcolor='#FAFAFA',plot_bgcolor='#FFFFFF',
        font=dict(family='Georgia, serif',color='#111111',size=12),
        title=dict(
            text=f'<b>CG-CRASH Interactive Analysis Dashboard</b>  [{ds_name.upper()}]',
            font=dict(size=20,color='#1A1A1A'),x=0.5),
        barmode='group',
        legend=dict(bgcolor='rgba(255,255,255,0.95)',bordercolor='#CCCCCC',
                    borderwidth=1,font=dict(size=11),
                    x=1.01,xanchor='left'),
        hoverlabel=dict(bgcolor='white',font_size=12,bordercolor='#CCCCCC'),
        margin=dict(l=220,r=120,t=80,b=60))
    fig.update_xaxes(gridcolor='#EEEEEE',zerolinecolor='#DDDDDD',color='#444444')
    fig.update_yaxes(gridcolor='#EEEEEE',zerolinecolor='#DDDDDD',color='#444444',
                     tickfont=dict(size=10))
    # P2 y-axis: enough height for 20 concepts
    fig.update_yaxes(automargin=True,row=2,col=1)
    fig.update_yaxes(automargin=True,row=4,col=1)

    fig.add_annotation(
        text='ℹ Click legend to toggle | Hover for values | Scroll to zoom',
        xref='paper',yref='paper',x=0.5,y=-0.02,
        showarrow=False,font=dict(size=11,color='#666666'),align='center')

    fig.write_html(out,include_plotlyjs='cdn',
        config={'displayModeBar':True,'scrollZoom':True,
                'modeBarButtonsToAdd':['drawline','eraseshape'],
                'responsive':True})
    print(f'  [\u2713] {Path(out).name}')

# ── Fig 4: Paper Strip — CVPR-style wide figure ────────────────────────────
def fig_paper_strip(si, acts, probs, toas, cnames, fps, video_dir, out, ds_name):
    SAFE_C=GRN; ALRT_C=ORG; CRSH_C=RED
    BG=W; TXT=BK; GRID=G3
    T=acts.shape[1]; t_ax=np.arange(T)/fps
    toa_f=int(toas[si]); p=probs[si]
    cross=np.where(p>0.5)[0]
    alert_f=int(cross[0]) if len(cross) and cross[0]<toa_f else max(0,toa_f-8)
    kf=[max(0,int(toa_f*0.15)),max(0,alert_f-3),min(T-1,alert_f+2),min(T-1,toa_f)]
    kf_lbl=['Normal','Pre-Alert','Alert','Crash']; kf_cc=[SAFE_C,SAFE_C,ALRT_C,CRSH_C]
    top_idx,safe_v,crash_v,delta_v=top_delta_concepts(acts[si],toas[si],k=6)
    cnames_d=[short(cnames[i],36) for i in top_idx]
    vmax=max(safe_v.max(),crash_v.max())+1e-9
    vpath=get_vpath(video_dir,si)
    frames=[]
    for fi in kf:
        f=get_frames(vpath,1,W=380,H=214,start_f=fi,end_f=fi+1,total_model_frames=T) if vpath else None
        frames.append(f[0] if f is not None else None)
    fig=plt.figure(figsize=(22,10),facecolor=BG)
    gs=gridspec.GridSpec(3,1,figure=fig,height_ratios=[2.8,2.8,1.6],hspace=0.22)
    gs_fr=gridspec.GridSpecFromSubplotSpec(1,4,subplot_spec=gs[0],wspace=0.05)
    gs_br=gridspec.GridSpecFromSubplotSpec(1,4,subplot_spec=gs[1],wspace=0.05)
    ax_prob=fig.add_subplot(gs[2])
    for j,(fi,lbl,cc) in enumerate(zip(kf,kf_lbl,kf_cc)):
        ax=fig.add_subplot(gs_fr[j]); ax.set_axis_off()
        fr=frames[j]
        if fr is not None:
            ax.imshow(fr)
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(cc); sp.set_linewidth(4.5)
        else:
            ax.set_facecolor('#F0F0F0')
            ax.text(0.5,0.5,f'No video\nt={fi/fps:.1f}s',ha='center',va='center',fontsize=11,color='#999',transform=ax.transAxes)
        ax.text(0.5,1.03,lbl,ha='center',va='bottom',transform=ax.transAxes,fontsize=12,fontweight='bold',color=cc)
        ax.text(0.5,-0.04,f't={fi/fps:.1f}s   P={p[fi]:.2f}',ha='center',va='top',transform=ax.transAxes,fontsize=10,color=cc if lbl in ('Alert','Crash') else TXT)
    for j,(fi,lbl,cc) in enumerate(zip(kf,kf_lbl,kf_cc)):
        ax=fig.add_subplot(gs_br[j])
        cur_v=acts[si,fi,top_idx]
        y=np.arange(len(top_idx))
        ax.barh(y,safe_v,height=0.55,color='#B5DCDF',alpha=0.8,edgecolor='none',label='Safe baseline')
        ax.barh(y,cur_v,height=0.55,color=cc,alpha=0.85,edgecolor='white',lw=0.4,label='Current')
        ax.set_xlim(0,vmax*1.45)
        # Only show concept labels on first column to avoid repetition
        if j==0:
            ax.set_yticks(y); ax.set_yticklabels(cnames_d,fontsize=10)
            ax.set_title('Concept Activations\n(teal=safe baseline)',fontsize=9.5,fontweight='bold',pad=4)
            ax.legend(fontsize=8,loc='lower right')
        else:
            ax.set_yticks(y); ax.set_yticklabels(['']*len(y))
            ax.set_title(lbl,fontsize=10,fontweight='bold',color=cc,pad=4)
        ax.invert_yaxis(); ax.set_xlabel('Activation',fontsize=9.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_facecolor(BG); ax.grid(True,axis='x',color=GRID,linewidth=0.8)
        # Delta labels only on Crash column, placed clearly outside bars
        if lbl=='Crash':
            for yi,(sv,cv) in enumerate(zip(safe_v,cur_v)):
                d=cv-sv
                if abs(d)>0.01:
                    dlabel=f'+{d:.2f}' if d>0 else f'{d:.2f}'
                    ax.text(vmax*1.22,yi,dlabel,va='center',ha='left',
                        fontsize=9,fontweight='bold',
                        color=CRSH_C if d>0 else SAFE_C,
                        bbox=dict(boxstyle='round,pad=0.12',facecolor='white',
                                  edgecolor=CRSH_C if d>0 else SAFE_C,alpha=0.85,lw=0.6))
    ax_prob.fill_between(t_ax,0,p,where=p<=0.5,color=SAFE_C,alpha=0.15,interpolate=True)
    ax_prob.fill_between(t_ax,0,p,where=(p>0.5)&(t_ax<toa_f/fps),color=ALRT_C,alpha=0.22,interpolate=True)
    ax_prob.fill_between(t_ax,0,p,where=t_ax>=toa_f/fps,color=CRSH_C,alpha=0.22,interpolate=True)
    ax_prob.plot(t_ax,p,color=TXT,lw=2.2,zorder=4)
    ax_prob.axhline(0.5,color='#999',lw=1.0,ls='--',alpha=0.6)
    ax_prob.axvline(toa_f/fps,color=CRSH_C,lw=2.0,ls=':',zorder=5,label=f'Crash onset ({toa_f/fps:.1f}s)')
    if len(cross) and cross[0]<toa_f:
        ws=(toa_f-cross[0])/fps
        ax_prob.axvline(cross[0]/fps,color=ALRT_C,lw=1.5,ls=':',label=f'Alert ({cross[0]/fps:.1f}s, {ws:.1f}s early)')
    for fi,lbl,cc in zip(kf,kf_lbl,kf_cc):
        ax_prob.scatter(fi/fps,p[fi],s=80,color=cc,zorder=6,edgecolors='white',linewidths=1.2)
        ax_prob.axvline(fi/fps,color=cc,lw=0.8,ls=':',alpha=0.4,zorder=2)
    ax_prob.set_ylim(-0.03,1.10); ax_prob.set_xlim(0,t_ax[-1])
    ax_prob.set_ylabel('P(accident)',fontsize=11); ax_prob.set_xlabel('Time (seconds)',fontsize=11)
    ax_prob.legend(fontsize=10,loc='upper left',ncol=3,framealpha=0.95)
    ax_prob.spines['top'].set_visible(False); ax_prob.spines['right'].set_visible(False)
    ax_prob.grid(True,color=GRID,linewidth=0.8)
    ws=max(0,(toa_f-(cross[0] if len(cross) else toa_f)))/fps
    fig.suptitle(f'CG-CRASH · {ds_name.upper()} · Sample #{si+1} · P₀={p[0]:.3f}→{p.max():.3f} · {ws:.1f}s early warning',
        fontsize=13,fontweight='bold',color=TXT,y=1.008)
    fig.savefig(out,dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(fig); print(f'  [✓] {Path(out).name}')

# ── Fig 5: Timeline Concepts ─────────────────────────────────────────────────
def fig_timeline_concepts(si, acts, probs, toas, cnames, fps, video_dir, out, ds_name):
    """Tall figure: concept evolution Safe→Alert→Crash with fixed concept order and delta."""
    SAFE_C='#2E86AB'; ALRT_C='#F6AE2D'; CRSH_C='#C1292E'
    BG='#FFFFFF'; TXT='#1A1A2E'; GRID='#EEEEEE'
    T=acts.shape[1]; t_ax=np.arange(T)/fps
    toa_f=int(toas[si]); p=probs[si]
    cross=np.where(p>0.5)[0]
    alert_f=int(cross[0]) if len(cross) and cross[0]<toa_f else max(0,toa_f-8)
    # Use delta concepts — fixed order by Safe→Crash change
    top_idx,safe_v,crash_v,delta_v=top_delta_concepts(acts[si],toas[si],k=10)
    cnames_fixed=[short(cnames[i],52) for i in top_idx]
    stages=[
        ('Normal',  list(range(max(0,int(toa_f*0.1)),max(1,alert_f-2))), SAFE_C),
        ('Alert',   list(range(alert_f, min(T,toa_f))),                  ALRT_C),
        ('Crash',   list(range(toa_f,  min(T,toa_f+5))),                 CRSH_C),
    ]
    # Per-stage mean values in FIXED concept order
    stage_vals=[]
    for name,rng,cc in stages:
        if not rng: rng=[toa_f]
        stage_vals.append(acts[si,rng,:][:,top_idx].mean(0))
    vmax=max(v.max() for v in stage_vals)+1e-9

    fig=plt.figure(figsize=(14,17),facecolor=BG)
    gs=gridspec.GridSpec(4,1,figure=fig,height_ratios=[1.5,2.4,2.4,2.4],hspace=0.40)

    # Row 0: prob curve
    ax0=fig.add_subplot(gs[0])
    ax0.fill_between(t_ax,p,where=p<=0.5,color=SAFE_C,alpha=0.15,interpolate=True)
    ax0.fill_between(t_ax,p,where=(p>0.5)&(t_ax<toa_f/fps),color=ALRT_C,alpha=0.20,interpolate=True)
    ax0.fill_between(t_ax,p,where=t_ax>=toa_f/fps,color=CRSH_C,alpha=0.22,interpolate=True)
    ax0.plot(t_ax,p,color=TXT,lw=2.2)
    ax0.axhline(0.5,color='#AAA',lw=0.9,ls='--')
    ax0.axvline(toa_f/fps,color=CRSH_C,lw=2.0,ls=':',label=f'Crash @ {toa_f/fps:.1f}s')
    if len(cross) and cross[0]<toa_f:
        ax0.axvline(cross[0]/fps,color=ALRT_C,lw=1.5,ls=':',label=f'Alert @ {cross[0]/fps:.1f}s')
    for name,rng,cc in stages:
        if rng: ax0.axvspan(rng[0]/fps,rng[-1]/fps,color=cc,alpha=0.10,zorder=0)
    ax0.set_ylim(0,1.10); ax0.set_xlim(0,t_ax[-1])
    ax0.set_ylabel('P(accident)',fontsize=12); ax0.set_xlabel('Time (s)',fontsize=12)
    ax0.legend(fontsize=10,ncol=2,loc='upper left')
    ax0.set_title('Accident Prediction Confidence',fontsize=13,fontweight='bold',pad=6)
    ax0.grid(True,color=GRID); ax0.spines['top'].set_visible(False); ax0.spines['right'].set_visible(False)

    # Rows 1-3: same concept order, show absolute bar + safe baseline overlay
    for row_i,((name,rng,cc),vals) in enumerate(zip(stages,stage_vals)):
        ax=fig.add_subplot(gs[row_i+1])
        y=np.arange(len(top_idx))
        # Safe baseline as light background bar (FIXED x range = vmax)
        ax.barh(y,safe_v,height=0.60,color='#AED6F1',alpha=0.90,
                edgecolor='#2E86AB',lw=0.8,zorder=2)
        # Current stage bar
        ax.barh(y,vals,height=0.60,color=cc,alpha=0.85,edgecolor='white',lw=0.4,zorder=3)
        ax.set_xlim(0,vmax*1.50)  # shared x range, extra room for delta labels
        ax.set_yticks(y); ax.set_yticklabels(cnames_fixed,fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel('Mean Activation',fontsize=11)
        t0=rng[0]/fps if rng else 0; t1=rng[-1]/fps if len(rng)>1 else t0
        p_mean=float(np.mean([p[r] for r in rng])) if rng else 0
        ax.set_title(f'{name}  \u00b7  t={t0:.1f}\u2013{t1:.1f}s  \u00b7  P\u0304={p_mean:.2f}',
            fontsize=12,fontweight='bold',color=cc,pad=6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(True,axis='x',color=GRID)
        # Delta annotation — placed OUTSIDE bars at fixed right position
        for yi,(sv,v) in enumerate(zip(safe_v,vals)):
            d=v-sv
            arrow='\u2191' if d>0.02 else ('\u2193' if d<-0.02 else '\u2192')
            dcol=CRSH_C if d>0.02 else (SAFE_C if d<-0.02 else '#888888')
            ax.text(vmax*1.22,yi,f'{arrow}{d:+.2f}',va='center',ha='left',
                fontsize=9.5,fontweight='bold',color=dcol,
                bbox=dict(boxstyle='round,pad=0.15',facecolor='white',
                          edgecolor=dcol,alpha=0.85,lw=0.8))

    fig.suptitle(
        f'Concept Evolution: Normal → Alert → Crash  [{ds_name.upper()}  Sample #{si+1}]\n'
        f'P₀={p[0]:.3f} → Peak={p.max():.3f}  ·  '
        f'Early warning {max(0,toa_f-(cross[0] if len(cross) else toa_f))/fps:.1f}s  ·  '
        f'Top-10 concepts ranked by Safe→Crash Δ activation',
        fontsize=12,fontweight='bold',color=TXT,y=1.02)
    fig.savefig(out,dpi=180,bbox_inches='tight',facecolor=BG)
    plt.close(fig); print(f'  [✓] {Path(out).name}')


# ── Main ──────────────────────────────────────────────────────────────────────
def run(ds_name, ckpt_dir, out_dir, concept_file, device, max_s=200):
    ckpt=Path(ckpt_dir)/f'{ds_name}_full'/'best_model.pth'
    if not ckpt.exists(): print(f'  [!] Missing {ckpt}'); return
    print(f'\n=== {ds_name.upper()} ===')
    model=load_model(str(ckpt),device)
    m=DS_META[ds_name]
    dataset=m['cls'](m['data_path'],m['feature'],m['phase'])
    cnames=load_concepts(concept_file)
    # Use dedicated visualizations folder
    out=Path(out_dir)/ds_name; out.mkdir(parents=True,exist_ok=True)
    # Ensure directory exists before any file writing
    out.mkdir(parents=True,exist_ok=True)

    # Load or extract activations
    cache=ROOT/'output'/'visualizations'/ds_name/'activations.npz'
    if cache.exists():
        print('  Using cached activations...')
        c=np.load(str(cache))
        acts,probs,labels,toas=c['acts'],c['probs'],c['labels'],c['toas']
    else:
        print('  Extracting...')
        acts,probs,labels,toas=extract_all(model,dataset,device,max_s)
        np.savez_compressed(str(out/'activations.npz'),
            acts=acts,probs=probs,labels=labels,toas=toas)

    fps=m['fps']; video_dir=m.get('video_dir')
    anno=None
    if ds_name=='crash' and 'anno_file' in m:
        try: anno=read_anno(m['anno_file'])
        except: pass

    # Smart sample selection — get top 5 for more choice
    best_idx=select_best_samples(acts,probs,labels,toas,n=5)
    print(f'  Best samples: {best_idx} (peak probs: {[f"{probs[i].max():.3f}" for i in best_idx]})')

    print('  Generating figures...')
    # Hero figure — generate for top 3 samples to give more choices
    for k in range(min(3,len(best_idx))):
        si=best_idx[k]
        suffix='' if k==0 else f'_sample{k+1}'
        fig_hero(si,acts,probs,toas,cnames,fps,video_dir,
                 str(out/f'best_case_study{suffix}.png'),ds_name,anno)
    # Concept importance
    fig_concepts(acts,labels,cnames,str(out/'concept_importance.png'),ds_name)
    # Multi-case — show top 5
    fig_multi(best_idx,acts,probs,toas,cnames,fps,video_dir,
              str(out/'multi_case.png'),ds_name)
    # Interactive HTML
    make_html(acts,probs,labels,toas,cnames,fps,video_dir,
              str(out/'interactive.html'),ds_name)
    # Smooth GIF animation (best sample)
    make_animation(best_idx[0],acts,probs,toas,cnames,fps,
                   video_dir,str(out/'animation_keyframes.mp4'),ds_name)
    # Paper strip — generate for top 2 samples
    for k in range(min(2,len(best_idx))):
        si=best_idx[k]
        suffix='' if k==0 else f'_sample{k+1}'
        fig_paper_strip(si,acts,probs,toas,cnames,fps,video_dir,
                        str(out/f'paper_strip{suffix}.png'),ds_name)
    # Timeline concepts
    fig_timeline_concepts(best_idx[0],acts,probs,toas,cnames,fps,video_dir,
                          str(out/'timeline_concepts.png'),ds_name)
    print(f'  Done -> {out}/')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='crash',choices=['crash','dad','a3d','all'])
    ap.add_argument('--gpu',default='0')
    ap.add_argument('--ckpt_dir',default='output/v3_final')
    ap.add_argument('--out_dir',default='visualizations')
    ap.add_argument('--concept_file',default='/data/sony/LFCRASH/000_all_concept_set.txt')
    ap.add_argument('--max_samples',type=int,default=200)
    args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    datasets=['crash','dad','a3d'] if args.dataset=='all' else [args.dataset]
    for ds in datasets:
        run(ds,str(ROOT/args.ckpt_dir),str(ROOT/args.out_dir),
            args.concept_file,device,args.max_samples)
    print('\n=== ALL DONE ===')

if __name__=='__main__':
    main()






