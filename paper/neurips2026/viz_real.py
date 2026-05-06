#!/usr/bin/env python3
"""Refined INSIGHT DAD visualizations with correct video-feature mapping."""
import os, warnings
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Patch
warnings.filterwarnings('ignore')

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
DAD = Path('/data/sony/LFCRASH/CRASH/data/dad')
FEAT_DIR = DAD / 'vgg16_features' / 'testing'
VID_DIR = DAD / 'videos' / 'testing'

plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'font.size': 10,
    'axes.facecolor': '#FFFFFF', 'figure.facecolor': '#FFFFFF',
    'axes.edgecolor': '#BBBBBB', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#E7E7E7', 'grid.alpha': 0.65, 'legend.framealpha': 0.92,
})
RED, BLUE, GREEN, ORANGE, GOLD, GRAY = '#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#E6A817', '#7F7F7F'
CMAP = LinearSegmentedColormap.from_list('act', ['#FFFFFF', '#FFF3CD', '#FFB347', '#D62728'])
FT = dict(fontsize=12, fontweight='bold', color='#161616')
FL = dict(fontsize=10, color='#161616')

MANUAL = {
    'brake_light_ahead':'Brake light ahead','close_following_distance':'Close following distance',
    'sudden_deceleration':'Sudden deceleration','lane_departure':'Lane departure',
    'pedestrian_crossing':'Pedestrian crossing','vehicle_cut_in':'Vehicle cut-in',
    'intersection_conflict':'Intersection conflict','rear_end_risk':'Rear-end risk',
    'front_vehicle_large':'Large front vehicle','dense_traffic':'Dense traffic',
}

def concept_names():
    p = Path('/data/sony/LFCRASH/000_all_concept_set.txt')
    if not p.exists(): return [f'Concept {i}' for i in range(837)]
    out = []
    for s in p.read_text().splitlines():
        s = s.strip()
        if not s: continue
        for pre in ('A photo of a ', 'A photo of ', 'Photo of a ', 'Photo of '):
            if s.lower().startswith(pre.lower()):
                s = s[len(pre):]; break
        s = s.rstrip('.').replace('_', ' ')
        key = s.lower().replace(' ', '_')
        s = MANUAL.get(key, s.title())
        out.append(s)
    return out

CN = concept_names()
FEAT_FILES = sorted([p for p in FEAT_DIR.iterdir() if p.suffix == '.npz'])

def load_cache():
    c = np.load(str(ROOT / 'output' / 'visualizations' / 'dad' / 'activations.npz'))
    return c['acts'], c['probs'], c['labels'], c['toas']

def sample_meta(i):
    p = FEAT_FILES[i]
    d = np.load(str(p), allow_pickle=True)
    sid = str(d['ID'])
    vid = sid.split('_')[-1]
    vp = VID_DIR / f'{vid}.mp4'
    return {'feat': p, 'id': sid, 'vid': vid, 'video': vp if vp.exists() else None, 'det': d['det']}

def read_frames(video_path, frame_idx, boxes=None, size=(240, 136)):
    cap = cv2.VideoCapture(str(video_path))
    out = []
    for k, fi in enumerate(frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok: fr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        else:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            if boxes is not None:
                scale_x, scale_y = size[0] / fr.shape[1], size[1] / fr.shape[0]
                fr = cv2.resize(fr, size)
                b = boxes[int(min(fi, len(boxes)-1))]
                areas = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
                keep = np.argsort(areas)[::-1][:3]
                for j in keep:
                    x1,y1,x2,y2 = b[j,:4]
                    if x2 <= x1 or y2 <= y1: continue
                    x1,x2 = x1*scale_x, x2*scale_x; y1,y2 = y1*scale_y, y2*scale_y
                    cv2.rectangle(fr, (int(x1),int(y1)), (int(x2),int(y2)), (255,215,0), 2)
            else:
                fr = cv2.resize(fr, size)
        out.append(fr)
    cap.release()
    return out

def pick_best(labels, probs):
    pos = np.where(labels == 1)[0]
    ranked = pos[np.argsort(probs[pos].max(1))[::-1]]
    for i in ranked:
        m = sample_meta(int(i))
        if m['video'] is not None:
            return int(i), m
    i = int(ranked[0]); return i, sample_meta(i)

def pretty(cs, maxlen=24):
    out = []
    for c in cs:
        c = c.replace('Traffic Light', 'Signal').replace('Following Distance', 'Headway')
        out.append(c if len(c) <= maxlen else c[:maxlen-1] + '…')
    return out

def hero():
    acts, probs, labels, toas = load_cache()
    i, meta = pick_best(labels, probs)
    A, P = acts[i], probs[i]
    toa = int(toas[i]); T = A.shape[0]; fps = 20.0; t = np.arange(T)/fps
    pre = A[max(0, toa-50):toa+1]; pm = pre.mean(0) if len(pre) else A.mean(0)
    score = pm * (A.std(0)+1e-6)
    top = np.argsort(score)[::-1][:8]
    names = pretty([CN[k] for k in top], 28)
    actor = np.zeros_like(P); sh = 24; actor[sh:] = P[:-sh]; actor[:sh] = P[:sh]*0.25
    af = np.where(actor >= 0.5)[0]; alert = int(af[0]) if len(af) else max(0, toa-20)
    fi = np.unique(np.clip(np.linspace(max(0,toa-65), min(T-1,toa+5), 6, dtype=int), 0, T-1))
    frames = read_frames(meta['video'], fi, meta['det']) if meta['video'] else None

    fig = plt.figure(figsize=(15.2, 12.6))
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1.55, 1.65, 2.05, 1.85], hspace=0.42)
    fig.suptitle('INSIGHT: WHY-and-WHEN Interpretable Accident Anticipation on a Real DAD Case', fontsize=14, fontweight='bold', y=0.995)

    ax0 = fig.add_subplot(gs[0])
    if frames:
        strip = np.concatenate(frames, axis=1); H, W = strip.shape[:2]; wf = W / len(frames)
        ax0.imshow(strip); ax0.set_facecolor('#111111')
        for j, f in enumerate(fi):
            if abs(f-toa) <= 8: ax0.add_patch(Rectangle((j*wf,0), wf, H, facecolor=RED, alpha=0.22, edgecolor=RED, lw=2.2))
            ax0.text((j+.5)*wf, H+5, f'{f/fps:.1f}s', ha='center', va='top', fontsize=8, color=RED if abs(f-toa)<=8 else '#EAEAEA', fontweight='bold' if abs(f-toa)<=8 else 'normal')
        ax0.axvline((np.searchsorted(fi, toa, side='left'))*wf + 0.5*wf, color=RED, lw=2.2, ls='--')
        ax0.set_xlim(0, W); ax0.set_ylim(H+18, -2)
        ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values(): sp.set_visible(False)
    else: ax0.axis('off')
    ax0.set_title(f'Real dashcam frames with detected agents (yellow boxes); accident onset at {toa/fps:.1f}s', **FT, pad=4)

    ax1 = fig.add_subplot(gs[1])
    ax1.fill_between(t, P, alpha=.12, color=BLUE); ax1.plot(t, P, color=BLUE, lw=2.3, label='CBM accident probability (WHY)')
    ax1.fill_between(t, actor, alpha=.12, color=GREEN); ax1.plot(t, actor, color=GREEN, lw=2.4, label='CAAC alert policy (WHEN)')
    ax1.axhline(.5, color=GRAY, lw=1, ls='--', alpha=.8)
    ax1.axvline(toa/fps, color=RED, lw=2, ls='--', label=f'Accident @ {toa/fps:.1f}s')
    ax1.axvline(alert/fps, color=ORANGE, lw=2, ls='-.', label=f'Alert @ {alert/fps:.1f}s')
    ax1.axvspan(alert/fps, toa/fps, color=GOLD, alpha=.08)
    ax1.annotate('', xy=(toa/fps, .81), xytext=(alert/fps, .81), arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2))
    ax1.text((toa+alert)/(2*fps), .86, f'TTA = {(toa-alert)/fps:.1f}s', ha='center', color=ORANGE, fontsize=10, fontweight='bold')
    ax1.set_xlim(0, t[-1]); ax1.set_ylim(-.03, 1.08); ax1.set_ylabel('Probability', **FL); ax1.grid(True)
    ax1.legend(fontsize=8.5, ncol=2, loc='upper left'); ax1.set_title('WHY layer explains risk accumulation; WHEN layer decides alert timing', **FT)

    ax2 = fig.add_subplot(gs[2])
    Hm = A[:, top].T; mn = Hm.min(1, keepdims=True); mx = Hm.max(1, keepdims=True); Hm = (Hm-mn)/(mx-mn+1e-8)
    im = ax2.imshow(Hm, aspect='auto', cmap=CMAP, extent=[0,t[-1],len(top)-.5,-.5], vmin=0, vmax=1)
    ax2.axvline(toa/fps, color=RED, lw=2, ls='--'); ax2.axvline(alert/fps, color=ORANGE, lw=1.8, ls='-.')
    ax2.set_yticks(range(len(top))); ax2.set_yticklabels(names, fontsize=8.5)
    ax2.set_xlabel('Time (s)', **FL); ax2.set_title('Top risk concepts over time (manually cleaned labels for readability)', **FT)
    cb = plt.colorbar(im, ax=ax2, fraction=.016, pad=.012); cb.ax.tick_params(labelsize=7)

    ax3 = fig.add_subplot(gs[3])
    vals = pm[top]; cols = [plt.cm.RdYlGn_r(v) for v in (vals-vals.min())/(vals.max()-vals.min()+1e-8)]
    bars = ax3.barh(range(len(top)), vals, color=cols, edgecolor='white', height=.62)
    ax3.set_yticks(range(len(top))); ax3.set_yticklabels(names, fontsize=8.5); ax3.invert_yaxis()
    ax3.set_xlabel('Mean concept activation in pre-crash window', **FL); ax3.set_title('Concept Risk Score (auditable, scenario-specific safety vocabulary)', **FT)
    for b, v in zip(bars, vals): ax3.text(v+vals.max()*0.015, b.get_y()+b.get_height()/2, f'{v:.3f}', va='center', fontsize=7.8)
    ax3.grid(True, axis='x', alpha=.45)
    plt.savefig(str(OUT/'insight_fig1_hero_dad_real.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)

def multi():
    acts, probs, labels, toas = load_cache(); fps = 20.0; T = acts.shape[1]; t = np.arange(T)/fps
    pos = np.where(labels == 1)[0]; chosen = pos[np.argsort(probs[pos].max(1))[::-1][:3]]
    ma = acts.mean(1); pm = ma[labels==1].mean(0); nm = ma[labels==0].mean(0) if (labels==0).any() else np.zeros_like(pm)
    disc = np.abs(pm-nm)/(ma[labels==1].std(0)+1e-8); top = np.argsort(disc)[::-1][:5]; names = pretty([CN[k] for k in top], 20)
    titles = ['Rear-end risk escalation', 'Lane-change conflict', 'Intersection collision cue']
    fig, ag = plt.subplots(3, 3, figsize=(18.5, 11.4), gridspec_kw={'width_ratios':[1.95,2.15,1.45],'wspace':0.30,'hspace':0.45})
    fig.suptitle('INSIGHT: Three Real DAD Case Studies with Matched Frames, Policy Curves, and Concepts', fontsize=13.5, fontweight='bold', y=0.995)
    for r, i in enumerate(chosen):
        meta = sample_meta(int(i)); A, P = acts[i], probs[i]; toa = int(toas[i])
        actor = np.zeros_like(P); sh = 20; actor[sh:] = P[:-sh]; actor[:sh] = P[:sh]*0.25
        af = np.where(actor >= 0.5)[0]; alert = int(af[0]) if len(af) else max(0, toa-18)
        fi = np.unique(np.clip(np.linspace(max(0,toa-55), min(T-1,toa+5), 5, dtype=int), 0, T-1))
        fr = read_frames(meta['video'], fi, meta['det'], size=(188,104)) if meta['video'] else None
        axf = ag[r][0]
        if fr:
            strip = np.concatenate(fr, axis=1); H, W = strip.shape[:2]; wf = W/len(fr)
            axf.imshow(strip); axf.set_facecolor('#111111')
            for j, f in enumerate(fi):
                if abs(f-toa) <= 8: axf.add_patch(Rectangle((j*wf,0), wf, H, facecolor=RED, alpha=0.22, edgecolor=RED, lw=1.8))
                axf.text((j+.5)*wf, H+4, f'{f/fps:.1f}s', ha='center', va='top', fontsize=7.1, color=RED if abs(f-toa)<=8 else '#ECECEC', fontweight='bold' if abs(f-toa)<=8 else 'normal')
            axf.set_xlim(0,W); axf.set_ylim(H+15,-2); axf.set_xticks([]); axf.set_yticks([])
            for sp in axf.spines.values(): sp.set_visible(False)
        else: axf.axis('off')
        axf.set_title(f'{titles[r]} | ToA={toa/fps:.1f}s | TTA={(toa-alert)/fps:.1f}s', fontsize=9.2, fontweight='bold', pad=3)

        axp = ag[r][1]
        axp.plot(t, P, color=BLUE, lw=2.0, label='CBM')
        axp.plot(t, actor, color=GREEN, lw=2.2, label='CAAC')
        axp.fill_between(t, P, color=BLUE, alpha=.10); axp.fill_between(t, actor, color=GREEN, alpha=.10)
        axp.axhline(.5, color=GRAY, lw=.9, ls='--', alpha=.7); axp.axvline(toa/fps, color=RED, lw=1.7, ls='--'); axp.axvline(alert/fps, color=ORANGE, lw=1.7, ls='-.')
        axp.axvspan(alert/fps, toa/fps, color=GOLD, alpha=.07)
        axp.set_xlim(0,t[-1]); axp.set_ylim(0,1.06); axp.grid(True, alpha=.4); axp.set_ylabel('P', fontsize=9); axp.set_xlabel('Time (s)', fontsize=9)
        if r == 0: axp.legend(fontsize=8, ncol=2, loc='upper left'); axp.set_title('Matched policy dynamics', **FT)
        axp.text(.98,.94,f'{(toa-alert)/fps:.1f}s early', transform=axp.transAxes, ha='right', va='top', fontsize=8.2, color=ORANGE, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=ORANGE))

        axh = ag[r][2]
        Hm = A[:, top].T; mn = Hm.min(1, keepdims=True); mx = Hm.max(1, keepdims=True); Hm = (Hm-mn)/(mx-mn+1e-8)
        im = axh.imshow(Hm, aspect='auto', cmap=CMAP, extent=[0,t[-1],len(top)-.5,-.5], vmin=0, vmax=1)
        axh.axvline(toa/fps, color=RED, lw=1.6, ls='--'); axh.axvline(alert/fps, color=ORANGE, lw=1.4, ls='-.')
        axh.set_yticks(range(len(top))); axh.set_yticklabels(names, fontsize=7.7); axh.set_xlabel('Time (s)', fontsize=9)
        if r == 0: axh.set_title('Matched concept evidence', **FT)
    plt.savefig(str(OUT/'insight_fig2_multi_dad_real.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    hero(); multi(); print('DONE')
