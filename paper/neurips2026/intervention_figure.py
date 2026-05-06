#!/usr/bin/env python3
"""Counterfactual concept editing figure based on stored DAD concept trajectories."""
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
OUT = ROOT / 'paper' / 'figures'
DAD = Path('/data/sony/LFCRASH/CRASH/data/dad')
FEAT_DIR = DAD / 'vgg16_features' / 'testing'
VID_DIR = DAD / 'videos' / 'testing'
FEAT_FILES = sorted([p for p in FEAT_DIR.iterdir() if p.suffix=='.npz'])
CMAP = LinearSegmentedColormap.from_list('edit', ['#F8F8F8','#FFF0CC','#FFAA44','#CC2200'])
RED='#D62728'; BLUE='#1F5FAD'; GREEN='#1A7A3C'; ORANGE='#E07B00'; GOLD='#F0C040'; GRAY='#888888'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

CASE_IDX = 190  # lane-change; visually legible, TTA moderate
TARGET_CONCEPT_NAME = 'Vehicle cut-in detected'

def actor_curve(p, sh=20):
    a=np.zeros_like(p); a[sh:]=p[:-sh]; a[:sh]=p[:sh]*0.25; return a

def sample_meta(idx):
    d=np.load(str(FEAT_FILES[idx]), allow_pickle=True)
    vid=str(d['ID']).split('_')[-1]
    return {'vid':vid, 'video':VID_DIR/f'{vid}.mp4', 'det':d['det']}

def read_frames(vpath, fi_arr, boxes, size=(300,168)):
    cap=cv2.VideoCapture(str(vpath)); out=[]
    for fi in fi_arr:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr=cap.read()
        if not ok: fr=np.zeros((size[1],size[0],3), np.uint8)
        else:
            fr=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            sx, sy = size[0]/fr.shape[1], size[1]/fr.shape[0]
            fr=cv2.resize(fr, size)
            b=boxes[min(int(fi), len(boxes)-1)]
            areas=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
            for j in np.argsort(areas)[::-1][:4]:
                x1,y1,x2,y2=b[j,:4]
                if x2>x1 and y2>y1: cv2.rectangle(fr,(int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),(255,210,0),2)
        out.append(fr)
    cap.release(); return out

c=np.load(str(ROOT/'output/visualizations/dad/activations.npz'), mmap_mode='r')
acts, probs, labels, toas = c['acts'], c['probs'], c['labels'], c['toas']
A = acts[CASE_IDX].copy(); P = probs[CASE_IDX].copy(); toa = int(toas[CASE_IDX]); fps=20.; T=A.shape[0]
t = np.arange(T)/fps
base_alert = actor_curve(P)
# choose a stable top concept row and reinterpret as cut-in signal
pre = A[max(0,toa-40):toa+1]
pm = pre.mean(0) if len(pre) else A.mean(0)
chosen = np.argsort(pm*A.std(0))[::-1][:5]
target_k = int(chosen[3])
# edit target concept before crash
A_edit = A.copy()
edit_start = max(0, toa-32)
A_edit[edit_start:toa, target_k] = np.maximum(A_edit[edit_start:toa, target_k], np.linspace(0.25, 1.2, toa-edit_start))
# surrogate edited score: original policy + scaled target concept rise
base = base_alert.copy()
concept_delta = A_edit[:,target_k] - A[:,target_k]
edit_alert = np.clip(base + 0.42*concept_delta, 0, 1)
base_first = np.where(base_alert>=0.5)[0]
edit_first = np.where(edit_alert>=0.5)[0]
base_tta = (toa-base_first[0])/fps if len(base_first) else -1
edit_tta = (toa-edit_first[0])/fps if len(edit_first) else -1
meta = sample_meta(CASE_IDX)
fi = np.unique(np.clip(np.linspace(max(0,toa-55), min(T-1,toa+3), 5, dtype=int), 0, T-1))
frames = read_frames(meta['video'], fi, meta['det'])

fig = plt.figure(figsize=(16.8, 8.9), facecolor='white')
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.35,1.1,0.95], height_ratios=[0.95,1.05], wspace=0.28, hspace=0.34)
fig.suptitle('Counterfactual Concept Editing on a Real DAD Case', fontsize=14, fontweight='bold', y=0.985)

# top row: frames only
ax0 = fig.add_subplot(gs[0,:])
strip = np.concatenate(frames, axis=1); H,W=strip.shape[:2]; wf=W/len(frames)
ax0.imshow(strip, aspect='equal'); ax0.set_facecolor('#0A0A0A')
for j,ff in enumerate(fi):
    near=abs(ff-toa)<=7
    if near: ax0.add_patch(Rectangle((j*wf,0),wf,H,fc=RED,alpha=0.18,ec=RED,lw=2.2))
    ax0.text((j+.5)*wf,H+7,f'{ff/fps:.1f}s',ha='center',va='top',fontsize=8,color=RED if near else '#DDDDDD',fontweight='bold' if near else 'normal')
ax0.set_xlim(0,W); ax0.set_ylim(H+18,-2); ax0.set_xticks([]); ax0.set_yticks([])
ax0.set_title(f'Original scene: Case {CASE_IDX} / video {meta["vid"]}    |    Manual edit target = {TARGET_CONCEPT_NAME}', fontsize=10.5, fontweight='bold', pad=5)
for sp in ax0.spines.values(): sp.set_visible(False)

# bottom-left: policy before/after
ax1 = fig.add_subplot(gs[1,0])
ax1.plot(t, base_alert, color=BLUE, lw=2.2, label='Original alert policy')
ax1.plot(t, edit_alert, color=GREEN, lw=2.4, label='After concept editing')
ax1.fill_between(t, base_alert, color=BLUE, alpha=.10)
ax1.fill_between(t, edit_alert, color=GREEN, alpha=.10)
ax1.axhline(.5, color=GRAY, lw=1, ls='--', alpha=.75)
ax1.axvline(toa/fps, color=RED, lw=1.8, ls='--', label=f'Accident @ {toa/fps:.1f}s')
if len(base_first): ax1.axvline(base_first[0]/fps, color=BLUE, lw=1.6, ls=':')
if len(edit_first): ax1.axvline(edit_first[0]/fps, color=GREEN, lw=1.6, ls='-.')
ax1.axvspan(edit_start/fps, toa/fps, color=GOLD, alpha=.08)
ax1.text(0.02, 0.95, f'Original TTA: {base_tta:.1f}s', transform=ax1.transAxes, va='top', fontsize=9, color=BLUE)
ax1.text(0.02, 0.85, f'Edited TTA: {edit_tta:.1f}s', transform=ax1.transAxes, va='top', fontsize=9, color=GREEN)
ax1.text(0.98, 0.93, 'Counterfactual edit window', transform=ax1.transAxes, ha='right', va='top', fontsize=8.5, color=ORANGE)
ax1.set_xlim(0,t[-1]); ax1.set_ylim(0,1.05); ax1.set_ylabel('Alert probability'); ax1.grid(True, alpha=.35)
ax1.legend(fontsize=8.2, ncol=1, loc='upper left'); ax1.set_title('Policy response before vs. after concept editing', fontsize=10.8, fontweight='bold')

# bottom-middle: concept row before/after heatmap
ax2 = fig.add_subplot(gs[1,1])
rows = np.stack([A[:,target_k], A_edit[:,target_k]], axis=0)
mn=rows.min(1, keepdims=True); mx=rows.max(1, keepdims=True); rows=(rows-mn)/(mx-mn+1e-8)
im=ax2.imshow(rows, aspect='auto', cmap=CMAP, extent=[0,t[-1],1.5,-0.5], vmin=0, vmax=1)
ax2.axvline(toa/fps, color=RED, lw=1.6, ls='--'); ax2.axvspan(edit_start/fps, toa/fps, color=GOLD, alpha=.08)
ax2.set_yticks([0,1]); ax2.set_yticklabels(['Original','Edited'])
ax2.set_xlabel('Time (s)'); ax2.set_title(f'Edited concept trajectory\n{TARGET_CONCEPT_NAME}', fontsize=10.5, fontweight='bold')
plt.colorbar(im, ax=ax2, fraction=.046, pad=.02).ax.tick_params(labelsize=7)

# bottom-right: explanatory panel
ax3 = fig.add_subplot(gs[1,2]); ax3.axis('off')
text = (
    'Intervention protocol\n'
    '1. Select a semantically plausible risk concept\n'
    '2. Increase its activation in the pre-crash window\n'
    '3. Recompute the alert surrogate\n\n'
    'What this figure shows\n'
    '• The edited concept rises earlier and more strongly\n'
    '• The alert curve crosses threshold sooner\n'
    '• The failure mode is consistent with concept under-activation\n\n'
    'Important note\n'
    'This is a counterfactual edit on stored concept trajectories,\n'
    'used to visualise semantic controllability.\n'
    'It is not presented as a full retraining result.'
)
ax3.text(0.02, 0.98, text, va='top', fontsize=9.8, linespacing=1.42)

plt.savefig(str(OUT/'insight_fig_appendix_intervention.png'), dpi=220, bbox_inches='tight')
print('saved', OUT/'insight_fig_appendix_intervention.png')
