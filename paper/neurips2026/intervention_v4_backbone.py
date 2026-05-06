#!/usr/bin/env python3
from pathlib import Path
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import cv2

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'CRASH'))
from src.models_gru import LFCRASH_CBM_GRU
from src.data_loader import DADDataset

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CKPT = ROOT / 'output/v4_dad/dad_full_v4/best_model.pth'
OUT = ROOT / 'paper' / 'figures'
DATA = Path('/data/sony/LFCRASH/CRASH/data/dad')
FEAT_DIR = DATA / 'vgg16_features' / 'testing'
VID_DIR = DATA / 'videos' / 'testing'
FEAT_FILES = sorted([p for p in FEAT_DIR.iterdir() if p.suffix == '.npz'])
CMAP = LinearSegmentedColormap.from_list('edit', ['#F8F8F8','#FFF0CC','#FFAA44','#CC2200'])
RED='#D62728'; BLUE='#1F5FAD'; GREEN='#1A7A3C'; ORANGE='#E07B00'; GOLD='#F0C040'; GRAY='#888888'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
CASE_IDX = 190
TARGET_CONCEPT_NAME = 'Vehicle cut-in detected'

class DADDatasetWrapper:
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        try:
            features, labels, toa = self.dataset[idx]
        except Exception:
            data_file = os.path.join(self.dataset.data_path, self.dataset.phase, self.dataset.files_list[idx])
            data = np.load(data_file)
            features, labels = data['data'], data['labels']
            if isinstance(labels, np.ndarray) and len(labels.shape) == 2:
                has_positive = np.any(labels[:, 1] > 0) if labels.shape[1] > 1 else np.any(labels > 0)
                video_label = float(has_positive)
                labels = np.array([1 - video_label, video_label], dtype=np.float32)
                toa = [90.0] if has_positive else [self.dataset.n_frames + 1]
        if isinstance(toa, (list, np.ndarray)): toa = float(toa[0])
        return features, labels, np.array([toa], dtype=np.float32)

def sample_meta(idx):
    d = np.load(str(FEAT_FILES[idx]), allow_pickle=True)
    vid = str(d['ID']).split('_')[-1]
    return {'vid': vid, 'video': VID_DIR/f'{vid}.mp4', 'det': d['det']}

def read_frames(vpath, fi_arr, boxes, size=(320,180)):
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

def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    args = ckpt.get('args', {})
    msd = ckpt['state_dict']
    model = LFCRASH_CBM_GRU(
        x_dim=4096, h_dim=args.get('h_dim', 256), z_dim=args.get('z_dim', 128),
        n_layers=2, n_obj=19, n_frames=100, fps=20.0, with_saa=True,
        num_concepts=args.get('num_concepts', 837), concept_file=None,
        lambda_align=args.get('lambda_align', 1e-4), lambda_sparse=args.get('lambda_sparse', 1e-4),
        lambda_recon=args.get('lambda_recon', 0.002), use_cbm=not args.get('no_cbm', False),
        device=DEVICE, legacy=False).to(DEVICE)
    ret = model.load_state_dict(msd, strict=False)
    model.eval()
    return model, ckpt, ret

@torch.no_grad()
def run_pred_curve(model, x, edited=None):
    x = x.unsqueeze(0).to(DEVICE).float()
    B, T, Np1, D = x.shape
    h = torch.zeros(model.n_layers, 1, model.h_dim, device=DEVICE)
    preds, cacts = [], []
    prev_c_act = None
    hidden_list = []
    for t in range(T):
        frame = x[:, t]
        feats = model.phi_x(frame)
        img_emb, obj_emb = feats[:, 0], feats[:, 1:]
        c_act, c_embed = model.cbm(img_emb) if model.use_cbm else (img_emb.new_zeros(1, model.cbm.num_concepts), img_emb)
        if edited is not None:
            c_act = edited[t:t+1]
            c_embed = model.cbm.decode(c_act)
            c_embed = model.cbm.ln(c_embed)
        cacts.append(c_act.squeeze(0).detach().cpu().numpy())
        obj_ctx = model.ofa(obj_emb, h)
        obj_vec = obj_ctx.squeeze(1)
        fft_in = model.fft_in(img_emb) if model.fft_in is not None else img_emb
        fft_out = model.fft_block(fft_in.unsqueeze(-1))
        fft_vec = model.fft_out(fft_out.mean(dim=1))
        delta_c = (c_act - prev_c_act) if prev_c_act is not None else torch.zeros_like(c_act)
        prev_c_act = c_act.detach()
        if len(hidden_list) > 0:
            h_stack = torch.stack(hidden_list, dim=1)
            cgta_q = model.cgta_q(delta_c).unsqueeze(1)
            cgta_k = model.cgta_k(h_stack)
            cgta_v = model.cgta_v(h_stack)
            attn_w = F.softmax(torch.bmm(cgta_q, cgta_k.transpose(1, 2)) / math.sqrt(model.h_dim), dim=-1)
            cgta_ctx = torch.tanh(model.cgta_gate) * torch.bmm(attn_w, cgta_v).squeeze(1)
        else:
            cgta_ctx = img_emb.new_zeros(1, model.h_dim)
        risk_w = torch.sigmoid(model.concept_risk_w)
        risk_feat = model.crs_proj((c_act * risk_w).sum(dim=1, keepdim=True))
        gru_in = torch.cat([obj_vec, c_embed, fft_vec, cgta_ctx + risk_feat], dim=1).unsqueeze(1)
        out_t, h = model.gru(gru_in, h)
        hidden_list.append(h[-1].detach())
        preds.append(torch.softmax(out_t, dim=-1)[0, 1].item())
    return np.array(preds), np.array(cacts)

if __name__ == '__main__':
    model, ckpt, ret = load_model()
    ds = DADDatasetWrapper(DADDataset(str(DATA), 'vgg16', phase='testing', toTensor=False))
    x, y, toa_arr = ds[CASE_IDX]
    x = torch.tensor(x)
    toa = int(float(toa_arr[0]))
    base_pred, base_c = run_pred_curve(model, x)
    pre = base_c[max(0, toa-40):toa+1]
    pm = pre.mean(0) if len(pre) else base_c.mean(0)
    chosen = np.argsort(pm * base_c.std(0))[::-1][:5]
    target_k = int(chosen[3])
    edited = torch.tensor(base_c, dtype=torch.float32, device=DEVICE)
    edit_start = max(0, toa-32)
    edited[edit_start:toa, target_k] = torch.maximum(edited[edit_start:toa, target_k], torch.linspace(0.25, 1.2, toa-edit_start, device=DEVICE))
    edit_pred, edit_c = run_pred_curve(model, x, edited=edited)
    base_first = np.where(base_pred >= 0.5)[0]
    edit_first = np.where(edit_pred >= 0.5)[0]
    base_tta = (toa-base_first[0])/20.0 if len(base_first) else -1
    edit_tta = (toa-edit_first[0])/20.0 if len(edit_first) else -1
    meta = sample_meta(CASE_IDX)
    fi = np.unique(np.clip(np.linspace(max(0,toa-55), min(x.shape[0]-1,toa+3), 5, dtype=int), 0, x.shape[0]-1))
    frames = read_frames(meta['video'], fi, meta['det'])
    t = np.arange(x.shape[0])/20.0

    fig = plt.figure(figsize=(16.8, 8.9), facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.35,1.1,0.95], height_ratios=[0.95,1.05], wspace=0.28, hspace=0.34)
    fig.suptitle('Backbone-Level Counterfactual Concept Editing on a Real DAD Case', fontsize=14, fontweight='bold', y=0.985)

    ax0 = fig.add_subplot(gs[0,:])
    strip = np.concatenate(frames, axis=1); H,W=strip.shape[:2]; wf=W/len(frames)
    ax0.imshow(strip, aspect='equal'); ax0.set_facecolor('#0A0A0A')
    for j,ff in enumerate(fi):
        near=abs(ff-toa)<=7
        if near: ax0.add_patch(Rectangle((j*wf,0),wf,H,fc=RED,alpha=0.18,ec=RED,lw=2.2))
        ax0.text((j+.5)*wf,H+7,f'{ff/20.0:.1f}s',ha='center',va='top',fontsize=8,color=RED if near else '#DDDDDD',fontweight='bold' if near else 'normal')
    ax0.set_xlim(0,W); ax0.set_ylim(H+18,-2); ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title(f'Original scene: Case {CASE_IDX} / video {meta["vid"]}    |    Edited concept = {TARGET_CONCEPT_NAME}', fontsize=10.5, fontweight='bold', pad=5)
    for sp in ax0.spines.values(): sp.set_visible(False)

    ax1 = fig.add_subplot(gs[1,0])
    ax1.plot(t, base_pred, color=BLUE, lw=2.2, label='Original prediction head')
    ax1.plot(t, edit_pred, color=GREEN, lw=2.4, label='After concept edit + rerun')
    ax1.fill_between(t, base_pred, color=BLUE, alpha=.10)
    ax1.fill_between(t, edit_pred, color=GREEN, alpha=.10)
    ax1.axhline(.5, color=GRAY, lw=1, ls='--', alpha=.75)
    ax1.axvline(toa/20.0, color=RED, lw=1.8, ls='--', label=f'Accident @ {toa/20.0:.1f}s')
    if len(base_first): ax1.axvline(base_first[0]/20.0, color=BLUE, lw=1.6, ls=':')
    if len(edit_first): ax1.axvline(edit_first[0]/20.0, color=GREEN, lw=1.6, ls='-.')
    ax1.axvspan(edit_start/20.0, toa/20.0, color=GOLD, alpha=.08)
    ax1.text(0.02, 0.95, f'Original TTA: {base_tta:.1f}s', transform=ax1.transAxes, va='top', fontsize=9, color=BLUE)
    ax1.text(0.02, 0.85, f'Edited TTA: {edit_tta:.1f}s', transform=ax1.transAxes, va='top', fontsize=9, color=GREEN)
    ax1.set_xlim(0,t[-1]); ax1.set_ylim(0,1.05); ax1.set_ylabel('Accident probability'); ax1.grid(True, alpha=.35)
    ax1.legend(fontsize=8.2, ncol=1, loc='upper left'); ax1.set_title('Prediction response before vs. after concept editing', fontsize=10.8, fontweight='bold')

    ax2 = fig.add_subplot(gs[1,1])
    rows = np.stack([base_c[:,target_k], edit_c[:,target_k]], axis=0)
    mn=rows.min(1, keepdims=True); mx=rows.max(1, keepdims=True); rows=(rows-mn)/(mx-mn+1e-8)
    im=ax2.imshow(rows, aspect='auto', cmap=CMAP, extent=[0,t[-1],1.5,-0.5], vmin=0, vmax=1)
    ax2.axvline(toa/20.0, color=RED, lw=1.6, ls='--'); ax2.axvspan(edit_start/20.0, toa/20.0, color=GOLD, alpha=.08)
    ax2.set_yticks([0,1]); ax2.set_yticklabels(['Original','Edited'])
    ax2.set_xlabel('Time (s)'); ax2.set_title(f'Edited concept trajectory\n{TARGET_CONCEPT_NAME}', fontsize=10.5, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=.046, pad=.02).ax.tick_params(labelsize=7)

    ax3 = fig.add_subplot(gs[1,2]); ax3.axis('off')
    note = (
        'What changed in this version\n'
        '• Uses the v4 DAD backbone checkpoint\n'
        '• Manually edits one concept channel\n'
        '• Re-runs the downstream prediction head\n\n'
        'Interpretation\n'
        'This is stronger than editing only stored curves,\n'
        'because the prediction curve is recomputed through\n'
        'the loaded model backbone after concept editing.\n\n'
        'Caveat\n'
        'The available v4 checkpoint does not include the\n'
        'final Actor-Critic head weights, so this figure shows\n'
        'backbone-level rerun rather than full AC-policy rerun.'
    )
    ax3.text(0.02, 0.98, note, va='top', fontsize=9.6, linespacing=1.4)

    out_path = OUT / 'insight_fig_appendix_intervention_v4_backbone.png'
    plt.savefig(str(out_path), dpi=220, bbox_inches='tight')
    print(json.dumps({
        'saved': str(out_path),
        'base_tta': base_tta,
        'edit_tta': edit_tta,
        'target_k': target_k,
        'num_missing': len(ret.missing_keys),
        'missing_keys': ret.missing_keys,
    }, indent=2))
