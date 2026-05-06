#!/usr/bin/env python3
"""visualize_cgta_attention.py — CGTA Attention Matrix Visualization"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
ROOT       = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT  = CRASH_ROOT / 'data'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru  import LFCRASH_CBM_GRU
from src.data_loader import CrashDataset

OUT = ROOT / 'output' / 'paper_figures_v2'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman','DejaVu Serif'],
    'font.size': 11, 'axes.facecolor': '#FAFAFA', 'figure.facecolor': '#FFFFFF',
    'axes.spines.top': False, 'axes.spines.right': False,
})
CMAP_ATTN = LinearSegmentedColormap.from_list('attn',['#F7FBFF','#9ECAE1','#3182BD','#08306B'])
CMAP_ACT  = LinearSegmentedColormap.from_list('act', ['#FFFFFF','#FFF3CD','#FF7F0E','#D62728'])


def collate_fn(batch):
    xs,ys,toas=zip(*batch)
    xs=torch.from_numpy(np.stack(xs)).float()
    ys=torch.from_numpy(np.stack(ys)).float()
    tf=[float(t[0]) if hasattr(t,'__len__') else float(t) for t in toas]
    return xs,ys,torch.tensor(tf)


def load_cgta_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    has_cgta = any('cgta' in k for k in state.keys())
    print(f'  has_cgta={has_cgta}, epoch={ckpt.get("epoch","?")}, AP={ckpt.get("AP",0):.4f}')
    model = LFCRASH_CBM_GRU(
        x_dim=4096, h_dim=256, z_dim=512, n_layers=2,
        n_obj=19, n_frames=50, fps=10.0, with_saa=True,
        num_concepts=837, concept_file=None,
        lambda_align=1e-4, lambda_sparse=5e-5, lambda_recon=1e-4,
        use_cbm=True, device=str(device), legacy=not has_cgta,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, has_cgta


@torch.no_grad()
def extract_sample(model, x_np, device):
    x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
    B,T = 1, x.shape[1]
    h   = torch.zeros(2, B, model.h_dim, device=device)
    h_history, attn_rows = [], []
    probs, c_acts, c_deltas = [], [], []
    prev_c = None; hh_rsd = []
    for t in range(T):
        frame = x[:,t]; feats = model.phi_x(frame)
        img_emb = feats[:,0]; obj_emb = feats[:,1:]
        if model.use_cbm:
            c_act, c_embed = model.cbm(img_emb)
        else:
            c_act  = img_emb.new_zeros(B, model.cbm.num_concepts)
            c_embed = img_emb
        delta_c = (c_act - prev_c) if prev_c is not None else torch.zeros_like(c_act)
        prev_c  = c_act.detach()
        obj_vec = model.ofa(obj_emb, h).squeeze(1)
        fi      = model.fft_in(img_emb) if model.fft_in is not None else img_emb
        fft_vec = model.fft_out(model.fft_block(fi.unsqueeze(-1)).mean(1))
        if not model._legacy and len(h_history) > 0:
            h_stack = torch.stack(h_history, dim=1)
            q = model.cgta_q(delta_c).unsqueeze(1)
            k = model.cgta_k(h_stack)
            raw = torch.bmm(q, k.transpose(1,2)) / (model.h_dim**0.5)
            aw  = F.softmax(raw, dim=-1)[0,0].cpu().numpy()
            v   = model.cgta_v(h_stack)
            ctx = torch.bmm(F.softmax(raw,dim=-1), v).squeeze(1)
            cgta_ctx = torch.tanh(model.cgta_gate) * ctx
            rw = torch.sigmoid(model.concept_risk_w)
            rs = (c_act * rw).sum(1, keepdim=True)
            rf = model.crs_proj(rs)
            gru_in = torch.cat([obj_vec, c_embed, fft_vec, cgta_ctx+rf], 1).unsqueeze(1)
        else:
            aw = np.ones(max(1,t)) / max(1,t)
            if not model._legacy:
                gru_in = torch.cat([obj_vec, c_embed, fft_vec,
                                    img_emb.new_zeros(B, model.h_dim)], 1).unsqueeze(1)
            else:
                gru_in = torch.cat([obj_vec, c_embed, fft_vec], 1).unsqueeze(1)
        attn_rows.append(aw)
        out_t, h = model.gru(gru_in, h)
        h_history.append(h[-1]); hh_rsd.append(h.detach())
        if len(hh_rsd) >= 10 and (t+1)%10 == 0:
            h = model._apply_rsd(hh_rsd[-10:])
        probs.append(F.softmax(out_t,dim=-1)[0,1].item())
        c_acts.append(c_act[0].cpu().numpy())
        c_deltas.append(delta_c[0].cpu().numpy())
    T2 = len(probs)
    A  = np.zeros((T2, T2))
    for t, aw in enumerate(attn_rows):
        n = min(len(aw), t+1)
        A[t,:n] = aw[:n]
    return A, np.array(probs), np.stack(c_acts), np.stack(c_deltas)


def load_concepts():
    p = '/data/sony/LFCRASH/000_all_concept_set.txt'
    lines = [l.strip() for l in open(p) if l.strip()]
    cleaned = []
    for s in lines:
        for pre in ('A photo of a ','A photo of ','Photo of a ','Photo of '):
            if s.lower().startswith(pre.lower()): s = s[len(pre):]; break
        cleaned.append(s.rstrip('.').capitalize())
    return cleaned


def plot_and_save(A, probs, c_acts, c_deltas, toa_f, cnames, fps, idx, ds_name):
    T   = len(probs)
    tax = np.arange(T) / fps
    pre = slice(max(0,toa_f-5), min(T,toa_f+1))
    score = np.abs(c_deltas[pre]).mean(0) * c_acts[pre].mean(0)
    top6  = np.argsort(score)[::-1][:6]

    fig = plt.figure(figsize=(16,13), facecolor='white')
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[2.2,2.2,2.4], hspace=0.45, wspace=0.30)

    # Attention matrix (lower-triangular, causal)
    ax0 = fig.add_subplot(gs[0,0])
    mask = np.tril(np.ones_like(A))
    A_masked = np.where(mask.astype(bool), A, np.nan)
    im0 = ax0.imshow(A_masked, aspect='auto', cmap=CMAP_ATTN,
                     origin='upper', vmin=0,
                     extent=[0, tax[-1], tax[-1], 0])
    if toa_f < T:
        ax0.axvline(toa_f/fps, color='#E74C3C', lw=1.8, ls='--', alpha=0.8)
        ax0.axhline(toa_f/fps, color='#E74C3C', lw=1.8, ls='--', alpha=0.8)
    ax0.set_xlabel('Past Timestep (s)', fontsize=10)
    ax0.set_ylabel('Current Timestep (s)', fontsize=10)
    ax0.set_title('CGTA Attention Matrix\nattn[t,s] = frame t attending to past frame s',
                  fontsize=11, fontweight='bold')
    plt.colorbar(im0, ax=ax0, shrink=0.7, label='Attention Weight')
    ax0.text(0.98, 0.02, f'Crash @ {toa_f/fps:.1f}s',
             transform=ax0.transAxes, ha='right', fontsize=9,
             color='#E74C3C', fontweight='bold')

    # Prediction probability curve
    ax1 = fig.add_subplot(gs[0,1])
    ax1.fill_between(tax, probs, alpha=0.15, color='#C0392B')
    ax1.plot(tax, probs, color='#C0392B', lw=2.5, label='P(accident)')
    ax1.axhline(0.5, color='#7F7F7F', lw=1.2, ls='--', label='Threshold')
    if toa_f < T:
        ax1.axvline(toa_f/fps, color='#E67E22', lw=2, ls=':', label=f'Crash @ {toa_f/fps:.1f}s')
    ax1.set_xlim(0, tax[-1]); ax1.set_ylim(0, 1.08)
    ax1.set_xlabel('Time (s)', fontsize=10); ax1.set_ylabel('P(accident)', fontsize=10)
    ax1.set_title('Accident Prediction Confidence', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper left'); ax1.grid(True, alpha=0.4)

    # Concept activation heatmap
    ax2 = fig.add_subplot(gs[1,:])
    heat = c_acts[:, top6].T  # (6, T)
    rmin = heat.min(1,keepdims=True); rmax = heat.max(1,keepdims=True)
    heat_n = (heat-rmin)/(rmax-rmin+1e-8)
    im2 = ax2.imshow(heat_n, aspect='auto', cmap=CMAP_ACT,
                     extent=[0,tax[-1],5.5,-0.5], vmin=0, vmax=1)
    labs = [cnames[i][:55]+('\u2026' if len(cnames[i])>55 else '') for i in top6]
    ax2.set_yticks(range(6)); ax2.set_yticklabels(labs, fontsize=8.5)
    if toa_f < T: ax2.axvline(toa_f/fps, color='#E67E22', lw=2, ls=':', zorder=5)
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_title('Top-6 Concept Activations (selected by pre-crash delta score)',
                  fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.6, pad=0.01, label='Norm. Activation')

    # Concept delta heatmap (dynamics)
    ax3 = fig.add_subplot(gs[2,:])
    delt = c_deltas[:, top6].T
    dmax = np.abs(delt).max() + 1e-8
    im3  = ax3.imshow(delt, aspect='auto', cmap='RdBu_r',
                      extent=[0,tax[-1],5.5,-0.5], vmin=-dmax, vmax=dmax)
    ax3.set_yticks(range(6)); ax3.set_yticklabels(labs, fontsize=8.5)
    if toa_f < T: ax3.axvline(toa_f/fps, color='#E67E22', lw=2, ls=':', zorder=5)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_title('Concept Activation Dynamics \u0394c_t (CGTA Query Signal)',
                  fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=ax3, shrink=0.6, pad=0.01, label='\u0394 Activation')

    fig.suptitle(
        f'CG-CRASH: CGTA Attention \u0026 Concept Dynamics [{ds_name.upper()} Sample #{idx}]\n'
        f'Crash onset @ {toa_f/fps:.1f}s | Peak P(accident)={max(probs):.3f}',
        fontsize=13, fontweight='bold', y=1.01
    )
    out = OUT / f'fig6_cgta_attention_{ds_name}.pdf'
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [\u2713] {out.name}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=6)
    ap.add_argument('--ckpt', default='output/cgta_crs_training/crash_cgta/best_model.pt')
    ap.add_argument('--n_try', type=int, default=80)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model from {args.ckpt}...')
    model, has_cgta = load_cgta_model(str(ROOT/args.ckpt), device)
    cnames = load_concepts()

    ds   = CrashDataset(str(DATA_ROOT/'crash'), 'vgg16', phase='test', toTensor=False)
    ld   = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    fps  = 10.0

    print(f'Scanning {args.n_try} samples for best crash case...')
    best = None
    for i, (x, y, toa) in enumerate(ld):
        if i >= args.n_try: break
        if y[0,1].item() < 0.5: continue
        A, probs, acts, deltas = extract_sample(model, x[0].numpy(), device)
        peak = probs.max()
        if best is None or peak > best['peak']:
            best = dict(i=i, A=A, probs=probs, acts=acts, deltas=deltas,
                        toa=int(toa[0].item()), peak=peak)
        if peak > 0.95: break
    if best is None: print('No crash sample found'); return
    print(f'  Best sample idx={best["i"]} peak={best["peak"]:.4f} toa={best["toa"]}')
    plot_and_save(best['A'], best['probs'], best['acts'], best['deltas'],
                  best['toa'], cnames, fps, best['i'], 'crash')
    print('Done.')


if __name__ == '__main__':
    main()
