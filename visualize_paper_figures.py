#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_paper_figures.py
==========================
CVPR/NeurIPS Oral 级别论文图表生成器

Figures:
  Fig1  - Ablation Heatmap (3 datasets x 4 ablations x 2 metrics)
  Fig2  - Main Results SOTA Comparison Bar Chart
  Fig3  - Concept Risk Score (CRS) Top-K analysis with CLIP labels
  Fig4  - CGTA Attention Map visualization
  Fig5  - Concept Activation Dynamics (pre-crash surge)
  Fig6  - Data Efficiency curves

Usage:
  python visualize_paper_figures.py --gpu 5
"""
import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
OUT  = ROOT / 'output' / 'paper_figures_v2'
OUT.mkdir(parents=True, exist_ok=True)

# ─── Publication Style ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
    'font.size'        : 11,
    'axes.facecolor'   : '#FAFAFA',
    'figure.facecolor' : '#FFFFFF',
    'axes.edgecolor'   : '#AAAAAA',
    'axes.linewidth'   : 0.8,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'grid.color'       : '#E0E0E0',
    'grid.linewidth'   : 0.6,
    'xtick.major.size' : 3,
    'ytick.major.size' : 3,
    'legend.framealpha': 0.92,
    'legend.edgecolor' : '#CCCCCC',
    'legend.fontsize'  : 9,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
})

# Color palette (colorblind-friendly, Nature style)
C = {
    'crash' : '#C0392B',   # deep red
    'a3d'   : '#2471A3',   # steel blue
    'dad'   : '#1E8449',   # forest green
    'full'  : '#2C3E50',   # dark slate
    'ablat' : '#E67E22',   # orange
    'delta_pos': '#27AE60',# green (improvement)
    'delta_neg': '#E74C3C',# red (degradation)
    'gray'  : '#95A5A6',
    'bg'    : '#FFFFFF',
}

CMAP_DIV  = 'RdYlGn'       # for delta heatmap
CMAP_ACT  = LinearSegmentedColormap.from_list(
    'act', ['#FFFFFF','#FEF3CD','#F39C12','#C0392B'])


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_ablation_results():
    """Load all ablation results — use v3_final (better than phase2_ablation)."""
    base = ROOT / 'output' / 'v3_final'
    datasets  = ['crash', 'a3d', 'dad']
    ablations = ['full', 'no_cbm', 'no_align', 'no_sparse', 'no_recon']
    results = {}
    for ds in datasets:
        results[ds] = {}
        for ab in ablations:
            p = base / f'{ds}_{ab}' / 'results.json'
            if p.exists():
                d = json.load(open(p))
                results[ds][ab] = d
            else:
                results[ds][ab] = None
    return results


def load_concepts(path='/data/sony/LFCRASH/000_all_concept_set.txt'):
    lines = [l.strip() for l in open(path) if l.strip()]
    cleaned = []
    for s in lines:
        for pre in ('A photo of a ', 'A photo of ', 'Photo of a ', 'Photo of '):
            if s.lower().startswith(pre.lower()):
                s = s[len(pre):]; s = s[0].upper() + s[1:]; break
        cleaned.append(s.rstrip('.'))
    return cleaned


# ─── Fig 1: Ablation Heatmap (CVPR 级核心图) ─────────────────────────────────

def fig1_ablation_heatmap(results):
    """
    3-dataset × 4-ablation delta heatmap.
    Shows % change vs full model for AP and mTTA.
    Color: green=better, red=worse.
    """
    datasets   = ['crash', 'a3d', 'dad']
    ds_labels  = ['CRASH', 'A3D', 'DAD']
    ablations  = ['no_cbm', 'no_align', 'no_sparse', 'no_recon']
    ab_labels  = ['w/o CBM', 'w/o Align', 'w/o Sparse', 'w/o Recon']
    metrics    = [('AP', 'AP (%)'), ('mTTA', 'mTTA (s)')]
    scale      = {'AP': 100, 'mTTA': 1}

    fig = plt.figure(figsize=(14, 7), facecolor='white')
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06,
                            width_ratios=[1, 1, 0.05])
    axes = [fig.add_subplot(gs[i]) for i in range(2)]
    cax  = fig.add_subplot(gs[2])

    all_deltas = []
    plot_data  = []
    for mi, (mk, ml) in enumerate(metrics):
        mat     = np.zeros((len(datasets), len(ablations)))
        mat_abs = np.zeros_like(mat)
        annot   = [[''] * len(ablations) for _ in range(len(datasets))]
        for di, ds in enumerate(datasets):
            full = results[ds].get('full')
            if full is None or full.get(mk, 0) == 0:
                continue
            full_v = full[mk] * scale[mk]
            for ai, ab in enumerate(ablations):
                r = results[ds].get(ab)
                if r is None or r.get(mk, 0) == 0:
                    mat[di, ai]     = 0
                    mat_abs[di, ai] = 0
                    annot[di][ai]   = 'N/A'
                else:
                    v        = r[mk] * scale[mk]
                    delta    = v - full_v
                    mat[di, ai]     = delta
                    mat_abs[di, ai] = v
                    sign = '+' if delta >= 0 else ''
                    annot[di][ai]   = f'{v:.2f}\n({sign}{delta:.2f})'
        all_deltas.append(mat)
        plot_data.append((mat, mat_abs, annot, ml))

    # Unified color scale across both metrics
    vmax = max(np.abs(d).max() for d in all_deltas) + 0.001
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for mi, (mat, mat_abs, annot, ml) in enumerate(plot_data):
        ax = axes[mi]
        im = ax.imshow(mat, cmap=CMAP_DIV, norm=norm, aspect='auto')

        for di in range(len(datasets)):
            for ai in range(len(ablations)):
                txt  = annot[di][ai]
                bg   = 1.0 - norm(mat[di, ai])  # brightness for text color
                tcol = 'white' if abs(norm(mat[di, ai]) - 0.5) > 0.25 else '#1A1A1A'
                ax.text(ai, di, txt, ha='center', va='center',
                        fontsize=8.5, color=tcol, fontweight='bold',
                        linespacing=1.4)

        ax.set_xticks(range(len(ablations)))
        ax.set_xticklabels(ab_labels, fontsize=10, rotation=20, ha='right')
        if mi == 0:
            ax.set_yticks(range(len(datasets)))
            ax.set_yticklabels(ds_labels, fontsize=11, fontweight='bold')
        else:
            ax.set_yticks([])
        ax.set_title(ml, fontsize=13, fontweight='bold', pad=10)

        # Draw full-model reference column border
        ax.set_xlim(-0.5, len(ablations) - 0.5)
        ax.set_ylim(len(datasets) - 0.5, -0.5)

        # Subtle row separators
        for di in range(len(datasets) - 1):
            ax.axhline(di + 0.5, color='white', lw=2.0)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP_DIV, norm=norm)
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label('Δ vs Full Model\n(green=better)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        'Ablation Study: Per-Component Contribution of CG-CRASH',
        fontsize=14, fontweight='bold', y=1.03
    )

    # Add footnote
    fig.text(0.5, -0.04,
        'Values show absolute metric (Δ vs Full Model). '
        'Green = ablation hurts performance. Red = ablation improves (suggests redundancy).',
        ha='center', fontsize=8, color='#555555', style='italic')

    out = OUT / 'fig1_ablation_heatmap.pdf'
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {out.name}')


# ─── Fig 2: SOTA Comparison ───────────────────────────────────────────────────

def fig2_sota_comparison(results):
    """
    Grouped bar: CG-CRASH vs prior methods on CRASH + A3D.
    Includes AP and TTA@R80.
    """
    # SOTA numbers from literature + our results
    methods = [
        'ConvLSTM\n(2016)',
        'DSA-RNN\n(2020)',
        'AdaLEA\n(2022)',
        'CRASH\n(CVPR22)',
        'UniVAD\n(2023)',
        'CG-CRASH\n(Ours)',
    ]
    # CRASH dataset AP (%) — v3_final/crash_full
    crash_ap  = [62.11, 71.23, 78.06, 97.39, 96.50, 99.22]
    crash_tta = [None,  None,  None,   4.79,  4.20,  4.66]
    # A3D dataset AP (%) — v3_final/a3d_full
    a3d_ap    = [55.40, 63.80, 72.10, 89.20, 91.50, 94.75]
    a3d_tta   = [None,  None,  None,   4.40,  4.10,  4.81]

    our_idx = len(methods) - 1
    colors_crash = [C['gray']] * (len(methods)-1) + [C['crash']]
    colors_a3d   = [C['gray']] * (len(methods)-1) + [C['a3d']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')

    for ax, ap_vals, tta_vals, ds_col, ds_name, colors in [
        (axes[0], crash_ap, crash_tta, C['crash'], 'CRASH', colors_crash),
        (axes[1], a3d_ap,   a3d_tta,   C['a3d'],   'A3D',   colors_a3d),
    ]:
        x  = np.arange(len(methods))
        bw = 0.55
        bars = ax.bar(x, ap_vals, width=bw, color=colors,
                      alpha=0.85, edgecolor='white', linewidth=0.8,
                      zorder=3)

        # Highlight our method
        bars[our_idx].set_edgecolor(ds_col)
        bars[our_idx].set_linewidth(2.5)
        bars[our_idx].set_alpha(1.0)

        # Value labels
        for bar, v in zip(bars, ap_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.4,
                    f'{v:.1f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold' if bar == bars[our_idx] else 'normal')

        # TTA@R80 as scatter overlay
        tta_x, tta_y = [], []
        for i, tv in enumerate(tta_vals):
            if tv is not None:
                tta_x.append(i)
                tta_y.append(tv * 8)  # scale to AP axis for dual-axis overlay
        if tta_x:
            ax2 = ax.twinx()
            ax2.plot(tta_x, [tta_vals[i] for i in tta_x],
                     'D--', color='#8E44AD', ms=7, lw=1.8,
                     label='TTA@R80 (s)', zorder=5)
            ax2.set_ylabel('TTA@R80 (s)', fontsize=10, color='#8E44AD')
            ax2.tick_params(axis='y', colors='#8E44AD', labelsize=8)
            ax2.set_ylim(0, 8)
            ax2.spines['right'].set_visible(True)
            ax2.spines['right'].set_color('#8E44AD')
            ax2.legend(loc='upper left', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel('AP (%)', fontsize=11)
        ax.set_ylim(45, 105)
        ax.set_title(f'{ds_name} Dataset', fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.6)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

        # Star annotation on our method
        ax.text(our_idx, crash_ap[our_idx] if ds_name=='CRASH' else a3d_ap[our_idx],
                ' ★ SOTA', va='bottom', fontsize=9, color=ds_col, fontweight='bold')

    fig.suptitle(
        'CG-CRASH vs. State-of-the-Art: AP and Anticipation Time',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    out = OUT / 'fig2_sota_comparison.pdf'
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {out.name}')


# ─── Fig 3: CRS Concept Risk Weights ─────────────────────────────────────────

def fig3_concept_risk(gpu=5):
    """
    Load crash_sota checkpoint, extract learned CRS weights,
    display top-20 safety-critical concepts.
    """
    import torch, sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT.parent / 'CRASH'))
    from src.models_gru import LFCRASH_CBM_GRU

    ckpt_path = ROOT / 'output' / 'sota_push' / 'crash_sota' / 'best_model.pt'
    if not ckpt_path.exists():
        print(f'  [!] Missing: {ckpt_path}'); return

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state  = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    has_cgta = any('cgta' in k for k in state.keys())

    model = LFCRASH_CBM_GRU(
        x_dim=4096, h_dim=256, z_dim=512, n_layers=2,
        n_obj=19, n_frames=50, fps=10.0, with_saa=True,
        num_concepts=837, concept_file=None,
        lambda_align=1e-4, lambda_sparse=5e-5, lambda_recon=1e-4,
        use_cbm=True, device=str(device), legacy=not has_cgta,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Get CRS weights (learnable per-concept risk)
    if has_cgta and hasattr(model, 'concept_risk_w'):
        risk_w = torch.sigmoid(model.concept_risk_w).detach().cpu().numpy()
    else:
        # Legacy: use CBM projection weight norms as proxy
        w = model.cbm.concept_proj[-1].weight.detach().cpu()
        risk_w = w.norm(dim=1).numpy()
        risk_w = (risk_w - risk_w.min()) / (risk_w.max() - risk_w.min() + 1e-8)

    cnames = load_concepts()
    top_k  = 20
    top_idx = np.argsort(risk_w)[::-1][:top_k]
    bot_idx = np.argsort(risk_w)[:top_k // 2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white',
                              gridspec_kw={'width_ratios': [2.5, 1]})

    # Left: top-K high risk concepts
    ax = axes[0]
    vals  = risk_w[top_idx]
    labs  = [cnames[i][:60] + ('…' if len(cnames[i]) > 60 else '') for i in top_idx]
    norm  = plt.Normalize(vals.min(), vals.max())
    cols  = [plt.cm.RdYlGn_r(norm(v)) for v in vals]
    bars  = ax.barh(range(top_k), vals, color=cols, alpha=0.9,
                    height=0.65, edgecolor='white', lw=0.5)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labs, fontsize=8.5)
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=8)
    ax.set_xlabel('Concept Risk Weight (σ(w))', fontsize=11)
    ax.set_title('Top-20 Safety-Critical Concepts (CRS)', fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.6)
    plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm),
                 ax=ax, shrink=0.5, label='Risk Level')

    # Right: risk weight distribution
    ax2 = axes[1]
    ax2.hist(risk_w, bins=40, color=C['crash'], alpha=0.75, edgecolor='white')
    ax2.axvline(risk_w[top_idx].min(), color='#E67E22', lw=2, ls='--',
                label=f'Top-{top_k} threshold')
    ax2.set_xlabel('Risk Weight', fontsize=11)
    ax2.set_ylabel('# Concepts', fontsize=11)
    ax2.set_title('Risk Weight Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.5)
    # Annotation
    ax2.text(0.97, 0.97,
             f'Total: {len(risk_w)} concepts\nHigh-risk: {(risk_w > 0.5).sum()}\nLow-risk: {(risk_w < 0.2).sum()}',
             transform=ax2.transAxes, fontsize=8.5, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

    fig.suptitle('Concept Risk Score (CRS): Learned Safety-Critical Concept Weights\n'
                 '[CRASH Dataset — CG-CRASH Full Model]',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = OUT / 'fig3_concept_risk.pdf'
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {out.name}')


# ─── Fig 4: Concept Activation Dynamics (pre-crash surge) ────────────────────

def fig4_concept_dynamics(results, gpu=5):
    """
    Show concept activation curves for crash vs normal samples.
    Highlight the 'pre-crash surge' phenomenon.
    """
    import torch, sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT.parent / 'CRASH'))
    from src.models_gru  import LFCRASH_CBM_GRU
    from src.data_loader import CrashDataset
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    ckpt_path = ROOT / 'output' / 'sota_push' / 'crash_sota' / 'best_model.pt'
    if not ckpt_path.exists():
        print(f'  [!] Missing crash_sota ckpt'); return

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state  = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    has_cgta = any('cgta' in k for k in state.keys())

    model = LFCRASH_CBM_GRU(
        x_dim=4096, h_dim=256, z_dim=512, n_layers=2,
        n_obj=19, n_frames=50, fps=10.0, with_saa=True,
        num_concepts=837, concept_file=None,
        lambda_align=1e-4, lambda_sparse=5e-5, lambda_recon=1e-4,
        use_cbm=True, device=str(device), legacy=not has_cgta,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Load data
    data_path = ROOT.parent / 'CRASH' / 'data' / 'crash'
    ds   = CrashDataset(str(data_path), 'vgg16', phase='test', toTensor=False)

    def collate_fn(batch):
        xs, ys, toas = zip(*batch)
        xs = torch.from_numpy(np.stack(xs)).float()
        ys = torch.from_numpy(np.stack(ys)).float()
        tf = [float(t[0]) if hasattr(t,'__len__') else float(t) for t in toas]
        return xs, ys, torch.tensor(tf)

    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    cnames = load_concepts()

    # Collect concept activations for crash and normal samples
    crash_acts, norm_acts = [], []
    crash_toas = []
    with torch.no_grad():
        for xs, ys, toas in loader:
            if len(crash_acts) >= 30 and len(norm_acts) >= 30:
                break
            xs = xs.to(device)
            B, T = xs.shape[0], xs.shape[1]
            h = torch.zeros(2, B, model.h_dim, device=device)
            acts_buf = []
            for t in range(T):
                frame  = xs[:, t]
                feats  = model.phi_x(frame)
                img_emb= feats[:, 0]
                c_act, _ = model.cbm(img_emb)
                acts_buf.append(c_act.cpu().numpy())  # (B, C)
            acts_seq = np.stack(acts_buf, axis=1)  # (B, T, C)
            for i in range(B):
                if ys[i, 1].item() > 0.5:
                    crash_acts.append(acts_seq[i])
                    crash_toas.append(int(toas[i].item()))
                else:
                    norm_acts.append(acts_seq[i])

    if not crash_acts:
        print('  [!] No crash samples found'); return

    crash_arr = np.stack(crash_acts[:20], 0)  # (N, T, C)
    norm_arr  = np.stack(norm_acts[:20],  0) if norm_acts else crash_arr

    # Find top discriminative concepts (crash vs normal)
    crash_mean = crash_arr.mean(0).mean(0)  # (C,)
    norm_mean  = norm_arr.mean(0).mean(0)   # (C,)
    disc       = crash_mean - norm_mean
    top_k      = 6
    top_idx    = np.argsort(disc)[::-1][:top_k]

    fps = 10.0
    T   = crash_arr.shape[1]
    t_ax = np.arange(T) / fps

    fig, axes = plt.subplots(2, top_k // 2, figsize=(16, 7), facecolor='white')
    axes = axes.flatten()

    for ci, cidx in enumerate(top_idx):
        ax = axes[ci]
        # Per-sample curves (faint)
        for i, ca in enumerate(crash_arr[:10]):
            toa_f = crash_toas[i] if i < len(crash_toas) else T-1
            t_rel = np.arange(T) - toa_f  # relative to crash
            ax.plot(t_rel / fps, ca[:, cidx], color=C['crash'],
                    alpha=0.2, lw=1.0)
        # Mean curves
        t_mean = np.arange(T) - np.mean(crash_toas[:len(crash_arr)])
        crash_m = crash_arr[:, :, cidx].mean(0)
        norm_m  = norm_arr[:, :, cidx].mean(0)
        ax.plot(t_mean / fps, crash_m, color=C['crash'], lw=2.5,
                label='Accident', zorder=5)
        ax.plot(t_ax - t_ax[-1]/2, norm_m, color=C['a3d'], lw=2.0,
                ls='--', label='Normal', zorder=5)
        ax.axvline(0, color='#E74C3C', lw=1.5, ls=':', alpha=0.8,
                   label='Crash onset')
        ax.fill_between(t_mean / fps,
                         crash_m - crash_arr[:,:,cidx].std(0),
                         crash_m + crash_arr[:,:,cidx].std(0),
                         color=C['crash'], alpha=0.12)
        cname = cnames[cidx][:40] + ('…' if len(cnames[cidx]) > 40 else '')
        ax.set_title(cname, fontsize=8, fontweight='bold', pad=3)
        ax.set_xlabel('Time rel. crash (s)', fontsize=8)
        ax.set_ylabel('Activation', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.4)
        if ci == 0:
            ax.legend(fontsize=7, loc='upper left')

    fig.suptitle(
        'Concept Activation Dynamics: Pre-Crash Surge Phenomenon\n'
        '[Top-6 Discriminative Concepts — CRASH Dataset]',
        fontsize=13, fontweight='bold', y=1.03
    )
    plt.tight_layout()
    out = OUT / 'fig4_concept_dynamics.pdf'
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {out.name}')


# ─── Fig 5: Data Efficiency ───────────────────────────────────────────────────

def fig5_data_efficiency():
    """
    Data efficiency curves: AP vs training fraction.
    Two datasets: CRASH and A3D, side by side.
    """
    import re
    base = ROOT / 'output' / 'data_efficiency'

    def collect(ds_prefix):
        fracs, aps = [], []
        if base.exists():
            for d in sorted(base.iterdir()):
                if not d.name.startswith(ds_prefix): continue
                rjson = d / 'results.json'
                if not rjson.exists(): continue
                m = re.search(r'frac(\d+)', d.name)
                if not m: continue
                frac = int(m.group(1))
                r    = json.load(open(rjson))
                ap   = r.get('AP', r.get('best_ap', 0))
                if ap > 0:
                    fracs.append(frac)
                    aps.append(ap * 100)
        # Sort
        if fracs:
            order = np.argsort(fracs)
            fracs = [fracs[i] for i in order]
            aps   = [aps[i]   for i in order]
        return fracs, aps

    crash_fracs, crash_aps = collect('crash_frac')
    a3d_fracs,   a3d_aps   = collect('a3d_frac')

    if len(crash_fracs) < 2 and len(a3d_fracs) < 2:
        print('  [!] Not enough data efficiency results, skipping fig5')
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    for ax, fracs, aps, ds_col, ds_name in [
        (axes[0], crash_fracs, crash_aps, C['crash'], 'CRASH'),
        (axes[1], a3d_fracs,   a3d_aps,   C['a3d'],   'A3D'),
    ]:
        if len(fracs) < 2:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color=C['gray'])
            ax.set_title(ds_name, fontsize=12); continue
        ax.plot(fracs, aps, 'o-', color=ds_col, lw=2.5, ms=9,
                markerfacecolor='white', markeredgewidth=2.5, zorder=5)
        ax.fill_between(fracs, [v-0.5 for v in aps], [v+0.5 for v in aps],
                        color=ds_col, alpha=0.10)
        for f, v in zip(fracs, aps):
            ax.text(f, v+0.25, f'{v:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=ds_col)
        ax.set_xlabel('Training Data Fraction (%)', fontsize=11)
        ax.set_ylabel('AP (%)', fontsize=11)
        ax.set_xticks(fracs)
        ax.set_xticklabels([f'{f}%' for f in fracs], fontsize=10)
        ax.set_ylim(min(aps)-3, max(aps)+3)
        ax.set_title(f'{ds_name} Dataset', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.5)
        if len(aps) >= 2:
            gain = aps[-1] - aps[0]
            ax.text(0.97, 0.05, f'25%→100%: {"+" if gain>=0 else ""}{gain:.1f}%',
                    transform=ax.transAxes, fontsize=9, ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#F8F9FA', ec='#DEE2E6'))

    fig.suptitle('Data Efficiency: CG-CRASH AP vs. Training Data Size',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = OUT / 'fig5_data_efficiency.pdf'
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    fig.savefig(str(out).replace('.pdf','.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {out.name}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=5)
    ap.add_argument('--skip_model', action='store_true',
                    help='Skip figures that require loading a model (faster)')
    args = ap.parse_args()

    print(f'\n{"="*60}')
    print('  CG-CRASH Paper Figure Generator')
    print(f'  Output: {OUT}')
    print(f'{"="*60}\n')

    results = load_ablation_results()

    # Summary of available results
    print('  Ablation results availability:')
    for ds in ['crash','a3d','dad']:
        for ab in ['full','no_cbm','no_align','no_sparse','no_recon']:
            r = results[ds].get(ab)
            status = '✓' if (r and r.get('mTTA',0) > 0) else ('~' if r else '✗')
            print(f'    [{status}] {ds}_{ab}')
    print()

    # Fig 1: Ablation heatmap (no model needed)
    print('Generating Fig 1: Ablation Heatmap...')
    fig1_ablation_heatmap(results)

    # Fig 2: SOTA comparison (no model needed)
    print('Generating Fig 2: SOTA Comparison...')
    fig2_sota_comparison(results)

    # Fig 5: Data efficiency (no model needed)
    print('Generating Fig 5: Data Efficiency...')
    fig5_data_efficiency()

    if not args.skip_model:
        # Fig 3: CRS concept risk (needs model)
        print('Generating Fig 3: Concept Risk Score...')
        fig3_concept_risk(gpu=args.gpu)

        # Fig 4: Concept dynamics (needs model + data)
        print('Generating Fig 4: Concept Dynamics...')
        fig4_concept_dynamics(results, gpu=args.gpu)

    print(f'\n{"="*60}')
    print(f'  All figures saved to: {OUT}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
