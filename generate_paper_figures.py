#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_paper_figures.py  -  CG-CRASH publication figure generator
Outputs (PDF + PNG 300dpi):
  fig1_main_results    Main results vs baselines
  fig2_ablation        Full ablation (3 datasets x 4 conditions)
  fig3_concept_disc    Top discriminative concepts per dataset
  fig4_training_curves AP training curves
  fig5_cgta_diagram    CGTA + CRS architecture diagram
Usage:
  python generate_paper_figures.py [--out_dir output/paper_figures]
"""
import json, argparse, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ---- Palette (IBM Carbon, colorblind-safe) -----------------------------------
C = dict(blue='#0f62fe', teal='#009d9a', purple='#8a3ffc',
         red='#da1e28', orange='#ff832b', green='#198038',
         gray='#8d8d8d', black='#161616', bg='#ffffff', bg2='#f4f4f4',
         border='#e0e0e0')
DS_COL  = {'CRASH': C['blue'], 'A3D': C['teal'], 'DAD': C['purple']}
ABL_COL = {'full': C['blue'], 'no_cbm': C['orange'], 'no_align': C['teal'],
           'no_sparse': C['purple'], 'no_recon': C['red']}
ABL_LBL = {'full': 'Full (CG-CRASH)', 'no_cbm': 'No-CBM',
            'no_align': 'No-Align', 'no_sparse': 'No-Sparse',
            'no_recon': 'No-Recon'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.facecolor': '#ffffff', 'figure.facecolor': '#ffffff',
    'axes.edgecolor': '#e0e0e0', 'axes.labelcolor': '#161616',
    'xtick.color': '#525252', 'ytick.color': '#525252',
    'text.color': '#161616', 'grid.color': '#e8e8e8',
    'grid.linestyle': '-', 'grid.alpha': 0.7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'legend.fontsize': 10, 'legend.framealpha': 0.95,
    'legend.edgecolor': '#e0e0e0',
})

# ---- Data -------------------------------------------------------------------
V3 = {
    'crash': {
        'full':      {'AP': 99.22, 'mTTA': 4.662, 'TTA_R80': 4.263, 'P_R80': 99.33},
        'no_align':  {'AP': 99.20, 'mTTA': 4.797, 'TTA_R80': 4.613, 'P_R80': 99.33},
        'no_cbm':    {'AP': 99.54, 'mTTA': 4.152, 'TTA_R80': 3.503, 'P_R80': 99.65},
        'no_recon':  {'AP': 99.21, 'mTTA': 4.468, 'TTA_R80': 3.846, 'P_R80': 99.32},
        'no_sparse': {'AP': 99.50, 'mTTA': 4.410, 'TTA_R80': 3.636, 'P_R80': 99.65},
    },
    'a3d': {
        'full':      {'AP': 94.75, 'mTTA': 4.808, 'TTA_R80': 4.444, 'P_R80': 94.86},
        'no_align':  {'AP': 96.03, 'mTTA': 4.267, 'TTA_R80': 3.700, 'P_R80': 93.40},
        'no_cbm':    {'AP': 93.47, 'mTTA': 4.606, 'TTA_R80': 3.941, 'P_R80': 93.55},
        'no_recon':  {'AP': 93.38, 'mTTA': 4.660, 'TTA_R80': 4.070, 'P_R80': 93.43},
        'no_sparse': {'AP': 94.38, 'mTTA': 4.514, 'TTA_R80': 3.439, 'P_R80': 94.50},
    },
    'dad': {
        'full':      {'AP': 63.20, 'mTTA': 1.722, 'TTA_R80': 2.637, 'P_R80': 45.99},
        'no_align':  {'AP': 62.63, 'mTTA': 1.900, 'TTA_R80': 3.086, 'P_R80': 44.25},
        'no_cbm':    {'AP': 65.07, 'mTTA': 2.313, 'TTA_R80': 3.120, 'P_R80': 46.67},
        'no_recon':  {'AP': 62.86, 'mTTA': 2.028, 'TTA_R80': 2.748, 'P_R80': 44.37},
        'no_sparse': {'AP': 66.05, 'mTTA': 2.016, 'TTA_R80': 2.420, 'P_R80': 49.22},
    },
}

BASELINES = {
    'DSA-RNN':          {'AP': 71.23, 'mTTA': 2.88},
    'ConvLSTM':         {'AP': 62.11, 'mTTA': 2.45},
    'AdaLEA':           {'AP': 78.06, 'mTTA': 3.21},
    'CRASH (original)': {'AP': 97.39, 'mTTA': 4.79},
    'CG-CRASH (ours)':  {'AP': 99.22, 'mTTA': 4.66},
}


def load_history(v3_dir, dataset, condition):
    p = Path(v3_dir) / f'{dataset}_{condition}' / 'history.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _save(fig, out_dir, name):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(p / f'{name}.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='#ffffff', edgecolor='none')
    plt.close(fig)
    print(f'  [ok] {name}.pdf + .png')


# ---- Fig 1: Main Results ----------------------------------------------------
def fig1_main_results(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('#ffffff')

    # Left: AP vs baselines on CRASH
    ax = axes[0]
    methods = list(BASELINES.keys())
    aps     = [BASELINES[m]['AP'] for m in methods]
    colors  = [C['gray']] * (len(methods) - 1) + [C['blue']]
    bars = ax.barh(methods, aps, color=colors, height=0.55,
                   edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, aps):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{v:.1f}%', va='center', fontsize=10,
                fontweight='bold' if v == max(aps) else 'normal',
                color=C['blue'] if v == max(aps) else C['black'])
    ax.set_xlim(50, 104)
    ax.set_xlabel('Average Precision (AP %)')
    ax.set_title('CRASH Dataset: AP vs Prior Methods',
                 fontsize=13, fontweight='bold', pad=8)
    ax.axvline(97.39, color=C['gray'], lw=1.5, ls='--', alpha=0.7,
               label='CRASH baseline (97.39%)')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='x', alpha=0.5)
    ax.invert_yaxis()

    # Right: Three-dataset overview
    ax2 = axes[1]
    datasets = ['CRASH', 'A3D', 'DAD']
    ds_keys  = ['crash', 'a3d', 'dad']
    aps_f  = [V3[k]['full']['AP']   for k in ds_keys]
    mttas  = [V3[k]['full']['mTTA'] for k in ds_keys]
    cols2  = [DS_COL[d] for d in datasets]
    x  = np.arange(3)
    bw = 0.38
    b1 = ax2.bar(x - bw/2, aps_f, bw, color=cols2, alpha=0.88,
                 edgecolor='white', linewidth=0.5, label='AP (%)')
    ax2r = ax2.twinx()
    b2 = ax2r.bar(x + bw/2, mttas, bw, color=cols2, alpha=0.45,
                  edgecolor='white', linewidth=0.5, hatch='//', label='mTTA (s)')
    for bar, v in zip(b1, aps_f):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                 f'{v:.1f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    for bar, v in zip(b2, mttas):
        ax2r.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                  f'{v:.1f}s', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=12)
    ax2.set_ylabel('AP (%)')
    ax2r.set_ylabel('mTTA (s)')
    ax2.set_ylim(0, 112)
    ax2r.set_ylim(0, 8)
    ax2.set_title('CG-CRASH: Three-Dataset Results',
                  fontsize=13, fontweight='bold', pad=8)
    h1 = mpatches.Patch(color=C['gray'], alpha=0.9, label='AP (%)')
    h2 = mpatches.Patch(color=C['gray'], alpha=0.45, hatch='//', label='mTTA (s)')
    ax2.legend(handles=[h1, h2], fontsize=10, loc='upper right')

    plt.tight_layout(pad=1.5)
    _save(fig, out_dir, 'fig1_main_results')


# ---- Fig 2: Ablation --------------------------------------------------------
def fig2_ablation(out_dir):
    conds     = ['full', 'no_cbm', 'no_align', 'no_sparse', 'no_recon']
    c_labels  = ['Full\n(CG-CRASH)', 'No-CBM', 'No-Align', 'No-Sparse', 'No-Recon']
    metrics   = ['AP', 'mTTA', 'TTA_R80', 'P_R80']
    m_labels  = ['AP (%)', 'mTTA (s)', 'TTA@R80 (s)', 'P@R80 (%)']
    datasets  = ['crash', 'a3d', 'dad']
    ds_labels = ['CRASH', 'A3D', 'DAD']

    fig, axes = plt.subplots(len(datasets), len(metrics),
                             figsize=(16, 10), sharex='col')
    fig.patch.set_facecolor('#ffffff')

    x = np.arange(len(conds))
    colors = [ABL_COL[c] for c in conds]

    for di, (ds, dsl) in enumerate(zip(datasets, ds_labels)):
        for mi, (met, mlab) in enumerate(zip(metrics, m_labels)):
            ax = axes[di][mi]
            vals = [V3[ds][c][met] for c in conds]
            bars = ax.bar(x, vals, color=colors, alpha=0.85,
                          edgecolor='white', linewidth=0.5, width=0.65)
            bars[0].set_edgecolor(C['blue'])
            bars[0].set_linewidth(2.0)
            ymax = max(vals)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + ymax * 0.01, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=7.5, color='#333333')
            ax.set_ylim(min(vals) * 0.92, max(vals) * 1.10)
            ax.grid(True, axis='y', alpha=0.4)
            if di == 0:
                ax.set_title(mlab, fontsize=11, fontweight='bold', pad=6)
            if mi == 0:
                ax.set_ylabel(dsl, fontsize=12, fontweight='bold',
                              color=DS_COL[dsl])
            if di == len(datasets) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(c_labels, fontsize=8)
            else:
                ax.set_xticks([])

    handles = [mpatches.Patch(color=ABL_COL[c], label=ABL_LBL[c]) for c in conds]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=10, bbox_to_anchor=(0.5, -0.03),
               framealpha=0.95, edgecolor='#e0e0e0')
    fig.suptitle('CG-CRASH: Ablation Study (Each Condition Trained from Scratch)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(pad=1.2, h_pad=0.8, w_pad=0.6)
    _save(fig, out_dir, 'fig2_ablation')


# ---- Fig 3: Concept Discriminability ----------------------------------------
def fig3_concept_disc(out_dir, v2_dir):
    concept_dir = Path(v2_dir) / 'concept_eval'
    datasets  = ['crash', 'a3d', 'dad']
    ds_labels = ['CRASH', 'A3D', 'DAD']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#ffffff')

    for ax, ds, dsl in zip(axes, datasets, ds_labels):
        cfile = concept_dir / f'{ds}_concept_summary.json'
        if not cfile.exists():
            ax.text(0.5, 0.5, f'No data\n({ds})',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color=C['gray'])
            ax.set_title(dsl, fontsize=13, fontweight='bold')
            continue
        with open(cfile) as f:
            cdata = json.load(f)
        top = cdata.get('top_discriminative', [])[:10]
        if not top:
            continue
        names  = [t['concept'][:48] + ('...' if len(t['concept']) > 48 else '')
                  for t in top]
        scores = [t['discriminability'] for t in top]
        norm   = [s / max(scores) for s in scores]
        cols   = plt.cm.RdPu([0.35 + 0.65 * n for n in norm])
        y = np.arange(len(top))
        bars = ax.barh(y, scores, color=cols, height=0.65,
                       edgecolor='white', linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Cohen's d", fontsize=10)
        ax.set_title(f'{dsl}: Top-10 Safety-Critical Concepts',
                     fontsize=12, fontweight='bold', pad=6,
                     color=DS_COL[dsl])
        for bar, v in zip(bars, scores):
            ax.text(v + max(scores) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:.2f}', va='center', fontsize=8)
        ax.grid(True, axis='x', alpha=0.4)

    fig.suptitle('CG-CRASH: Concept-Level Interpretability',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(pad=1.5, w_pad=2.0)
    _save(fig, out_dir, 'fig3_concept_disc')


# ---- Fig 4: Training Curves -------------------------------------------------
def fig4_training_curves(out_dir, v3_dir):
    conds   = ['full', 'no_cbm', 'no_align', 'no_sparse', 'no_recon']
    cols    = [C['blue'], C['orange'], C['teal'], C['purple'], C['red']]
    styles  = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    datasets  = ['crash', 'a3d', 'dad']
    ds_labels = ['CRASH', 'A3D', 'DAD']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#ffffff')

    for ax, ds, dsl in zip(axes, datasets, ds_labels):
        for cond, col, ls in zip(conds, cols, styles):
            h = load_history(v3_dir, ds, cond)
            if h is None:
                continue
            epochs = [r['epoch'] for r in h]
            aps    = [r['test']['AP'] * 100 for r in h]
            ax.plot(epochs, aps, color=col, linestyle=ls,
                    linewidth=2.0, label=ABL_LBL[cond], alpha=0.9)
            best_ep = epochs[int(np.argmax(aps))]
            best_ap = max(aps)
            ax.scatter([best_ep], [best_ap], color=col, s=60, zorder=6,
                       edgecolors='white', linewidths=1.2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AP (%)')
        ax.set_title(dsl, fontsize=13, fontweight='bold',
                     pad=6, color=DS_COL[dsl])
        ax.grid(True, alpha=0.4)
        if ds == 'crash':
            ax.legend(fontsize=8.5, loc='lower right')

    fig.suptitle('CG-CRASH: Training Curves (AP) — All Ablation Conditions',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(pad=1.5, w_pad=1.5)
    _save(fig, out_dir, 'fig4_training_curves')


# ---- Fig 5: CGTA + CRS Architecture Diagram ---------------------------------
def fig5_cgta_diagram(out_dir):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#ffffff')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_axis_off()

    def box(x, y, w, h, label, sub='', color=C['blue'], alpha=0.15, fs=9, sfs=7):
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                              facecolor=color, alpha=alpha,
                              edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        yoff = 0.12 if sub else 0
        ax.text(x + w/2, y + h/2 + yoff, label,
                ha='center', va='center', fontsize=fs,
                fontweight='bold', color=color)
        if sub:
            ax.text(x + w/2, y + h/2 - 0.22, sub,
                    ha='center', va='center', fontsize=sfs, color='#444444')

    def arrow(x1, y1, x2, y2, color='#555555', label='', rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                   connectionstyle=f'arc3,rad={rad}'))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label,
                    ha='center', va='bottom', fontsize=8,
                    color=color, style='italic')

    for t in range(5):
        xt = 0.5 + t * 2.6
        box(xt, 5.5, 2.0, 0.9, f'Frame t+{t}', 'VGG16 (4096-d)',
            color=C['gray'], alpha=0.12)
        box(xt, 4.2, 2.0, 0.9, 'CBM', 'c_t in R^837',
            color=C['teal'], alpha=0.18)
        box(xt, 2.9, 2.0, 0.9, 'GRU', 'h_t in R^256',
            color=C['blue'], alpha=0.18)
        box(xt, 1.6, 2.0, 0.75, f'P(acc|t+{t})',
            color=C['red'], alpha=0.18)
        arrow(xt+1.0, 5.5, xt+1.0, 5.1)
        arrow(xt+1.0, 4.2, xt+1.0, 3.8)
        arrow(xt+1.0, 2.9, xt+1.0, 2.35)

    # CGTA arc
    ax.annotate('', xy=(3.1, 3.35), xytext=(5.7, 4.65),
                arrowprops=dict(arrowstyle='->', color=C['teal'], lw=2.2,
                                connectionstyle='arc3,rad=-0.35',
                                linestyle='dashed'))
    ax.text(4.1, 4.3, 'CGTA (delta_c_t -> attend h_past)',
            ha='center', va='center', fontsize=9,
            color=C['teal'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f4f4',
                      edgecolor=C['teal'], alpha=0.9))

    # CRS
    box(9.0, 0.3, 3.5, 0.9, 'CRS: Concept Risk Score',
        'r = sum(w_c * c_t)  [learnable]',
        color=C['orange'], alpha=0.18, fs=9, sfs=7.5)
    arrow(10.75, 1.6, 10.75, 1.2, color=C['orange'])

    ax.text(7.0, 6.7,
            'CG-CRASH: Concept-Guided Temporal Attention (CGTA) + Concept Risk Score (CRS)',
            ha='center', va='center', fontsize=13,
            fontweight='bold', color=C['black'])
    ax.text(7.0, 6.35,
            'CGTA uses concept activation dynamics to attend over past GRU states '
            '-- linking interpretability directly to prediction.',
            ha='center', va='center', fontsize=9, color='#525252')

    _save(fig, out_dir, 'fig5_cgta_diagram')


# ---- Main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='output/paper_figures')
    ap.add_argument('--v3_dir',  default='output/v3_final')
    ap.add_argument('--v2_dir',  default='output/v2_20260314')
    args = ap.parse_args()

    print('\n=== CG-CRASH Paper Figure Generator ===')
    print(f'Output: {args.out_dir}\n')
    print('Fig 1: Main results...')
    fig1_main_results(args.out_dir)
    print('Fig 2: Ablation study...')
    fig2_ablation(args.out_dir)
    print('Fig 3: Concept discriminability...')
    fig3_concept_disc(args.out_dir, args.v2_dir)
    print('Fig 4: Training curves...')
    fig4_training_curves(args.out_dir, args.v3_dir)
    print('Fig 5: CGTA+CRS diagram...')
    fig5_cgta_diagram(args.out_dir)
    print(f'\n=== ALL DONE -> {args.out_dir}/ ===')


if __name__ == '__main__':
    main()
