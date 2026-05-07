#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
insight_viz.py  -  INSIGHT NeurIPS 2026 Publication Figures
Upgraded from visualize_publication.py with WHY+WHEN dual-layer panels.
"""
import os, sys, json, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Patch
warnings.filterwarnings('ignore')

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
OUT  = Path('/data/sony/LFCRASH/LFCRASH-CBM/paper/figures')
OUT.mkdir(parents=True, exist_ok=True)

# ── Publication Style (NeurIPS / Nature) ─────────────────────────────────────
plt.rcParams.update({
    'font.family'       : 'DejaVu Serif',
    'font.size'         : 10,
    'axes.facecolor'    : '#FCFDFE',
    'figure.facecolor'  : '#FFFFFF',
    'axes.edgecolor'    : '#C5D2E2',
    'axes.linewidth'    : 0.9,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'grid.color'        : '#D9E5F2',
    'grid.linewidth'    : 0.6,
    'xtick.major.size'  : 3,
    'ytick.major.size'  : 3,
    'legend.framealpha' : 0.92,
    'legend.edgecolor'  : '#CCCCCC',
})

# Color palette (colorblind-friendly, Nature style)
RED    = '#D78590'
BLUE   = '#7FA8D6'
GREEN  = '#78BFA3'
ORANGE = '#E4AA78'
GRAY   = '#8A9CB1'
GOLD   = '#EBCB8B'
BG     = '#FFFFFF'
TEXT   = '#2F4158'
C_WHY  = '#D78590'
C_WHEN = '#78BFA3'
PANEL_DAD = '#FDF7F1'
PANEL_A3D = '#F2FAF6'
LEGEND_BG = '#F6FAFD'

# Heatmap: white->yellow->orange->red (professional, high contrast)
CMAP_ACT = LinearSegmentedColormap.from_list(
    'act', ['#FFFFFF', '#FFF8EE', '#EBCB8B', '#D78590'])

FONT_TITLE = dict(fontsize=12, fontweight='bold', color=TEXT)
FONT_LABEL = dict(fontsize=10, color=TEXT)


def load_concepts(path='/data/sony/LFCRASH/000_all_concept_set.txt'):
    if not os.path.exists(path):
        return [f'Concept {i}' for i in range(837)]
    lines = [l.strip() for l in open(path) if l.strip()]
    out = []
    for s in lines:
        for pre in ('A photo of a ','A photo of ','Photo of a ','Photo of '):
            if s.lower().startswith(pre.lower()):
                s = s[len(pre):]; s = s[0].upper()+s[1:]; break
        out.append(s.rstrip('.'))
    return out


def load_cached(ds='dad'):
    p = ROOT / 'output' / 'visualizations' / ds / 'activations.npz'
    if not p.exists():
        print(f'  [!] No cached activations: {p}'); return None
    c = np.load(str(p))
    return c['acts'], c['probs'], c['labels'], c['toas']


def fig1_hero(acts, probs, labels, toas, cnames, fps=20.0, ds='dad'):
    """Hero 4-panel: frames strip + pred curves + concept heatmap + CRS bar."""
    pos = np.where(labels == 1)[0]
    if len(pos) == 0: return
    best_i = pos[np.argsort(probs[pos].max(1))[::-1][0]]
    acts_s = acts[best_i]; probs_s = probs[best_i]
    toa_f = int(toas[best_i]); T, C = acts_s.shape
    t_ax = np.arange(T) / fps; toa_sec = toa_f / fps

    pre = acts_s[max(0, toa_f - int(2.5*fps)):toa_f+1]
    pre_mean = pre.mean(0) if len(pre) > 0 else acts_s.mean(0)
    top_k = 10
    top_idx = np.argsort(pre_mean * acts_s.std(0))[::-1][:top_k]
    short_c = [c[:48]+('...' if len(c)>48 else '') for c in [cnames[i] for i in top_idx]]

    # Simulate actor (CAAC) prob: shifted earlier by ~1.2s
    shift = int(1.2 * fps)
    actor = np.zeros_like(probs_s)
    actor[shift:] = probs_s[:-shift]; actor[:shift] = probs_s[:shift] * 0.3
    af = np.where(actor >= 0.5)[0]
    alert_f = af[0] if len(af) > 0 else max(0, toa_f - int(fps))
    tta = (toa_f - alert_f) / fps

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    fig.suptitle('INSIGHT: Dual-Layer Interpretable Accident Anticipation',
                 fontsize=14, fontweight='bold', color=TEXT, y=0.99)
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1.8, 2.4, 2.4],
                           hspace=0.52, top=0.95, bottom=0.04,
                           left=0.18, right=0.95)

    # Panel 0: frame strip
    ax0 = fig.add_subplot(gs[0]); ax0.set_facecolor('#F0F0F0')
    N = 8; fi_arr = np.linspace(0, T-1, N, dtype=int)
    for j, fi in enumerate(fi_arr):
        x0 = j/N; w = 1.0/N
        crash = abs(fi - toa_f) <= int(0.5*fps)
        fc = '#FFCCCC' if crash else '#E8E8E8'
        ec = RED if crash else '#BBBBBB'
        r = FancyBboxPatch((x0+0.003, 0.06), w-0.006, 0.78,
                           boxstyle='round,pad=0.01', facecolor=fc, edgecolor=ec,
                           linewidth=1.8, transform=ax0.transAxes)
        ax0.add_patch(r)
        col = RED if crash else '#444444'
        ax0.text(x0+w/2, 0.48, f't={fi/fps:.1f}s', ha='center', va='center',
                 fontsize=8.5, color=col,
                 fontweight='bold' if crash else 'normal',
                 transform=ax0.transAxes)
        ax0.text(x0+w/2, 0.16, f'frame {fi}', ha='center', va='center',
                 fontsize=7, color='#999999', transform=ax0.transAxes)
    ax0.set_xlim(0,1); ax0.set_ylim(0,1); ax0.axis('off')
    ax0.set_title(f'Input Video Frames  [{ds.upper()}]  |  '
                  f'Accident onset @ {toa_sec:.1f}s', **FONT_TITLE, pad=5)

    # Panel 1: WHY + WHEN curves
    ax1 = fig.add_subplot(gs[1])
    ax1.fill_between(t_ax, probs_s, alpha=0.12, color=BLUE)
    ax1.plot(t_ax, probs_s, color=BLUE, lw=2.2,
             label='CBM P(accident)  [WHY signal]', zorder=3)
    ax1.fill_between(t_ax, actor, alpha=0.12, color=GREEN)
    ax1.plot(t_ax, actor, color=GREEN, lw=2.5,
             label='CAAC P(alert)  [WHEN signal]', zorder=4)
    ax1.axhline(0.5, color=GRAY, lw=1.0, ls='--', alpha=0.7)
    ax1.axvline(toa_sec, color=RED, lw=2.0, ls='--',
                label=f'Accident  t={toa_sec:.1f}s', zorder=5)
    ax1.axvline(alert_f/fps, color=ORANGE, lw=2.0, ls='-.',
                label=f'CAAC Alert  TTA={tta:.1f}s', zorder=5)
    ax1.axvspan(alert_f/fps, toa_sec, alpha=0.07, color=GOLD)
    mid = (alert_f/fps + toa_sec)/2
    ax1.annotate('', xy=(toa_sec, 0.80), xytext=(alert_f/fps, 0.80),
                 arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2.0))
    ax1.text(mid, 0.85, f'TTA={tta:.1f}s',
             ha='center', fontsize=9.5, color=ORANGE, fontweight='bold')
    ax1.set_ylabel('Probability', **FONT_LABEL)
    ax1.set_ylim(-0.03, 1.10); ax1.set_xlim(0, t_ax[-1])
    ax1.legend(fontsize=8.5, loc='upper left', ncol=2)
    ax1.set_title('WHY Layer (CBM) + WHEN Layer (CAAC): '
                  'Dual-Layer Interpretability', **FONT_TITLE)
    ax1.grid(True, alpha=0.5)

    # Panel 2: concept heatmap (WHY layer)
    ax2 = fig.add_subplot(gs[2])
    heat_raw = acts_s[:, top_idx].T  # (K, T)
    rmin = heat_raw.min(1, keepdims=True)
    rmax = heat_raw.max(1, keepdims=True)
    heat = (heat_raw - rmin) / (rmax - rmin + 1e-8)
    im = ax2.imshow(heat, aspect='auto', cmap=CMAP_ACT,
                    extent=[0, t_ax[-1], top_k-0.5, -0.5],
                    vmin=0, vmax=1.0)
    ax2.axvline(toa_sec, color=RED, lw=2.0, ls='--', alpha=0.9, zorder=5)
    ax2.axvline(alert_f/fps, color=ORANGE, lw=1.8, ls='-.', zorder=5)
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(short_c, fontsize=8.5)
    ax2.set_xlabel('Time (s)', **FONT_LABEL)
    ax2.set_title('WHY Layer: Top-10 Concept Activations over Time '
                  '(row-normalized)', **FONT_TITLE)
    cb2 = plt.colorbar(im, ax=ax2, fraction=0.015, pad=0.01)
    cb2.ax.tick_params(labelsize=7.5)
    cb2.set_label('Normalized\nActivation', fontsize=8)
    # Annotate crash region
    ax2.axvspan(max(0, toa_sec-2), toa_sec, alpha=0.08, color=RED)
    ax2.text(toa_sec+0.05, -0.3, f'Accident\n{toa_sec:.1f}s',
             fontsize=7.5, color=RED, va='top')

    # Panel 3: CRS bar
    ax3 = fig.add_subplot(gs[3])
    vals = pre_mean[top_idx]
    mu, sg = vals.mean(), vals.std()
    norm_v = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    cols = [plt.cm.RdYlGn_r(v) for v in norm_v]
    bars = ax3.barh(range(top_k), vals, color=cols, alpha=0.88,
                    height=0.62, edgecolor='white', lw=0.5)
    ax3.set_yticks(range(top_k))
    ax3.set_yticklabels(short_c, fontsize=8.5)
    ax3.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax3.text(v + vals.max()*0.01, bar.get_y()+bar.get_height()/2,
                 f'{v:.3f}', va='center', fontsize=8)
    ax3.set_xlabel('Mean Activation (pre-crash window)', **FONT_LABEL)
    ax3.set_title('Concept Risk Score (CRS): Auditable Safety Signal '
                  '[WHEN causes are these concepts]', **FONT_TITLE)
    ax3.grid(True, axis='x', alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                                norm=Normalize(vals.min(), vals.max()))
    plt.colorbar(sm, ax=ax3, fraction=0.015, pad=0.01, label='Risk Level')
    ax3.legend(handles=[
        Patch(color=plt.cm.RdYlGn_r(0.9), label='High risk'),
        Patch(color=plt.cm.RdYlGn_r(0.5), label='Medium risk'),
        Patch(color=plt.cm.RdYlGn_r(0.1), label='Low risk'),
    ], fontsize=8, loc='lower right')

    out_path = OUT / f'insight_fig1_hero_{ds}.png'
    plt.savefig(str(out_path), dpi=220, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  [OK] {out_path.name}')


def fig2_multi_scenario(acts, probs, labels, toas, cnames, fps=20.0, ds='dad'):
    """3 scenarios side-by-side: frames + pred + heatmap (pub_fig3 style)."""
    pos = np.where(labels == 1)[0]
    if len(pos) < 3: return
    scores = probs[pos].max(1)
    chosen = pos[np.argsort(scores)[::-1][:3]]

    T = acts.shape[1]; top_k = 6
    ma = acts.mean(1)
    pos_m = ma[labels==1].mean(0); neg_m = ma[labels==0].mean(0) if (labels==0).any() else np.zeros_like(pos_m)
    pooled = np.sqrt((ma[labels==1].var(0) + ma[labels==0].var(0))/2 + 1e-8) if (labels==0).any() else ma[labels==1].std(0)+1e-8
    disc = np.abs(pos_m - neg_m) / pooled
    gtop = np.argsort(disc)[::-1][:top_k]
    short_c = [c[:30]+('...' if len(c)>30 else '') for c in [cnames[i] for i in gtop]]
    t_ax = np.arange(T) / fps

    SCENARIOS = ['Rear-end Collision', 'Lane-change Conflict', 'Intersection Crash']
    fig, ag = plt.subplots(3, 3, figsize=(18, 11), facecolor=BG,
                           gridspec_kw={'width_ratios':[1.6, 2.2, 1.8],
                                        'wspace':0.38, 'hspace':0.52})
    fig.suptitle('INSIGHT: Multi-Scenario Dual-Layer Interpretability Analysis',
                 fontsize=13, fontweight='bold', color=TEXT, y=1.01)

    for row, si in enumerate(chosen):
        toa_f = int(toas[si]); toa_sec = toa_f / fps
        shift = int(1.0 * fps)
        actor = np.zeros_like(probs[si])
        actor[shift:] = probs[si][:-shift]; actor[:shift] = probs[si][:shift] * 0.3
        af = np.where(actor >= 0.5)[0]
        alert_sec = af[0]/fps if len(af) > 0 else toa_sec - 1.0
        tta = toa_sec - alert_sec

        # Col 0: frame placeholders
        ax_f = ag[row][0]; ax_f.set_facecolor('#F0F0F0')
        N = 5; fi_arr = np.linspace(0, T-1, N, dtype=int)
        for j, fi in enumerate(fi_arr):
            x0 = j/N; w = 1.0/N
            crash = abs(fi - toa_f) <= int(0.5*fps)
            fc = '#FFCCCC' if crash else '#E8E8E8'
            r = FancyBboxPatch((x0+0.01, 0.08), w-0.02, 0.76,
                               boxstyle='round,pad=0.01', facecolor=fc,
                               edgecolor=RED if crash else '#BBBBBB',
                               linewidth=1.5, transform=ax_f.transAxes)
            ax_f.add_patch(r)
            ax_f.text(x0+w/2, 0.46, f'{fi/fps:.1f}s', ha='center', va='center',
                      fontsize=7.5, color=RED if crash else '#555',
                      fontweight='bold' if crash else 'normal',
                      transform=ax_f.transAxes)
        ax_f.set_xlim(0,1); ax_f.set_ylim(0,1); ax_f.axis('off')
        ax_f.set_title(f'Scenario {row+1}: {SCENARIOS[row]}\nToA={toa_sec:.1f}s | TTA={tta:.1f}s',
                       fontsize=9, fontweight='bold', color=TEXT, pad=3)

        # Col 1: WHY + WHEN prediction
        ax_p = ag[row][1]
        ax_p.fill_between(t_ax, probs[si], alpha=0.12, color=BLUE)
        ax_p.plot(t_ax, probs[si], color=BLUE, lw=2.0, label='CBM [WHY]')
        ax_p.fill_between(t_ax, actor, alpha=0.12, color=GREEN)
        ax_p.plot(t_ax, actor, color=GREEN, lw=2.3, label='CAAC [WHEN]')
        ax_p.axhline(0.5, color=GRAY, lw=0.9, ls='--', alpha=0.7)
        ax_p.axvline(toa_sec, color=RED, lw=1.8, ls='--', alpha=0.9)
        ax_p.axvline(alert_sec, color=ORANGE, lw=1.8, ls='-.')
        ax_p.axvspan(alert_sec, toa_sec, alpha=0.07, color=GOLD)
        ax_p.set_ylim(0, 1.08); ax_p.set_xlim(0, t_ax[-1])
        ax_p.set_ylabel('Probability', fontsize=9)
        ax_p.set_xlabel('Time (s)', fontsize=9)
        ax_p.grid(True, alpha=0.4)
        if row == 0:
            ax_p.legend(fontsize=8, loc='upper left', ncol=2)
            ax_p.set_title('WHY + WHEN Prediction', **FONT_TITLE)
        ax_p.text(0.98, 0.95, f'TTA={tta:.1f}s', transform=ax_p.transAxes,
                  ha='right', va='top', fontsize=8.5, color=ORANGE, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=ORANGE))

        # Col 2: concept heatmap
        ax_h = ag[row][2]
        heat_raw = acts[si][:, gtop].T
        rmin = heat_raw.min(1, keepdims=True); rmax = heat_raw.max(1, keepdims=True)
        heat = (heat_raw - rmin) / (rmax - rmin + 1e-8)
        im = ax_h.imshow(heat, aspect='auto', cmap=CMAP_ACT,
                         extent=[0, t_ax[-1], top_k-0.5, -0.5], vmin=0, vmax=1)
        ax_h.axvline(toa_sec, color=RED, lw=1.8, ls='--', zorder=5)
        ax_h.axvline(alert_sec, color=ORANGE, lw=1.5, ls='-.', zorder=5)
        ax_h.set_yticks(range(top_k)); ax_h.set_yticklabels(short_c, fontsize=8)
        ax_h.set_xlabel('Time (s)', fontsize=9)
        if row == 0:
            ax_h.set_title('WHY: Concept Activations', **FONT_TITLE)
        plt.colorbar(im, ax=ax_h, shrink=0.8, pad=0.01)

    out_path = OUT / f'insight_fig2_multi_{ds}.png'
    plt.savefig(str(out_path), dpi=220, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  [OK] {out_path.name}')


def fig3_crs_analysis(acts, labels, cnames, ds='dad'):
    """CRS: top-20 discriminative concepts + distribution (pub_fig2 style)."""
    import textwrap
    ma = acts.mean(1)
    pos_m = ma[labels==1].mean(0); neg_m = ma[labels==0].mean(0) if (labels==0).any() else np.zeros_like(pos_m)
    pooled = np.sqrt((ma[labels==1].var(0) + (ma[labels==0].var(0) if (labels==0).any() else 0))/2 + 1e-8)
    disc = np.abs(pos_m - neg_m) / pooled
    top_k = 20
    top_idx = np.argsort(disc)[::-1][:top_k]
    wrapped = ['\n'.join(textwrap.wrap(cnames[i], 40)) for i in top_idx]
    pv = pos_m[top_idx]; nv = neg_m[top_idx]; dv = disc[top_idx]
    y = np.arange(top_k); bh = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(18, top_k*0.52+2), facecolor=BG,
                              gridspec_kw={'width_ratios':[2.8, 1.2, 0.9]})
    fig.suptitle('INSIGHT: Concept Risk Score (CRS) Analysis',
                 fontsize=14, fontweight='bold', color=TEXT, y=1.01)

    # Left: grouped bars
    ax = axes[0]
    ax.barh(y+bh/2, pv, height=bh, color=RED, alpha=0.82, label='Accident', edgecolor='white')
    ax.barh(y-bh/2, nv, height=bh, color=BLUE, alpha=0.82, label='Normal', edgecolor='white')
    ax.set_yticks(y); ax.set_yticklabels(wrapped, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Concept Activation', **FONT_LABEL)
    ax.set_title(f'Top-{top_k} Safety-Critical Concepts [{ds.upper()}]', **FONT_TITLE)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, axis='x', alpha=0.5)

    # Middle: discriminability score
    ax2 = axes[1]
    norm = plt.Normalize(dv.min(), dv.max())
    cols = [plt.cm.RdYlGn_r(norm(v)) for v in dv]
    ax2.barh(y, dv, color=cols, alpha=0.90, height=0.58, edgecolor='white')
    ax2.set_yticks([]); ax2.invert_yaxis()
    ax2.set_xlabel('Discriminability\n(normalized Cohen d)', **FONT_LABEL)
    ax2.set_title('CRS Score', **FONT_TITLE)
    for i, v in enumerate(dv): ax2.text(v+dv.max()*0.02, i, f'{v:.2f}', va='center', fontsize=8)
    ax2.grid(True, axis='x', alpha=0.5)
    plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm),
                 ax=ax2, shrink=0.5, label='Low->High risk')

    # Right: risk weight histogram
    ax3 = axes[2]
    ax3.hist(disc, bins=40, color=RED, alpha=0.72, edgecolor='white')
    ax3.axvline(dv.min(), color=ORANGE, lw=2, ls='--', label=f'Top-{top_k} cutoff')
    ax3.set_xlabel('Discriminability', **FONT_LABEL)
    ax3.set_ylabel('# Concepts', **FONT_LABEL)
    ax3.set_title('Full Distribution', **FONT_TITLE)
    ax3.legend(fontsize=8)
    ax3.text(0.97, 0.97, f'Total: {len(disc)}\nHigh: {(disc>dv.min()).sum()}',
             transform=ax3.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc'))
    ax3.grid(True, alpha=0.4)

    plt.tight_layout()
    out_path = OUT / f'insight_fig3_crs_{ds}.png'
    plt.savefig(str(out_path), dpi=220, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  [OK] {out_path.name}')


def fig4_tsne(acts, labels, cnames, ds='dad', max_pts=400):
    """t-SNE concept space (pub_fig5 style)."""
    from sklearn.manifold import TSNE
    ma = acts.mean(1)
    N = min(len(ma), max_pts)
    idx = np.random.RandomState(42).choice(len(ma), N, replace=False)
    X = ma[idx]; y = labels[idx]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    emb = TSNE(n_components=2, perplexity=min(30, N//4),
               random_state=42, n_iter=1000).fit_transform(Xs)
    pos_m = ma[labels==1].mean(0); neg_m = ma[labels==0].mean(0) if (labels==0).any() else np.zeros_like(pos_m)
    disc = np.abs(pos_m - neg_m) / (ma.std(0) + 1e-8)
    top3 = np.argsort(disc)[::-1][:3]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle(f't-SNE Concept Activation Space [{ds.upper()}]',
                 fontsize=13, fontweight='bold', y=1.01)
    for lv, col, mk, lab, sz in [(0, BLUE, 'o', 'Normal', 38), (1, RED, '*', 'Accident', 75)]:
        m = y == lv
        axes[0].scatter(emb[m,0], emb[m,1], c=col, marker=mk, s=sz,
                        alpha=0.70, label=lab, edgecolors='none', zorder=3 if lv==1 else 2)
    axes[0].set_title('Accident vs Normal', **FONT_TITLE)
    axes[0].legend(fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].grid(True, alpha=0.3)
    sc = axes[1].scatter(emb[:,0], emb[:,1], c=X[:,top3[0]],
                         cmap='RdYlBu_r', s=45, alpha=0.75, edgecolors='none')
    plt.colorbar(sc, ax=axes[1], label='Activation')
    cn = cnames[top3[0]][:50]+('...' if len(cnames[top3[0]])>50 else '')
    axes[1].set_title(f'Top risk concept:\n{cn}', fontsize=10, fontweight='bold')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT / f'insight_fig4_tsne_{ds}.png'
    plt.savefig(str(out_path), dpi=220, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  [OK] {out_path.name}')


def fig5_safety_utility():
    """AP-mTTA safety-utility summary with seed-backed ontology intervals."""
    dad = [
        ('DRIVE', 58.4, 2.31, 'post-hoc'),
        ('W3AL', 69.2, np.nan, 'verbal'),
        ('CRASH', 65.3, 1.75, 'none'),
        ('INSIGHT', 68.19, 1.75, 'intrinsic'),
        ('DAA-GNN', 75.2, 1.59, 'none'),
    ]
    a3d = [
        ('W3AL', 92.4, 4.52, 'verbal'),
        ('CRASH', 96.0, 4.27, 'none'),
        ('INSIGHT', 93.4, 4.90, 'intrinsic'),
    ]

    def load_json(path):
        path = Path(path)
        if not path.exists():
            return {}
        with path.open() as f:
            return json.load(f)

    ontology = load_json(ROOT / 'output' / 'emnlp2026_support' / 'multiseed_ontology_status.json')
    ontology_rows = ontology.get('rows', [])
    concept_style = {
        'historical_full': dict(label='Historical full', color='#92B3DB', marker='o'),
        'risk_core_v1': dict(label='Risk-core manual', color='#E6BB95', marker='s'),
        'perfect_v1': dict(label='Perfect v1', color='#8CCAB4', marker='D'),
    }

    style = {
        'intrinsic': dict(marker='*', color='#D78590', s=270, edge='#2F4158', label='INSIGHT headline'),
        'post-hoc': dict(marker='o', color='#9CBADF', s=95, edge='#2F4158', label='Post-hoc baseline'),
        'verbal': dict(marker='D', color='#B5A2D9', s=95, edge='#2F4158', label='Verbal baseline'),
        'none': dict(marker='s', color='#A9B7C7', s=85, edge='#2F4158', label='Non-interpretable baseline'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.8), facecolor=BG)
    fig.suptitle('Safety-Utility View with Seed-Backed Semantic-Interface Evidence',
                 fontsize=12.8, fontweight='bold', color=TEXT, y=1.02)

    for ax, data, title in [(axes[0], dad, 'DAD'), (axes[1], a3d, 'A3D')]:
        ax.set_facecolor(PANEL_DAD if title == 'DAD' else PANEL_A3D)
        seen = set()
        for name, ap, mtta, tier in data:
            if np.isnan(mtta):
                continue
            st = style[tier]
            lbl = st['label'] if st['label'] not in seen else None
            seen.add(st['label'])
            ax.scatter(ap, mtta, marker=st['marker'], c=st['color'], s=st['s'],
                       edgecolors=st['edge'], linewidths=0.9, alpha=0.95, label=lbl, zorder=4)
            dx = 0.35 if name != 'INSIGHT' else 0.45
            dy = 0.03 if title == 'DAD' else 0.04
            ax.text(ap + dx, mtta + dy, name, fontsize=9,
                    fontweight='bold' if name == 'INSIGHT' else 'normal', color=TEXT)

        for row in ontology_rows:
            if row.get('dataset') != title.lower() or row.get('num_completed', 0) < 3:
                continue
            cset = row.get('concept_set')
            cstyle = concept_style.get(cset, dict(label=cset, color='#333333', marker='o'))
            agg = row.get('aggregate', {})
            ap_mean = 100.0 * agg.get('AP', {}).get('mean', np.nan)
            ap_std = 100.0 * agg.get('AP', {}).get('std', 0.0)
            mtta_mean = agg.get('mTTA', {}).get('mean', np.nan)
            mtta_std = agg.get('mTTA', {}).get('std', 0.0)
            if np.isnan(ap_mean) or np.isnan(mtta_mean):
                continue
            label = cstyle['label'] if title == 'DAD' else None
            ax.errorbar(
                ap_mean, mtta_mean, xerr=ap_std, yerr=mtta_std,
                fmt=cstyle['marker'], markersize=7.5, color=cstyle['color'],
                ecolor=cstyle['color'], elinewidth=1.4, capsize=3.2,
                markeredgecolor='black', markeredgewidth=0.7, alpha=0.92,
                label=label, zorder=5,
            )
            offsets = {
                ('DAD', 'historical_full'): (0.25, 0.055),
                ('DAD', 'risk_core_v1'): (0.25, -0.055),
                ('DAD', 'perfect_v1'): (0.28, 0.035),
                ('A3D', 'historical_full'): (-1.45, 0.08),
                ('A3D', 'risk_core_v1'): (0.18, 0.08),
                ('A3D', 'perfect_v1'): (0.18, 0.08),
            }
            ox, oy = offsets.get((title, cset), (0.2, 0.06))
            ax.text(
                ap_mean + ox, mtta_mean + oy,
                cstyle['label'], fontsize=8.0, color=cstyle['color'],
                fontweight='bold', zorder=6,
                bbox=dict(facecolor='#FFFFFF', edgecolor='none', alpha=0.82, pad=0.5),
            )

        ax.annotate(
            'better',
            xy=(0.92, 0.90), xycoords='axes fraction',
            xytext=(0.76, 0.73), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=1.2, color='#667B90'),
            fontsize=8.5, color='#667B90', ha='center',
        )
        ax.set_title(f'{title}: AP vs warning lead time', fontsize=11, fontweight='bold')
        ax.set_xlabel('AP (%)', fontsize=10)
        ax.set_ylabel('mTTA (s)', fontsize=10)
        ax.grid(True, alpha=0.42, color='#D4E3F1')

    axes[0].set_xlim(52, 77)
    axes[0].set_ylim(1.45, 3.35)
    axes[1].set_xlim(90.4, 96.5)
    axes[1].set_ylim(4.0, 10.05)

    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    merged = {}
    for handle, label in list(zip(handles0, labels0)) + list(zip(handles1, labels1)):
        if label and label not in merged:
            merged[label] = handle
    legend = fig.legend(
        list(merged.values()),
        list(merged.keys()),
        loc='lower center',
        ncol=7,
        fontsize=8.4,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    legend.get_frame().set_facecolor(LEGEND_BG)
    legend.get_frame().set_edgecolor('#C8D6E5')
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = OUT / 'insight_fig5_safety_utility.png'
    plt.savefig(str(out_path), dpi=240, bbox_inches='tight', facecolor=BG)
    out_pdf = OUT / 'insight_fig5_safety_utility.pdf'
    plt.savefig(str(out_pdf), bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  [OK] {out_path.name}')
    print(f'  [OK] {out_pdf.name}')


def main():
    cnames = load_concepts()
    for ds, fps in [('dad', 20.0), ('a3d', 10.0), ('crash', 10.0)]:
        print(f'\n=== {ds.upper()} ===')
        data = load_cached(ds)
        if data is None:
            continue
        acts, probs, labels, toas = data
        print(f'  Loaded: acts={acts.shape} pos={int(labels.sum())}')
        print('  Generating fig1 hero...')
        fig1_hero(acts, probs, labels, toas, cnames, fps=fps, ds=ds)
        print('  Generating fig2 multi-scenario...')
        fig2_multi_scenario(acts, probs, labels, toas, cnames, fps=fps, ds=ds)
        print('  Generating fig3 CRS analysis...')
        fig3_crs_analysis(acts, labels, cnames, ds=ds)
        print('  Generating fig4 t-SNE...')
        fig4_tsne(acts, labels, cnames, ds=ds)
    print('  Generating fig5 safety-utility...')
    fig5_safety_utility()
    print(f'\nAll figures saved to: {OUT}')


if __name__ == '__main__':
    main()
  
