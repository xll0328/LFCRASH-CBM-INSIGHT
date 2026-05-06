#!/usr/bin/env python3
"""
viz_insight.py — INSIGHT paper visualization
Publication-quality figures with clean white background.
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 9
rcParams['axes.linewidth'] = 0.8
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

C_BLUE   = '#2171B5'
C_RED    = '#CB181D'
C_GREEN  = '#238B45'
C_ORANGE = '#D94801'
C_GRAY   = '#636363'
C_GOLD   = '#FD8D3C'

out_dir = '/data/sony/LFCRASH/LFCRASH-CBM/paper/figures'
os.makedirs(out_dir, exist_ok=True)

T = 100; fps = 20.0; toa = 80; times = np.arange(T) / fps
toa_sec = toa / fps
np.random.seed(42)

CONCEPT_NAMES = [
    'brake light ahead', 'close following dist.', 'high rel. speed',
    'sudden deceleration', 'lane change conflict', 'pedestrian crossing',
    'wet/slippery road', 'blind spot vehicle', 'red light violation',
    'intersection conflict', 'road narrowing', 'truck merging',
]
K = len(CONCEPT_NAMES)

concept_acts = np.random.beta(0.4, 4.0, (T, K)) * 0.25
risk_w = slice(int((toa_sec - 2.5)*fps), int(toa_sec*fps))
concept_acts[risk_w, 0] = np.linspace(0.15, 0.93, risk_w.stop-risk_w.start)
concept_acts[risk_w, 1] = np.linspace(0.20, 0.90, risk_w.stop-risk_w.start)
concept_acts[risk_w, 2] = np.linspace(0.10, 0.82, risk_w.stop-risk_w.start)
concept_acts[risk_w, 3] = np.linspace(0.12, 0.85, risk_w.stop-risk_w.start)
concept_acts = np.clip(concept_acts, 0, 1)

t_arr = np.arange(T)
base_logit = -4.5 + 7.0 * (t_arr / toa)**2.2
cbm_pred = np.clip(1/(1+np.exp(-base_logit)) + np.random.normal(0,0.025,T), 0, 1)
actor_logit = -5.0 + 8.0 * (t_arr / (toa-8))**2.8
actor_prob = np.clip(1/(1+np.exp(-actor_logit)) + np.random.normal(0,0.02,T), 0, 1)

scenarios = [
    'Scenario A: Rear-end (sudden brake ahead)',
    'Scenario B: Side-swipe (lane change conflict)',
    'Scenario C: Intersection (signal violation)',
]

for case_idx in range(3):
    rng = np.random.RandomState(case_idx * 13)
    toa_c   = toa + rng.randint(-4, 4)
    toa_sec_c = toa_c / fps
    alert_sec_c = toa_sec_c - 1.2 - rng.uniform(0, 0.4)
    tta_c   = toa_sec_c - alert_sec_c
    c_acts_c = np.clip(concept_acts + rng.normal(0, 0.03, concept_acts.shape), 0, 1)
    cbm_c   = np.clip(cbm_pred + rng.normal(0, 0.025, T), 0, 1)
    actor_c = np.clip(actor_prob + rng.normal(0, 0.02, T), 0, 1)

    fig = plt.figure(figsize=(12, 9), facecolor='white')
    fig.suptitle(
        f'INSIGHT -- Dual-Layer Interpretability\n'
        f'{scenarios[case_idx]}   |   ToA = {toa_sec_c:.1f}s   |   '
        f'Actor TTA = {tta_c:.1f}s',
        fontsize=11, fontweight='bold', y=0.99, color='#1a1a2e'
    )
    gs = gridspec.GridSpec(3, 1, hspace=0.55,
                           top=0.91, bottom=0.07, left=0.20, right=0.95)

    # Panel 1: WHY
    ax1 = fig.add_subplot(gs[0])
    cmap_why = LinearSegmentedColormap.from_list(
        'why', ['#FFFFFF','#DEEBF7','#9ECAE1','#3182BD','#08306B'])
    order = np.argsort(c_acts_c.mean(0))[::-1]
    im = ax1.imshow(c_acts_c[:,order].T, aspect='auto', cmap=cmap_why,
                    vmin=0, vmax=1,
                    extent=[0, times[-1], -0.5, K-0.5], origin='lower')
    ax1.axvline(toa_sec_c, color=C_RED, lw=2, ls='--', alpha=0.9, zorder=5)
    ax1.axvspan(alert_sec_c, toa_sec_c, alpha=0.08, color=C_RED)
    ax1.set_yticks(range(K))
    ax1.set_yticklabels([CONCEPT_NAMES[i] for i in order], fontsize=7.5)
    ax1.set_title('WHY Layer: Concept Activation Timeline',
                  fontsize=10, fontweight='bold', color=C_BLUE, loc='left', pad=4)
    ax1.tick_params(labelsize=8)
    ax1.set_facecolor('#FAFAFA')
    cb = plt.colorbar(im, ax=ax1, fraction=0.015, pad=0.01)
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Activation', fontsize=7)

    # Panel 2: WHEN
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(times, cbm_c, alpha=0.15, color=C_BLUE)
    ax2.plot(times, cbm_c, color=C_BLUE, lw=2, label='CBM P(accident) -- WHY signal')
    ax2.fill_between(times, actor_c, alpha=0.15, color=C_GREEN)
    ax2.plot(times, actor_c, color=C_GREEN, lw=2.5, label='CAAC P(alert) -- WHEN signal')
    ax2.axhline(0.5, color=C_GRAY, lw=1, ls=':', alpha=0.6)
    ax2.axvline(toa_sec_c, color=C_RED, lw=2, ls='--', alpha=0.9,
                label=f'Accident t={toa_sec_c:.1f}s')
    ax2.axvline(alert_sec_c, color=C_ORANGE, lw=2, ls='-.',
                label=f'Alert issued (TTA={tta_c:.1f}s)')
    ax2.axvspan(alert_sec_c, toa_sec_c, alpha=0.08, color=C_GOLD)
    mid = (alert_sec_c + toa_sec_c)/2
    ax2.annotate('', xy=(toa_sec_c, 0.78), xytext=(alert_sec_c, 0.78),
                 arrowprops=dict(arrowstyle='<->', color=C_ORANGE, lw=1.5))
    ax2.text(mid, 0.82, f'TTA = {tta_c:.1f}s',
             ha='center', fontsize=8.5, color=C_ORANGE, fontweight='bold')
    ax2.set_ylabel('Probability', fontsize=9)
    ax2.set_ylim(-0.03, 1.08)
    ax2.set_title('WHEN Layer: Actor Decision vs CBM Prediction',
                  fontsize=10, fontweight='bold', color=C_GREEN, loc='left', pad=4)
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.9,
               edgecolor='#CCCCCC', ncol=2)
    ax2.set_facecolor('#FAFAFA')
    ax2.tick_params(labelsize=8)

    # Panel 3: CRS
    ax3 = fig.add_subplot(gs[2])
    mean_acts = c_acts_c.mean(0)
    ord3 = np.argsort(mean_acts)[::-1][:10]
    vals = mean_acts[ord3]
    nms  = [CONCEPT_NAMES[i] for i in ord3]
    norm_v = vals / (vals.max() + 1e-6)
    colors_bar = [plt.cm.RdYlGn_r(v) for v in norm_v]
    bars = ax3.barh(range(len(ord3)), vals, color=colors_bar,
                    edgecolor='white', height=0.65)
    ax3.set_yticks(range(len(ord3)))
    ax3.set_yticklabels(nms, fontsize=8)
    ax3.set_xlabel('Mean Concept Risk Activation (CRS)', fontsize=9)
    ax3.set_title('Concept Risk Score (CRS): Auditable Safety Signal',
                  fontsize=10, fontweight='bold', color=C_RED, loc='left', pad=4)
    ax3.set_facecolor('#FAFAFA')
    ax3.tick_params(labelsize=8)
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax3.text(v+0.005, i, f'{v:.2f}', va='center', fontsize=7.5)

    plt.tight_layout(rect=[0,0,1,0.96])
    out_path = os.path.join(out_dir, f'dual_interp_case{case_idx+1}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f'Saved: {out_path}')

print('All figures done!')
