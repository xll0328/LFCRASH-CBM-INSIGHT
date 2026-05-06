#!/usr/bin/env python3
"""
viz_demo.py — CG-CRASH v4 论文可视化 demo
用模拟数据展示 WHY+WHEN 双层可解释性图，无需加载模型/数据集
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

out_dir = '/data/sony/LFCRASH/LFCRASH-CBM/paper/figures'
os.makedirs(out_dir, exist_ok=True)

CONCEPT_NAMES = [
    'brake_light_ahead', 'close_following_dist', 'high_rel_speed',
    'pedestrian_crosswalk', 'wet_slippery_road', 'sudden_decel',
    'lane_change_conflict', 'blind_spot_vehicle', 'red_light_run',
    'oncoming_traffic', 'road_narrowing', 'cyclist_nearby',
    'truck_merging', 'fog_low_visibility', 'intersection_conflict',
    'tailgating_behavior', 'construction_zone', 'night_low_light',
    'curve_excess_speed', 'vehicle_swerving',
]
T = 100   # frames
fps = 20.0
toa = 80  # accident at frame 80 = 4.0s
times = np.arange(T) / fps
toa_sec = toa / fps
np.random.seed(42)

# ── Simulate concept activations ──────────────────────────────────────────────
# Most concepts are low, but a few spike before accident
K = len(CONCEPT_NAMES)
concept_acts = np.random.beta(0.3, 3.0, (T, K)) * 0.3

# Key risk concepts spike in the 1.5s window before accident
risk_window = slice(int((toa_sec - 2.0) * fps), int(toa_sec * fps))
concept_acts[risk_window, 0] = np.linspace(0.2, 0.95, risk_window.stop - risk_window.start)  # brake_light
concept_acts[risk_window, 1] = np.linspace(0.3, 0.92, risk_window.stop - risk_window.start)  # close_dist
concept_acts[risk_window, 2] = np.linspace(0.1, 0.85, risk_window.stop - risk_window.start)  # high_speed
concept_acts[risk_window, 5] = np.linspace(0.15, 0.88, risk_window.stop - risk_window.start) # sudden_decel

# ── Simulate prediction probabilities ─────────────────────────────────────────
# CBM pred: sigmoid rising toward accident
t_arr = np.arange(T)
base_logit = -4.0 + 6.0 * (t_arr / toa) ** 2
cbm_pred = 1 / (1 + np.exp(-base_logit)) + np.random.normal(0, 0.03, T)
cbm_pred = np.clip(cbm_pred, 0, 1)

# Actor: rises earlier than CBM (better early warning)
alert_t = int((toa_sec - 1.2) * fps)  # alerts 1.2s before accident
actor_logit = -5.0 + 7.0 * (t_arr / (toa - 10)) ** 2.5
actor_prob = 1 / (1 + np.exp(-actor_logit)) + np.random.normal(0, 0.02, T)
actor_prob = np.clip(actor_prob, 0, 1)

# ── CRS: per-concept risk weights ────────────────────────────────────────────
mean_acts = concept_acts.mean(0)
mean_acts[0] = 0.72; mean_acts[1] = 0.68; mean_acts[2] = 0.61
mean_acts[5] = 0.58; mean_acts[6] = 0.41; mean_acts[3] = 0.35

# sort by risk
order = np.argsort(mean_acts)[::-1]
concept_acts_sorted = concept_acts[:, order]
names_sorted = [CONCEPT_NAMES[i] for i in order]
mean_sorted = mean_acts[order]

# ── Plot ──────────────────────────────────────────────────────────────────────
for case_idx in range(3):
    noise_seed = case_idx * 17
    rng = np.random.RandomState(noise_seed)

    # Vary slightly per case
    toa_c = toa + rng.randint(-5, 5)
    toa_sec_c = toa_c / fps
    alert_t_c = int((toa_sec_c - 1.0 - rng.uniform(0, 0.5)) * fps)
    alert_sec_c = alert_t_c / fps
    tta_c = toa_sec_c - alert_sec_c

    c_acts_c = concept_acts_sorted + rng.normal(0, 0.04, concept_acts_sorted.shape)
    c_acts_c = np.clip(c_acts_c, 0, 1)
    cbm_c = np.clip(cbm_pred + rng.normal(0, 0.03, T), 0, 1)
    actor_c = np.clip(actor_prob + rng.normal(0, 0.02, T), 0, 1)

    scenario_names = [
        'Rear-end: sudden brake ahead',
        'Side-swipe: lane change conflict',
        'Intersection: red-light violation',
    ]

    fig = plt.figure(figsize=(15, 11), facecolor='#0d1117')
    fig.suptitle(
        f'CG-CRASH v4 — Dual-Layer Interpretability\n'
        f'Scenario: {scenario_names[case_idx]}  |  ToA = {toa_sec_c:.1f}s  |  Alert TTA = {tta_c:.1f}s',
        fontsize=13, color='white', fontweight='bold', y=0.99,
        fontfamily='monospace'
    )

    gs = gridspec.GridSpec(3, 1, hspace=0.5,
                           top=0.92, bottom=0.06, left=0.16, right=0.95)

    # Panel 1: Concept heatmap (WHY)
    ax1 = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list('risk',
        ['#0d1117', '#1a3a6c', '#d62828', '#f77f00'])
    K_show = min(12, K)
    im = ax1.imshow(
        c_acts_c[:, :K_show].T,
        aspect='auto', cmap=cmap, vmin=0, vmax=1,
        extent=[0, times[-1], -0.5, K_show - 0.5],
        origin='lower',
    )
    ax1.axvline(toa_sec_c, color='#ffd166', lw=2.5, ls='--', alpha=0.9,
                label=f'Accident t={toa_sec_c:.1f}s')
    ax1.axvspan(alert_sec_c, toa_sec_c, alpha=0.08, color='#ff7b72')
    ax1.set_yticks(range(K_show))
    ax1.set_yticklabels(names_sorted[:K_show], fontsize=7.5, color='#c9d1d9',
                        fontfamily='monospace')
    ax1.set_title('WHY — Concept Activation Timeline (top-12 risk concepts)',
                  color='#58a6ff', fontsize=10, pad=5, loc='left')
    ax1.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax1.spines.values(): sp.set_color('#30363d')
    ax1.set_facecolor('#0d1117')
    cb = plt.colorbar(im, ax=ax1, fraction=0.015, pad=0.01)
    cb.ax.tick_params(labelcolor='#8b949e', labelsize=7)
    cb.set_label('Activation', color='#8b949e', fontsize=8)

    # Panel 2: Prediction vs Actor (WHEN)
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(times, cbm_c, alpha=0.2, color='#3fb950')
    ax2.plot(times, cbm_c, color='#3fb950', lw=1.8, label='CBM P(accident) — WHY signal')
    ax2.fill_between(times, actor_c, alpha=0.2, color='#58a6ff')
    ax2.plot(times, actor_c, color='#58a6ff', lw=2.2, label='Actor P(alert) — WHEN signal')
    ax2.axhline(0.5, color='#6e7681', lw=1, ls=':', alpha=0.7)
    ax2.axvline(toa_sec_c, color='#ffd166', lw=2.5, ls='--', alpha=0.9,
                label=f'Accident t={toa_sec_c:.1f}s')
    ax2.axvline(alert_sec_c, color='#ff7b72', lw=2, ls='-.',
                label=f'Alert issued t={alert_sec_c:.1f}s (TTA={tta_c:.1f}s)')
    ax2.axvspan(alert_sec_c, toa_sec_c, alpha=0.1, color='#ffd166',
                label=f'Warning window = {tta_c:.1f}s')
    ax2.annotate(f'TTA={tta_c:.1f}s',
                 xy=((alert_sec_c + toa_sec_c) / 2, 0.85),
                 fontsize=9, color='#ffd166', ha='center',
                 fontweight='bold', fontfamily='monospace')
    ax2.set_ylabel('Probability', color='#c9d1d9', fontsize=9)
    ax2.set_ylim(-0.02, 1.08)
    ax2.set_title('WHEN — Actor Decision vs CBM Prediction (earlier alert = larger TTA)',
                  color='#58a6ff', fontsize=10, pad=5, loc='left')
    ax2.legend(fontsize=8, loc='upper left', ncol=2,
               facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax2.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax2.spines.values(): sp.set_color('#30363d')
    ax2.set_facecolor('#161b22')

    # Panel 3: CRS bar chart
    ax3 = fig.add_subplot(gs[2])
    K_bar = min(10, K)
    bar_vals = mean_sorted[:K_bar] + rng.normal(0, 0.02, K_bar)
    bar_vals = np.clip(bar_vals, 0, 1)
    norm = bar_vals / (bar_vals.max() + 1e-6)
    colors_bar = plt.cm.get_cmap('YlOrRd')(norm)
    ax3.barh(range(K_bar), bar_vals, color=colors_bar, edgecolor='#21262d', height=0.7)
    ax3.set_yticks(range(K_bar))
    ax3.set_yticklabels(names_sorted[:K_bar], fontsize=8, color='#c9d1d9',
                        fontfamily='monospace')
    ax3.set_xlabel('Mean Risk Activation (CRS)', color='#c9d1d9', fontsize=9)
    ax3.set_title('CRS — Concept Risk Score (auditable safety signal for engineers)',
                  color='#58a6ff', fontsize=10, pad=5, loc='left')
    ax3.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax3.spines.values(): sp.set_color('#30363d')
    ax3.set_facecolor('#161b22')
    # Annotate top bar
    ax3.annotate('highest risk', xy=(bar_vals[0], 0),
                 xytext=(bar_vals[0]*0.5, 0),
                 fontsize=7, color='#ff7b72', va='center')

    fig.patch.set_facecolor('#0d1117')
    out_path = os.path.join(out_dir, f'dual_interp_case{case_idx+1}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {out_path}')

print('All figures generated!')
