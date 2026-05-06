#!/usr/bin/env python3
from pathlib import Path
import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
FIG = ROOT / 'paper/figures/insight_fig_concept_family_coverage.png'
FIG.parent.mkdir(parents=True, exist_ok=True)

risk_core_meta = json.loads((ROOT / 'output/concept_sets/risk_core_concept_set_v1.meta.json').read_text())
perfect_audit = json.loads((ROOT / 'output/concept_sets/perfect_concept_set_v1.audit.json').read_text())
perfect_family = json.loads((ROOT / 'output/concept_sets/perfect_concept_set_v1.family_meta.json').read_text())

family_order = [
    'vulnerable_road_users',
    'right_of_way_conflict',
    'occlusion_visibility',
    'relative_motion',
    'agent_behavior',
    'surface_weather',
    'road_layout_constraint',
    'traffic_density_obstacle',
    'imminent_crash_cue',
]
labels = [
    'VRU', 'RoW', 'Visibility', 'Motion', 'Behavior',
    'Surface', 'Layout', 'Obstacle', 'Crash cue'
]

risk_counts = [len(risk_core_meta['families'].get(k, [])) for k in family_order]
pre_counts = [perfect_audit['pre_family_counts'].get(k, 0) for k in family_order]
post_counts = [perfect_family[k]['count'] for k in family_order]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=220, gridspec_kw={'width_ratios':[1.1, 1]})
fig.patch.set_facecolor('white')

# left: before vs after
ax = axes[0]
x = np.arange(len(labels))
width = 0.36
bars1 = ax.bar(x - width/2, pre_counts, width, color='#c2410c', alpha=0.88, label='Large-vocab academic pool (611)')
bars2 = ax.bar(x + width/2, post_counts, width, color='#0f766e', alpha=0.92, label='Perfect concept set v1 (80)')
for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1.5, f'{int(h)}', ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Number of concepts')
ax.set_title('Family balancing from large-vocab pool to paper-ready ontology', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.22)
ax.legend(frameon=False, loc='upper right')

# right: risk_core vs perfect normalized diversity
ax = axes[1]
risk_pct = np.array(risk_counts, dtype=float) / max(sum(risk_counts), 1) * 100.0
post_pct = np.array(post_counts, dtype=float) / max(sum(post_counts), 1) * 100.0
bars1 = ax.barh(np.arange(len(labels)) + 0.18, risk_pct, height=0.34, color='#7c3aed', alpha=0.9, label='Risk-core manual set (30)')
bars2 = ax.barh(np.arange(len(labels)) - 0.18, post_pct, height=0.34, color='#0891b2', alpha=0.9, label='Perfect concept set v1 (80)')
for bars in [bars1, bars2]:
    for b in bars:
        w = b.get_width()
        ax.text(w + 0.25, b.get_y() + b.get_height()/2, f'{w:.1f}%', va='center', fontsize=8.2)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel('Share of ontology (%)')
ax.set_title('Compact manual prior vs. balanced paper-ready ontology', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.22)
ax.legend(frameon=False, loc='lower right')

fig.text(0.5, 0.01,
         'The final ontology does not merely shrink the vocabulary: it explicitly rebalances underrepresented crash-cue and obstacle families while consolidating over-fragmented visibility and behavior clusters.',
         ha='center', fontsize=10.4, color='#475569')
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.savefig(FIG, bbox_inches='tight')
print(f'Saved {FIG}')
