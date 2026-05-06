#!/usr/bin/env python3
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
FIG = ROOT / 'paper/figures/insight_fig_concept_evolution.png'
FIG.parent.mkdir(parents=True, exist_ok=True)

risk_core = json.loads((ROOT / 'output/concept_sets/risk_core_concept_set_v1.meta.json').read_text())
perfect_meta = json.loads((ROOT / 'output/concept_sets/perfect_concept_set_v1.meta.json').read_text())
review_notes = (ROOT / 'output/concept_sets/PER_CONCEPT_HUMAN_REVIEW.md').read_text().splitlines()

risk_examples = risk_core['families']['vulnerable_road_users'][:2] + risk_core['families']['right_of_way_conflict'][:2] + risk_core['families']['occlusion_visibility'][:2] + risk_core['families']['relative_motion'][:2]
perfect_examples = [
    'pedestrian crossing risk',
    'pedestrian crossing trajectory',
    'headlight glare impairment',
    'limited forward visibility',
    'merge conflict',
    'merge maneuver',
    'rear-end collision risk',
    'red-light compliance risk',
]
merge_examples = [
    'reduced visibility → visibility reduction',
    'pedestrian crosswalk → pedestrian crossing zone',
    'lane merging conflict → merge conflict',
    'headlight glare → headlight glare impairment',
    'wet pavement → wet road surface',
    'reduced traction → reduced tire traction',
    'rear-end risk → rear-end collision risk',
    'red signal compliance → red-light compliance risk',
]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, ax = plt.subplots(figsize=(15.8, 7.4), dpi=220)
ax.set_xlim(0, 15.8)
ax.set_ylim(0, 7.4)
ax.axis('off')
fig.patch.set_facecolor('white')

colors = {'manual':'#7c3aed','merge':'#c2410c','final':'#0f766e','text':'#111827','muted':'#475569'}

def panel(x, y, w, h, title, subtitle, items, color):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.18', linewidth=2.1, edgecolor=color, facecolor='white')
    ax.add_patch(rect)
    ax.text(x + 0.2, y + h - 0.3, title, fontsize=15, fontweight='bold', color=color, va='top')
    ax.text(x + 0.2, y + h - 0.7, subtitle, fontsize=10.7, color=colors['muted'], va='top')
    yy = y + h - 1.08
    for item in items:
        ax.text(x + 0.24, yy, u'• ' + item, fontsize=10.4, color=colors['text'], va='top')
        yy -= 0.46

ax.text(0.35, 6.9, 'Ontology evolution toward a paper-ready semantic interface', fontsize=19, fontweight='bold', color=colors['text'])
ax.text(0.35, 6.45, 'The key transition is not only from manual to discovered concepts, but from noisy phrase pools to canonical, balanced, and review-validated risk primitives.', fontsize=11.5, color=colors['muted'])

panel(0.45, 1.0, 4.55, 5.2, 'Manual risk-core prior (30)', 'High-precision human risk primitives', risk_examples, colors['manual'])
panel(5.65, 1.0, 4.55, 5.2, 'Canonical merge and polishing', 'Representative wording consolidations from the polishing pipeline', merge_examples, colors['merge'])
panel(10.85, 1.0, 4.45, 5.2, 'Perfect concept set v1 (80)', 'Final paper-ready ontology after balancing + human review', perfect_examples, colors['final'])

ax.annotate('', xy=(5.45, 3.6), xytext=(5.05, 3.6), arrowprops=dict(arrowstyle='-|>', lw=2.2, color='#64748b'))
ax.annotate('', xy=(10.65, 3.6), xytext=(10.25, 3.6), arrowprops=dict(arrowstyle='-|>', lw=2.2, color='#64748b'))

ax.text(7.9, 0.44, 'Progression: human prior → merge provenance and family balancing → review-validated paper ontology', ha='center', fontsize=12, color=colors['muted'], fontweight='bold')
plt.tight_layout()
fig.savefig(FIG, bbox_inches='tight')
print(f'Saved {FIG}')
