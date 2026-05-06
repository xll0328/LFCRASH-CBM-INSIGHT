#!/usr/bin/env python3
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
OUT = ROOT / 'paper/figures/insight_fig_concept_case_study.png'
OUT_PDF = OUT.with_suffix('.pdf')
OUT.parent.mkdir(parents=True, exist_ok=True)
merges = json.loads((ROOT / 'output/concept_sets/perfect_concept_set_v1.merge_examples.json').read_text())

cases = [
    ('Visibility cluster', ['reduced visibility', 'limited visibility', 'reduced visibility range'], 'visibility reduction'),
    ('Pedestrian-crossing cluster', ['pedestrian crosswalk', 'pedestrian crossing area', 'pedestrian crossing signage'], 'pedestrian crossing zone'),
    ('Merge cluster', ['lane merge conflict', 'lane merging conflict', 'merge lane conflict'], 'merge conflict'),
    ('Road-surface cluster', ['wet pavement', 'wet asphalt surface'], 'wet road surface'),
    ('Rear-end cluster', ['rear end collision risk', 'rear-end risk'], 'rear-end collision risk'),
    ('Signal-compliance cluster', ['traffic signal compliance risk', 'red signal compliance'], 'red-light compliance risk'),
]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5})
fig, ax = plt.subplots(figsize=(16.2, 8.4), dpi=220)
ax.set_xlim(0, 16.2)
ax.set_ylim(0, 8.4)
ax.axis('off')
fig.patch.set_facecolor('white')

palette = {'raw':'#E4AA78','merge':'#7B8DA4','final':'#78BFA3','text':'#2F4158','muted':'#5F7286'}
ax.text(0.35, 7.95, 'Case studies of semantic compression and canonicalization', fontsize=18, fontweight='bold', color=palette['text'])
ax.text(0.35, 7.5, 'These examples show that the final ontology is not a prettier synonym list: it preserves risk semantics while removing wording noise and fragment duplication.', fontsize=10.8, color=palette['muted'])

row_y = [6.5, 5.25, 4.0, 2.75, 1.5, 0.25]
for (title, raw_list, final_name), y in zip(cases, row_y):
    raw_box = FancyBboxPatch((0.5, y), 5.1, 0.9, boxstyle='round,pad=0.03,rounding_size=0.14', linewidth=1.8, edgecolor=palette['raw'], facecolor='white')
    ax.add_patch(raw_box)
    ax.text(0.7, y + 0.66, title, fontsize=11.1, fontweight='bold', color=palette['raw'])
    ax.text(0.7, y + 0.28, '  |  '.join(raw_list), fontsize=9.7, color=palette['text'])
    ax.add_patch(FancyArrowPatch((5.8, y + 0.45), (7.25, y + 0.45), arrowstyle='-|>', mutation_scale=16, linewidth=2, color=palette['merge']))
    final_box = FancyBboxPatch((7.5, y), 3.2, 0.9, boxstyle='round,pad=0.03,rounding_size=0.14', linewidth=1.8, edgecolor=palette['final'], facecolor='white')
    ax.add_patch(final_box)
    ax.text(7.7, y + 0.45, final_name, fontsize=10.5, color=palette['text'], va='center', fontweight='bold')
    ax.add_patch(FancyArrowPatch((10.95, y + 0.45), (12.2, y + 0.45), arrowstyle='-|>', mutation_scale=16, linewidth=2, color='#64748b'))
    ax.text(12.4, y + 0.45, 'Retained because it is shorter, canonical, and more intervention-ready.', fontsize=9.6, color=palette['muted'], va='center')

plt.tight_layout()
fig.savefig(OUT, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
print(f'Saved {OUT}')
print(f'Saved {OUT_PDF}')
