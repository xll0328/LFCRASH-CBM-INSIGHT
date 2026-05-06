#!/usr/bin/env python3
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path('/data/sony/LFCRASH/LFCRASH-CBM/paper/figures/insight_fig_concept_pipeline.png')
OUT_PDF = OUT.with_suffix('.pdf')
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
})

fig, ax = plt.subplots(figsize=(15.6, 7.4), dpi=220)
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')
fig.patch.set_facecolor('white')

palette = {
    'blue': '#7FA8D6',
    'teal': '#7EBFD1',
    'orange': '#E4AA78',
    'red': '#D78590',
    'green': '#78BFA3',
    'slate': '#2F4158',
    'gray': '#DEE7F0',
    'text': '#2F4158',
}

def box(x, y, w, h, title, body, color, fill='#FFFFFF', title_fs=13, body_fs=10.4):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle='round,pad=0.03,rounding_size=0.18',
                           linewidth=2.0, edgecolor=color, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + 0.18, y + h - 0.34, title, fontsize=title_fs, fontweight='bold', color=color, va='top')
    ax.text(x + 0.18, y + h - 0.78, body, fontsize=body_fs, color=palette['text'], va='top', linespacing=1.35)

def arrow(x1, y1, x2, y2, color='#475569'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=18,
                                 linewidth=2.0, color=color))

ax.text(0.35, 7.55, 'Risk Concept Discovery and Adjudication Pipeline', fontsize=18.5, fontweight='bold', color=palette['slate'])
ax.text(0.35, 7.10, 'From raw driving frames to trainable accident-risk ontology for CBM + Actor-Critic training', fontsize=11.5, color='#62758a')

box(0.5, 4.0, 2.6, 2.2,
    '1. Frame Mining',
    'Positive accident videos\n↓\nPre-crash frame manifest\nDiverse risk geometries\nand visibility conditions',
    palette['blue'], fill='#F2F7FD')

box(3.5, 4.0, 2.8, 2.2,
    '2. Multimodal Discovery',
    'Risk-first VLM prompting\nRaw concepts + risk factors\nCoarse family tags\nHigh recall, noisy pool',
    palette['teal'], fill='#F2FAFC')

box(6.8, 4.0, 2.8, 2.2,
    '3. Risk-First Refinement',
    'Remove scene clutter\nMerge near-duplicates\nPrefer reusable primitives\nCompact discovered-v1 export',
    palette['orange'], fill='#FDF6EE')

box(10.1, 4.0, 3.0, 2.2,
    '4. Canonical Adjudication',
    'Stronger multimodal judge\nCanonical naming\nFamily balancing\nFinal candidate ontology',
    palette['red'], fill='#FDEFF2')

box(13.6, 4.0, 1.9, 2.2,
    '5. Deployment',
    'CLIP text encoding\nPseudo labels\nCBM training\nWHY/WHEN model',
    palette['green'], fill='#F2F9F6')

arrow(3.1, 5.1, 3.5, 5.1)
arrow(6.3, 5.1, 6.8, 5.1)
arrow(9.6, 5.1, 10.1, 5.1)
arrow(13.1, 5.1, 13.6, 5.1)

ax.text(7.9, 2.85, 'Semantic interface is constructed, not assumed', ha='center', fontsize=14.2, fontweight='bold', color=palette['slate'])
ax.text(7.9, 2.45, 'Discovered concepts are evaluated by trainability, auditability, and intervention readiness, not just readability.', ha='center', fontsize=10.8, color='#62758a')

box(1.0, 0.55, 4.4, 1.35,
    'Example raw outputs',
    '"vehicle drifting into ego lane"\n"closing gap to front vehicle"\n"brake light onset"',
    '#A995D0', fill='#F6F2FC', title_fs=11.8, body_fs=8.9)
box(5.8, 0.55, 4.4, 1.35,
    'Example refined primitives',
    'lane intrusion\ncut-in risk\nbraking onset / visibility degradation',
    '#D78590', fill='#FDEFF2', title_fs=11.8, body_fs=8.9)
box(10.6, 0.55, 4.4, 1.35,
    'Output ontology families',
    'trajectory conflict\ndistance/proximity\nvisibility / control response',
    '#78BFA3', fill='#F2F9F6', title_fs=11.8, body_fs=8.9)

plt.tight_layout()
fig.savefig(OUT, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
print(f'Saved {OUT}')
print(f'Saved {OUT_PDF}')
