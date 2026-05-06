# -*- coding: utf-8 -*-
"""
src/concept_utils.py
====================
Concept-level analysis, visualization and intervention tools for CG-CRASH.

Usage:
    from src.concept_utils import (
        plot_concept_timeline,
        plot_top_concepts_bar,
        concept_intervention,
        concept_importance_heatmap,
        save_concept_report,
    )
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib (optional – graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _MPL = True
except ImportError:
    _MPL = False
    print('[concept_utils] matplotlib not available – plotting disabled')


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_concept_names(concept_file: str) -> List[str]:
    """Load concept names from text file (one per line)."""
    with open(concept_file, encoding='utf-8') as f:
        names = [l.strip() for l in f if l.strip()]
    return names


def short_name(concept: str, max_len: int = 45) -> str:
    """Truncate long concept descriptions for plot labels."""
    # Strip 'A photo of ' prefix common in CLIP prompts
    s = concept
    for prefix in ('A photo of a ', 'A photo of ', 'Photo of a ', 'Photo of '):
        if s.startswith(prefix):
            s = s[len(prefix):]
            s = s[0].upper() + s[1:]
            break
    if s.endswith('.'):
        s = s[:-1]
    return s[:max_len] + ('…' if len(s) > max_len else '')


# ─────────────────────────────────────────────────────────────────────────────
# 1. Temporal concept activation timeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_concept_timeline(
    acts: np.ndarray,          # (N, T, C)  or (T, C) for single sample
    labels: np.ndarray,        # (N,)  ignored if acts is (T,C)
    concept_names: List[str],
    top_k: int = 8,
    fps: float = 20.0,
    output_path: Optional[str] = None,
    title: str = 'Concept Activation Timeline',
) -> Optional[str]:
    """
    Plot mean concept activation over time for the top-k most
    discriminative concepts on positive (accident) samples.

    Returns path to saved figure, or None if matplotlib unavailable.
    """
    if not _MPL:
        return None

    if acts.ndim == 2:          # single sample (T, C)
        acts = acts[np.newaxis]
        labels = np.ones(1)

    # Discriminability
    mean_acts = acts.mean(axis=1)   # (N, C)
    pos = mean_acts[labels == 1]
    neg = mean_acts[labels == 0]
    if len(pos) == 0:
        return None
    disc = (np.abs(pos.mean(0) - neg.mean(0)) /
            (pos.std(0) + neg.std(0) + 1e-8)) if len(neg) > 0 else pos.mean(0)
    top_idx = np.argsort(disc)[::-1][:top_k]

    pos_acts = acts[labels == 1]    # (Np, T, C)
    T = acts.shape[1]
    time_axis = np.arange(T) / fps  # seconds

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = cm.tab10(np.linspace(0, 1, top_k))
    for rank, (idx, color) in enumerate(zip(top_idx, colors)):
        curve = pos_acts[:, :, idx].mean(axis=0)  # (T,)
        label = short_name(concept_names[idx]) if concept_names else f'C{idx}'
        ax.plot(time_axis, curve, label=label, color=color, linewidth=1.8)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Mean Activation', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Top-concepts bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_concepts_bar(
    acts: np.ndarray,           # (N, T, C)
    labels: np.ndarray,         # (N,)
    concept_names: List[str],
    top_k: int = 20,
    output_path: Optional[str] = None,
    title: str = 'Top Concepts: Positive vs Negative',
) -> Optional[str]:
    """Horizontal bar chart comparing top-k mean activations pos vs neg."""
    if not _MPL:
        return None

    mean_acts = acts.mean(axis=1)   # (N, C)
    pos = mean_acts[labels == 1]
    neg = mean_acts[labels == 0]
    if len(pos) == 0:
        return None

    disc = (np.abs(pos.mean(0) - (neg.mean(0) if len(neg) else 0)) /
            (pos.std(0) + (neg.std(0) if len(neg) else 1e-8) + 1e-8))
    top_idx = np.argsort(disc)[::-1][:top_k]

    names  = [short_name(concept_names[i]) if concept_names else f'C{i}' for i in top_idx]
    pos_v  = pos.mean(0)[top_idx]
    neg_v  = neg.mean(0)[top_idx] if len(neg) else np.zeros(top_k)

    y = np.arange(top_k)
    fig, ax = plt.subplots(figsize=(10, top_k * 0.4 + 1))
    ax.barh(y + 0.2, pos_v, height=0.35, label='Positive (accident)', color='#e74c3c', alpha=0.85)
    ax.barh(y - 0.2, neg_v, height=0.35, label='Negative (normal)',   color='#3498db', alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Activation', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Concept importance heatmap (per-frame x top-concept)
# ─────────────────────────────────────────────────────────────────────────────

def concept_importance_heatmap(
    acts: np.ndarray,           # (T, C)  single sample
    concept_names: List[str],
    top_k: int = 15,
    output_path: Optional[str] = None,
    title: str = 'Per-Frame Concept Activations',
) -> Optional[str]:
    """Heatmap: rows=top concepts, cols=time frames."""
    if not _MPL:
        return None

    assert acts.ndim == 2, 'acts must be (T, C) for heatmap'
    T, C = acts.shape
    top_idx = np.argsort(acts.mean(0))[::-1][:top_k]
    sub     = acts[:, top_idx].T    # (top_k, T)
    names   = [short_name(concept_names[i]) if concept_names else f'C{i}'
               for i in top_idx]

    fig, ax = plt.subplots(figsize=(min(T * 0.15 + 2, 16), top_k * 0.45 + 1))
    im = ax.imshow(sub, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Concept intervention (causal testing)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def concept_intervention(
    model,
    x: torch.Tensor,             # (1, T, N+1, x_dim)
    concept_idx: List[int],
    intervention_value: float = 0.0,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero (or set) specific concept activations and measure prediction change.

    Returns:
        pred_original  : (T, 2)  softmax probabilities
        pred_intervened: (T, 2)  probabilities after intervention
    """
    model.eval()
    x = x.to(device)

    # ── Original prediction ───────────────────────────────────────────────────
    _, orig_out, _ = model(x, None, None)
    orig_probs = torch.stack(
        [F.softmax(o, dim=-1) for o in orig_out], dim=1
    ).squeeze(0)   # (T, 2)

    # ── Patched forward: hook into cbm.encode ────────────────────────────────
    original_encode = model.cbm.encode

    def patched_encode(img_embed):
        acts = original_encode(img_embed)
        acts = acts.clone()
        acts[:, concept_idx] = intervention_value
        return acts

    model.cbm.encode = patched_encode
    try:
        _, intv_out, _ = model(x, None, None)
        intv_probs = torch.stack(
            [F.softmax(o, dim=-1) for o in intv_out], dim=1
        ).squeeze(0)   # (T, 2)
    finally:
        model.cbm.encode = original_encode   # always restore

    return orig_probs.cpu(), intv_probs.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary report
# ─────────────────────────────────────────────────────────────────────────────

def compute_discriminability(
    acts: np.ndarray,    # (N, T, C)
    labels: np.ndarray,  # (N,)
    concept_names: Optional[List[str]] = None,
    top_k: int = 20,
) -> List[Dict]:
    """Compute per-concept discriminability score and return top-k."""
    mean_acts = acts.mean(axis=1)   # (N, C)
    pos = mean_acts[labels == 1]
    neg = mean_acts[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return []
    disc = (np.abs(pos.mean(0) - neg.mean(0)) /
            (pos.std(0) + neg.std(0) + 1e-8))
    top_idx = np.argsort(disc)[::-1][:top_k]
    return [
        {
            'rank': r + 1,
            'idx': int(i),
            'concept': concept_names[i] if concept_names else f'concept_{i}',
            'discriminability': float(disc[i]),
            'mean_pos': float(pos.mean(0)[i]),
            'mean_neg': float(neg.mean(0)[i]),
        }
        for r, i in enumerate(top_idx)
    ]


def save_concept_report(
    acts: np.ndarray,           # (N, T, C)
    labels: np.ndarray,         # (N,)
    concept_names: List[str],
    output_dir: str,
    dataset: str = 'unknown',
    fps: float = 20.0,
    top_k: int = 20,
):
    """
    Full concept evaluation report:
      - JSON summary
      - Timeline plot (PNG)
      - Bar chart (PNG)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    disc = compute_discriminability(acts, labels, concept_names, top_k=top_k)

    # JSON
    mean_acts = acts.mean(axis=1)   # (N, C)
    pos = mean_acts[labels == 1]
    neg = mean_acts[labels == 0]

    summary = {
        'dataset': dataset,
        'n_samples': int(len(labels)),
        'n_positive': int((labels == 1).sum()),
        'n_negative': int((labels == 0).sum()),
        'n_concepts': acts.shape[2],
        'n_frames': acts.shape[1],
        'top_discriminative': disc,
        'top_positive': [
            {'rank': r+1,
             'concept': concept_names[i] if concept_names else f'concept_{i}',
             'mean_activation': float(pos.mean(0)[i])}
            for r, i in enumerate(np.argsort(pos.mean(0))[::-1][:top_k])
        ] if len(pos) > 0 else [],
        'top_negative': [
            {'rank': r+1,
             'concept': concept_names[i] if concept_names else f'concept_{i}',
             'mean_activation': float(neg.mean(0)[i])}
            for r, i in enumerate(np.argsort(neg.mean(0))[::-1][:top_k])
        ] if len(neg) > 0 else [],
    }

    with open(out / f'{dataset}_concept_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[concept_utils] Saved JSON: {out}/{dataset}_concept_summary.json')

    # Plots
    if _MPL:
        plot_concept_timeline(
            acts, labels, concept_names, top_k=min(8, top_k), fps=fps,
            output_path=str(out / f'{dataset}_concept_timeline.png'),
            title=f'[{dataset.upper()}] Concept Activation Timeline (Positive Samples)',
        )
        print(f'[concept_utils] Saved timeline: {out}/{dataset}_concept_timeline.png')

        plot_top_concepts_bar(
            acts, labels, concept_names, top_k=top_k,
            output_path=str(out / f'{dataset}_concept_bar.png'),
            title=f'[{dataset.upper()}] Top-{top_k} Discriminative Concepts',
        )
        print(f'[concept_utils] Saved bar chart: {out}/{dataset}_concept_bar.png')

    return summary
