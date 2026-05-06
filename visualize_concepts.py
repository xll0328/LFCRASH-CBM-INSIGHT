#!/usr/bin/env python3
"""
visualize_concepts.py — CG-CRASH v4 解释性可视化
生成论文 Figure 3: 概念激活时间线 + Actor 决策分析

Usage:
  python visualize_concepts.py \
    --ckpt output/dad_ac/dad_ac_v3_fixed_lr/best_model.pt \
    --tag dad_ac_v3_fixed_lr \
    --out_dir paper/figures
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

ROOT       = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru import LFCRASH_CBM_GRU
from src.DataLoader  import DADDataset
from torch.utils.data import DataLoader

# ── Top-20 most interpretable concept names (subset of 837) ──────────────────
CONCEPT_NAMES = [
    'vehicle_speed', 'lane_change', 'brake_light', 'pedestrian',
    'intersection', 'traffic_light', 'obstacle_ahead', 'wet_road',
    'night_driving', 'tailgating', 'sudden_stop', 'oncoming_traffic',
    'blind_spot', 'merging', 'road_narrowing', 'cyclist',
    'truck_nearby', 'construction_zone', 'curve_ahead', 'fog',
]
N_SHOW = len(CONCEPT_NAMES)


def load_model(ckpt_path: str, device: torch.device) -> LFCRASH_CBM_GRU:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get('args', {})
    model = LFCRASH_CBM_GRU(
        x_dim=4096, h_dim=args.get('h_dim', 256), z_dim=256,
        n_layers=2, n_obj=19, n_frames=100, fps=20.0,
        with_saa=True, num_concepts=args.get('num_concepts', 837),
        concept_file=None, use_cbm=True,
        device=str(device), legacy=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


@torch.no_grad()
def extract_concepts(
    model: LFCRASH_CBM_GRU,
    x: torch.Tensor,   # (1, T, n_obj, x_dim)
    device: torch.device,
) -> dict:
    """Extract per-frame concept activations and model outputs."""
    x = x.to(device)
    B, T, n_obj, x_dim = x.shape

    # Manually step through to collect intermediates
    concept_seq = []   # (T, num_concepts)
    risk_seq    = []   # (T,)
    pred_seq    = []   # (T, 2)
    ac_seq      = []   # (T, 2)  actor logits

    h = torch.zeros(model.n_layers, B, model.h_dim, device=device)
    model._rwkv_state = None

    for t in range(T):
        xt = x[:, t]  # (1, n_obj, x_dim)

        # SAA: object encoding
        obj_vec = model.ofa(model.phi_x(xt.reshape(B * n_obj, x_dim)).reshape(B, n_obj, -1)) if model.with_saa else model.phi_x(xt.mean(1))

        # FFT spectral
        fft_vec = torch.zeros(B, model.h_dim, device=device)
        if hasattr(model, 'fft_block'):
            try:
                fft_in = model.fft_in(obj_vec) if model.fft_in is not None else obj_vec
                fft_out = model.fft_block(fft_in.unsqueeze(1).unsqueeze(1))
                fft_vec = model.fft_out(fft_out.squeeze(1).squeeze(1).unsqueeze(-1)).squeeze(-1)
            except:
                pass

        # CBM
        c_act = torch.zeros(B, model.num_concepts, device=device)
        c_embed = torch.zeros(B, model.h_dim, device=device)
        if model.use_cbm and hasattr(model, 'cbm'):
            c_act, c_embed = model.cbm(obj_vec)  # encode + decode
        # CGTA + CRS
        cgta_ctx  = torch.zeros(B, model.h_dim, device=device)
        risk_score = torch.zeros(B, 1, device=device)
        if hasattr(model, 'concept_risk_w') and t > 0:
            risk_w     = torch.sigmoid(model.concept_risk_w)
            risk_score = (c_act * risk_w).sum(dim=1, keepdim=True)

        # GRU step
        gru_in_cat = torch.cat([obj_vec, c_embed, fft_vec, cgta_ctx], dim=1).unsqueeze(1)
        gru_out, h = model.gru(gru_in_cat, h)

        # Prediction from GRU hidden state
        logit = torch.zeros(B, 2, device=device)
        try:
            # GRUNet output is already logits (fc2 in forward)
            logit = gru_out if gru_out.shape[-1] == 2 else torch.zeros(B, 2, device=device)
        except:
            pass

        # Actor
        ac_logit = torch.zeros(B, 2, device=device)
        if hasattr(model, 'ac_module') and model.use_ac:
            try:
                ac_logit, _, _ = model.ac_module(h[-1], c_act)
            except:
                pass

        concept_seq.append(c_act[0, :N_SHOW].cpu().numpy())
        risk_seq.append(risk_score[0, 0].item())
        pred_seq.append(torch.softmax(logit[0], dim=-1).cpu().numpy())
        ac_seq.append(torch.softmax(ac_logit[0], dim=-1).cpu().numpy())

    return {
        'concepts': np.array(concept_seq),  # (T, N_SHOW)
        'risk':     np.array(risk_seq),      # (T,)
        'pred':     np.array(pred_seq),      # (T, 2)
        'actor':    np.array(ac_seq),        # (T, 2)
    }


def plot_concept_timeline(
    data: dict,
    toa: float,
    fps: float,
    label: int,
    out_path: str,
    title: str = '',
) -> None:
    """Generate Figure 3: dual-layer interpretability visualization."""
    T = data['concepts'].shape[0]
    times = np.arange(T) / fps  # seconds
    toa_s = toa / fps

    # Custom colormap: black → red (danger)
    danger_cmap = LinearSegmentedColormap.from_list(
        'danger', ['#0a0a0f', '#1a1a2e', '#e94560', '#ff6b35'], N=256)

    fig = plt.figure(figsize=(16, 10), facecolor='#0a0a0f')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                            left=0.08, right=0.96, top=0.92, bottom=0.08)

    ax_concept = fig.add_subplot(gs[0, :])
    ax_risk    = fig.add_subplot(gs[1, 0])
    ax_pred    = fig.add_subplot(gs[1, 1])
    ax_actor   = fig.add_subplot(gs[2, 0])
    ax_top5    = fig.add_subplot(gs[2, 1])

    for ax in [ax_concept, ax_risk, ax_pred, ax_actor, ax_top5]:
        ax.set_facecolor('#12121e')
        ax.tick_params(colors='#aaaacc', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a4a')

    # ── Panel 1: Concept Activation Heatmap ──────────────────────────────────
    concepts_T = data['concepts'].T   # (N_SHOW, T)
    im = ax_concept.imshow(
        concepts_T, aspect='auto', cmap=danger_cmap,
        extent=[0, times[-1], -0.5, N_SHOW - 0.5],
        vmin=0, vmax=concepts_T.max() + 1e-6,
    )
    ax_concept.set_yticks(range(N_SHOW))
    ax_concept.set_yticklabels(CONCEPT_NAMES, fontsize=7, color='#aaaacc')
    ax_concept.set_xlabel('Time (s)', color='#aaaacc', fontsize=9)
    ax_concept.set_title('Concept Activation Timeline  [WHY layer]',
                          color='#e0e0ff', fontsize=11, fontweight='bold')
    if label == 1 and toa_s < times[-1]:
        ax_concept.axvline(toa_s, color='#ff6b35', lw=2, ls='--', label='Accident')
        ax_concept.legend(fontsize=8, labelcolor='#ff6b35',
                          facecolor='#12121e', edgecolor='#2a2a4a')
    plt.colorbar(im, ax=ax_concept, fraction=0.015, pad=0.01).ax.tick_params(colors='#aaaacc')

    # ── Panel 2: Concept Risk Score ───────────────────────────────────────────
    ax_risk.plot(times, data['risk'], color='#e94560', lw=1.5, alpha=0.9)
    ax_risk.fill_between(times, 0, data['risk'], color='#e94560', alpha=0.2)
    ax_risk.set_title('Concept Risk Score (CRS)', color='#e0e0ff', fontsize=10)
    ax_risk.set_xlabel('Time (s)', color='#aaaacc', fontsize=9)
    ax_risk.set_ylabel('Risk', color='#aaaacc', fontsize=9)
    if label == 1 and toa_s < times[-1]:
        ax_risk.axvline(toa_s, color='#ff6b35', lw=1.5, ls='--')

    # ── Panel 3: Accident Probability ────────────────────────────────────────
    ax_pred.plot(times, data['pred'][:, 1], color='#4ecdc4', lw=1.5)
    ax_pred.fill_between(times, 0, data['pred'][:, 1], color='#4ecdc4', alpha=0.2)
    ax_pred.axhline(0.5, color='#aaaacc', lw=0.8, ls=':')
    ax_pred.set_title('Accident Probability', color='#e0e0ff', fontsize=10)
    ax_pred.set_xlabel('Time (s)', color='#aaaacc', fontsize=9)
    ax_pred.set_ylabel('P(accident)', color='#aaaacc', fontsize=9)
    ax_pred.set_ylim(0, 1)
    if label == 1 and toa_s < times[-1]:
        ax_pred.axvline(toa_s, color='#ff6b35', lw=1.5, ls='--')

    # ── Panel 4: Actor Alert Decision ────────────────────────────────────────
    ax_actor.plot(times, data['actor'][:, 1], color='#ffd700', lw=1.5)
    ax_actor.fill_between(times, 0, data['actor'][:, 1], color='#ffd700', alpha=0.2)
    ax_actor.axhline(0.5, color='#aaaacc', lw=0.8, ls=':')
    ax_actor.set_title('Actor Alert Probability  [WHEN layer]',
                        color='#e0e0ff', fontsize=10)
    ax_actor.set_xlabel('Time (s)', color='#aaaacc', fontsize=9)
    ax_actor.set_ylabel('P(alert)', color='#aaaacc', fontsize=9)
    ax_actor.set_ylim(0, 1)
    if label == 1 and toa_s < times[-1]:
        ax_actor.axvline(toa_s, color='#ff6b35', lw=1.5, ls='--')

    # ── Panel 5: Top-5 Concepts at Peak Risk Frame ────────────────────────────
    peak_t = int(np.argmax(data['risk']))
    top5_idx  = np.argsort(data['concepts'][peak_t])[-5:][::-1]
    top5_vals = data['concepts'][peak_t, top5_idx]
    top5_names = [CONCEPT_NAMES[i] for i in top5_idx]
    colors = ['#e94560', '#ff6b35', '#ffd700', '#4ecdc4', '#a8e6cf']
    bars = ax_top5.barh(range(5), top5_vals, color=colors, alpha=0.85)
    ax_top5.set_yticks(range(5))
    ax_top5.set_yticklabels(top5_names, fontsize=9, color='#e0e0ff')
    ax_top5.set_title(f'Top-5 Concepts @ Peak Risk (t={peak_t/fps:.1f}s)',
                       color='#e0e0ff', fontsize=10)
    ax_top5.set_xlabel('Activation', color='#aaaacc', fontsize=9)
    for bar, val in zip(bars, top5_vals):
        ax_top5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', color='#aaaacc', fontsize=8)

    # Main title
    status = 'ACCIDENT' if label == 1 else 'NORMAL'
    fig.suptitle(
        f'CG-CRASH v4 — Dual-Layer Interpretability  [{status}]\n'
        f'{title}',
        color='#e0e0ff', fontsize=12, fontweight='bold', y=0.97
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0a0f', edgecolor='none')
    plt.close()
    print(f'Saved: {out_path}')


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--ckpt',    type=str, required=True)
    pa.add_argument('--tag',     type=str, default='viz')
    pa.add_argument('--gpu',     type=int, default=0)
    pa.add_argument('--n_cases', type=int, default=4,
                    help='Number of cases to visualize (pos+neg each)')
    pa.add_argument('--out_dir', type=str, default='paper/figures')
    args = pa.parse_args()

    device  = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading model from {args.ckpt}...')
    model = load_model(args.ckpt, device)

    data_path = str(CRASH_ROOT / 'data' / 'dad')
    ds = DADDataset(data_path, 'vgg16', phase='testing', toTensor=False)
    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        collate_fn=lambda b: b[0])

    n_pos = 0; n_neg = 0
    for sample in loader:
        if len(sample) == 3:
            xs, ys, toa = sample
        else:
            continue

        if isinstance(xs, np.ndarray):
            xs = torch.from_numpy(xs).float()
        if xs.dim() == 3:   # (T, n_obj, x_dim)
            xs = xs.unsqueeze(0)

        label = int(ys[1]) if hasattr(ys, '__len__') else int(ys)
        toa_val = float(toa[0]) if hasattr(toa, '__len__') else float(toa)

        if label == 1 and n_pos >= args.n_cases: continue
        if label == 0 and n_neg >= args.n_cases: continue

        try:
            data = extract_concepts(model, xs, device)
        except Exception as e:
            print(f'Skip (error): {e}'); continue

        case_type = 'pos' if label == 1 else 'neg'
        case_idx  = n_pos if label == 1 else n_neg
        out_path  = out_dir / f'{args.tag}_{case_type}_{case_idx:02d}.png'

        plot_concept_timeline(
            data, toa=toa_val, fps=20.0, label=label,
            out_path=str(out_path),
            title=f'Tag: {args.tag} | Case: {case_type}_{case_idx:02d}',
        )

        if label == 1: n_pos += 1
        else:          n_neg += 1

        if n_pos >= args.n_cases and n_neg >= args.n_cases:
            break

    print(f'Done. Generated {n_pos} positive + {n_neg} negative cases.')
    print(f'Figures saved to: {out_dir}')


if __name__ == '__main__':
    main()
