#!/usr/bin/env python3
"""
visualize_v4.py - CG-CRASH v4 可解释性可视化
生成 NeurIPS 论文核心图表:
  Fig1: Concept Timeline (WHY)
  Fig2: Actor Decision Map (WHEN)
  Fig3: Concept Risk Ranking
  Fig4: Training curves
Usage:
  python visualize_v4.py --ckpt output/dad_ac/dad_ac_v1/best_model.pt --gpu 0
"""
import os, sys, json, argparse, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT       = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT  = CRASH_ROOT / 'data'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))
from src.models_gru  import LFCRASH_CBM_GRU
from src.data_loader import DADDataset

COLORS = {'accent':'#E74C3C','safe':'#2ECC71','concept':'#3498DB',
           'actor':'#F39C12','bg':'#1A1A2E','text':'#ECF0F1','grid':'#2D2D4E'}
RISK_CMAP = LinearSegmentedColormap.from_list('risk',
    ['#1A1A2E','#3498DB','#F39C12','#E74C3C'], N=256)

def set_style():
    plt.rcParams.update({'figure.facecolor':COLORS['bg'],'axes.facecolor':COLORS['bg'],
        'axes.edgecolor':COLORS['grid'],'axes.labelcolor':COLORS['text'],
        'xtick.color':COLORS['text'],'ytick.color':COLORS['text'],
        'text.color':COLORS['text'],'grid.color':COLORS['grid'],
        'grid.alpha':0.4,'font.size':10})

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get('args', {})
    model = LFCRASH_CBM_GRU(
        x_dim=4096, h_dim=args.get('h_dim',256), z_dim=args.get('z_dim',256),
        n_layers=2, n_obj=19, n_frames=100, fps=20.0,
        with_saa=True, num_concepts=args.get('num_concepts',837),
        use_cbm=True, device=str(device), legacy=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f'Loaded model AP={ckpt.get("AP","?")}')
    return model

@torch.no_grad()
def get_predictions(model, x, device):
    x = x.unsqueeze(0).to(device)
    B, T, Np1, D = x.shape
    h = torch.zeros(model.n_layers, 1, model.h_dim, device=device)
    preds, c_acts, ac_data, hidden_list = [], [], [], []
    prev_c_act = None
    for t in range(T):
        frame = x[:, t]
        feats = model.phi_x(frame)
        img_emb, obj_emb = feats[:, 0], feats[:, 1:]
        if model.use_cbm:
            c_act, c_embed = model.cbm(img_emb)
        else:
            c_act = img_emb.new_zeros(1, model.cbm.num_concepts); c_embed = img_emb
        c_acts.append(c_act.squeeze(0).cpu().numpy())
        obj_ctx = model.ofa(obj_emb, h)
        obj_vec = obj_ctx.squeeze(1)
        fft_in  = model.fft_in(img_emb) if model.fft_in is not None else img_emb
        fft_out = model.fft_block(fft_in.unsqueeze(-1))
        fft_vec = model.fft_out(fft_out.mean(dim=1))
        delta_c = (c_act - prev_c_act) if prev_c_act is not None else torch.zeros_like(c_act)
        prev_c_act = c_act.detach()
        if len(hidden_list) > 0:
            h_stack = torch.stack(hidden_list, dim=1)
            cgta_q  = model.cgta_q(delta_c).unsqueeze(1)
            cgta_k  = model.cgta_k(h_stack)
            cgta_v  = model.cgta_v(h_stack)
            attn_w  = F.softmax(torch.bmm(cgta_q, cgta_k.transpose(1,2))/math.sqrt(model.h_dim), dim=-1)
            cgta_ctx = torch.tanh(model.cgta_gate) * torch.bmm(attn_w, cgta_v).squeeze(1)
        else:
            cgta_ctx = img_emb.new_zeros(1, model.h_dim)
        risk_w    = torch.sigmoid(model.concept_risk_w)
        risk_feat = model.crs_proj((c_act * risk_w).sum(dim=1, keepdim=True))
        gru_in = torch.cat([obj_vec, c_embed, fft_vec, cgta_ctx + risk_feat], dim=1).unsqueeze(1)
        out_t, h = model.gru(gru_in, h)
        hidden_list.append(h[-1].detach())
        preds.append(torch.softmax(out_t, dim=-1)[0, 1].item())
        if model.use_ac:
            al, av, atw = model.ac_module(h[-1], c_act)
            ac_data.append({'alert':torch.softmax(al,dim=-1)[0,1].item(),
                            'value':av[0,0].item(),'tw':atw[0,0].item()})
    return np.array(preds), np.array(c_acts), ac_data


def plot_concept_timeline(c_acts, preds, toa, concept_names, out_path, top_k=15):
    """Fig 1: Concept Timeline — WHY is it dangerous?"""
    set_style()
    T, C = c_acts.shape
    t_axis = np.arange(T) / 20.0
    top_idx  = np.argsort(c_acts.mean(0))[::-1][:top_k]
    top_acts = c_acts[:, top_idx]
    top_names = [concept_names[i] if concept_names and i < len(concept_names)
                 else f'Concept-{i}' for i in top_idx]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={'height_ratios': [3, 1.5, 1]})
    ax = axes[0]
    im = ax.imshow(top_acts.T, aspect='auto', cmap=RISK_CMAP,
                   extent=[0, T/20, -0.5, top_k-0.5], origin='lower',
                   interpolation='bilinear', vmin=0)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([n[:25] for n in top_names], fontsize=7)
    ax.set_title('Concept Activation Timeline  [WHY is it dangerous?]',
                 color=COLORS['accent'], fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax, label='Activation')
    if toa > 0:
        axes[0].axvline(toa/20, color=COLORS['accent'], lw=2, ls='--', label='Accident')
        axes[1].axvline(toa/20, color=COLORS['accent'], lw=2, ls='--')

    ax = axes[1]
    ax.fill_between(t_axis, preds, alpha=0.4, color=COLORS['accent'])
    ax.plot(t_axis, preds, color=COLORS['accent'], lw=2)
    ax.axhline(0.5, color=COLORS['text'], lw=1, ls=':', alpha=0.5)
    ax.set_ylabel('P(accident)'); ax.set_ylim(0, 1); ax.grid(True)

    ax = axes[2]
    for i, (idx, name) in enumerate(list(zip(top_idx[:6], top_names[:6]))):
        ax.plot(t_axis, c_acts[:, idx], lw=1.5, alpha=0.85, label=name[:20])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Activation')
    ax.legend(loc='upper left', fontsize=6, ncol=2); ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(); print(f'Saved: {out_path}')


def plot_actor_decision(preds, ac_data, toa, out_path):
    """Fig 2: Actor Decision Map — WHEN to alert?"""
    set_style()
    T = len(preds)
    t_axis = np.arange(T) / 20.0
    alert_p = np.array([d['alert'] for d in ac_data])
    values  = np.array([d['value'] for d in ac_data])
    tw      = np.array([d['tw']    for d in ac_data])

    fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    ax = axes[0]
    ax.fill_between(t_axis, preds, alpha=0.3, color=COLORS['accent'], label='P(accident)')
    ax.plot(t_axis, preds, color=COLORS['accent'], lw=2)
    ax.fill_between(t_axis, alert_p, alpha=0.3, color=COLORS['actor'], label='Actor P(alert)')
    ax.plot(t_axis, alert_p, color=COLORS['actor'], lw=2, ls='--')
    if toa > 0: ax.axvline(toa/20, color=COLORS['accent'], lw=2, ls='--', label='Accident')
    ax.set_ylabel('Probability'); ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=9); ax.grid(True)
    ax.set_title('Actor-Critic Decision  [WHEN to alert?]',
                 color=COLORS['actor'], fontweight='bold', fontsize=12)

    ax = axes[1]
    ax.plot(t_axis, values, color=COLORS['concept'], lw=2)
    ax.fill_between(t_axis, values, alpha=0.2, color=COLORS['concept'])
    ax.set_ylabel('V(state)'); ax.set_title('Critic Value (Long-horizon Safety)', fontsize=10)
    ax.grid(True)

    ax = axes[2]
    ax.bar(t_axis, tw, width=0.04, color=COLORS['safe'], alpha=0.7)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Weight')
    ax.set_title('Temporal Importance (Earlier = Higher Reward)', fontsize=10); ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(); print(f'Saved: {out_path}')


def plot_concept_risk_ranking(model, concept_names, out_path, top_k=30):
    """Fig 3: Learned concept risk weights (CRS)."""
    set_style()
    risk_w = torch.sigmoid(model.concept_risk_w).detach().cpu().numpy()
    top_idx = np.argsort(risk_w)[::-1][:top_k]
    top_w   = risk_w[top_idx]
    top_nm  = [concept_names[i] if concept_names and i < len(concept_names)
               else f'C-{i}' for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [COLORS['accent'] if w > 0.6 else COLORS['actor'] if w > 0.4
              else COLORS['concept'] for w in top_w]
    bars = ax.barh(range(top_k), top_w[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([n[:35] for n in top_nm[::-1]], fontsize=8)
    ax.set_xlabel('Risk Weight (sigmoid)')
    ax.set_title('Concept Risk Score (CRS) — Learned Safety-Critical Concepts',
                 color=COLORS['accent'], fontweight='bold', fontsize=12)
    ax.axvline(0.5, color=COLORS['text'], lw=1, ls='--', alpha=0.5, label='threshold=0.5')
    ax.legend(fontsize=9); ax.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(); print(f'Saved: {out_path}')


def plot_training_curves(loss_log_path, out_path):
    """Fig 4: Training loss curves."""
    if not Path(loss_log_path).exists():
        print(f'Loss log not found: {loss_log_path}'); return
    set_style()
    with open(loss_log_path) as f:
        log = json.load(f)
    epochs = [d['epoch'] for d in log]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    pairs = [('total_loss','Total Loss'),('ce_loss','CE Loss'),
             ('ac_policy_loss','Policy Loss'),('ac_value_loss','Value Loss')]
    clrs = [COLORS['accent'],COLORS['concept'],COLORS['actor'],COLORS['safe']]
    for ax, (k, title), c in zip(axes.flat, pairs, clrs):
        if k in log[0]:
            vals = [d[k] for d in log]
            ax.plot(epochs, vals, color=c, lw=2)
            ax.fill_between(epochs, vals, alpha=0.15, color=c)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Epoch'); ax.grid(True)
    plt.suptitle('CG-CRASH v4 Training Curves',
                 color=COLORS['text'], fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(); print(f'Saved: {out_path}')


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--ckpt',  type=str, required=True)
    pa.add_argument('--gpu',   type=int, default=0)
    pa.add_argument('--n_samples', type=int, default=3,
                    help='Number of test samples to visualize')
    pa.add_argument('--concept_file', type=str,
                    default=str(ROOT / '000_all_concept_set.txt'))
    pa.add_argument('--out_dir', type=str, default=None)
    args = pa.parse_args()

    device  = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ckpt_p  = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_p.parent / 'viz_v4'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load concept names
    concept_names = []
    if Path(args.concept_file).exists():
        with open(args.concept_file) as f:
            concept_names = [l.strip() for l in f if l.strip()]
        print(f'Loaded {len(concept_names)} concept names')

    model = load_model(str(ckpt_p), device)

    # Training curves
    loss_log = ckpt_p.parent / 'loss_log.json'
    plot_training_curves(str(loss_log), str(out_dir / 'training_curves.png'))

    # Concept risk ranking
    plot_concept_risk_ranking(model, concept_names,
                              str(out_dir / 'concept_risk_ranking.png'))

    # Per-sample visualizations
    te_ds = DADDataset(str(DATA_ROOT / 'dad'), 'vgg16', phase='testing', toTensor=False)
    pos_samples = [(i, te_ds[i]) for i in range(len(te_ds))
                   if te_ds[i][1][1] > 0.5][:args.n_samples]
    print(f'Visualizing {len(pos_samples)} positive samples...')

    for idx, (ds_idx, (x, y, toa)) in enumerate(pos_samples):
        x_t = torch.from_numpy(x).float()
        toa_val = float(toa[0]) if hasattr(toa, '__len__') else float(toa)
        preds, c_acts, ac_data = get_predictions(model, x_t, device)

        plot_concept_timeline(
            c_acts, preds, toa_val, concept_names,
            str(out_dir / f'concept_timeline_sample{idx}.png'))

        if ac_data:
            plot_actor_decision(
                preds, ac_data, toa_val,
                str(out_dir / f'actor_decision_sample{idx}.png'))

    print(f'\nAll figures saved to: {out_dir}')


if __name__ == '__main__':
    main()
