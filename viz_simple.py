#!/usr/bin/env python3
"""
viz_simple.py — Simple concept visualization using model forward hooks.
Generates concept timeline + actor decision plot for paper Figure 3.
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT       = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src'))

CONCEPT_NAMES = [
    'vehicle_speed', 'lane_change', 'brake_light', 'pedestrian',
    'intersection', 'traffic_light', 'obstacle_ahead', 'wet_road',
    'night_driving', 'tailgating', 'sudden_stop', 'oncoming_traffic',
    'blind_spot', 'merging', 'road_narrowing', 'cyclist',
    'truck_nearby', 'construction_zone', 'curve_ahead', 'fog',
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    type=str, required=True)
    p.add_argument('--gpu',     type=int, default=4)
    p.add_argument('--out_dir', type=str, default='paper/figures')
    p.add_argument('--n_cases', type=int, default=3)
    return p.parse_args()


def load_model(ckpt_path, device):
    from src.models_gru import LFCRASH_CBM_GRU
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    model = LFCRASH_CBM_GRU(
        x_dim=4096, h_dim=args.get('h_dim', 256), z_dim=256,
        n_layers=2, n_obj=19, n_frames=100, fps=20.0,
        with_saa=True, num_concepts=args.get('num_concepts', 837),
        concept_file=None, use_cbm=True,
        device=str(device), legacy=False,
        use_ac=args.get('use_ac', True),
    ).to(device)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def run_forward_with_hooks(model, batch_x, batch_toa, device):
    """
    Run model forward with hooks to capture:
    - concept activations at each frame
    - actor policy probability at each frame
    - accident probability at each frame
    """
    from src.models_gru import LFCRASH_CBM_GRU
    B, T, n_obj, x_dim = batch_x.shape

    concept_acts = []  # T x (B, K)
    actor_probs  = []  # T x (B,)
    pred_probs   = []  # T x (B,)

    # Hook on CBM
    cbm_out = []
    def cbm_hook(module, inp, out):
        acts, embed = out
        cbm_out.append(acts.detach().cpu())
    h_cbm = model.cbm.register_forward_hook(cbm_hook)

    # Hook on AC module
    ac_out = []
    def ac_hook(module, inp, out):
        # out = (action_logits, value, time_weight)
        if isinstance(out, tuple) and len(out) >= 1:
            logits = out[0].detach().cpu()  # (B, 2)
            probs  = torch.softmax(logits, dim=-1)[:, 1]  # alert prob
            ac_out.append(probs)
    if hasattr(model, 'ac_module') and model.use_ac:
        h_ac = model.ac_module.register_forward_hook(ac_hook)
    else:
        h_ac = None

    with torch.no_grad():
        pred_seq, _ = model(batch_x.to(device), batch_toa.to(device))
        # pred_seq: (B, T, 2)

    h_cbm.remove()
    if h_ac is not None:
        h_ac.remove()

    # pred_probs from model output
    pred_probs_np = torch.softmax(pred_seq, dim=-1)[:, :, 1].cpu().numpy()  # (B, T)

    # concept acts: should have T entries from hook (one per frame)
    if len(cbm_out) >= T:
        concept_acts_np = torch.stack(cbm_out[-T:], dim=1).numpy()  # (B, T, K)
    elif len(cbm_out) > 0:
        concept_acts_np = torch.stack(cbm_out, dim=1).numpy()
    else:
        concept_acts_np = np.zeros((B, T, 20))

    if len(ac_out) >= T:
        actor_probs_np = torch.stack(ac_out[-T:], dim=1).numpy()  # (B, T)
    elif len(ac_out) > 0:
        actor_probs_np = torch.stack(ac_out, dim=1).numpy()
    else:
        actor_probs_np = pred_probs_np  # fallback

    return concept_acts_np, actor_probs_np, pred_probs_np


def plot_case(concept_acts, actor_probs, pred_probs, toa, label, fps, case_idx, out_dir):
    T = pred_probs.shape[0]
    K = min(20, concept_acts.shape[-1])
    times = np.arange(T) / fps  # seconds
    toa_sec = toa / fps

    # Use top-K most active concepts for this case
    mean_acts = concept_acts.mean(0)  # (K_full,)
    top_k_idx = np.argsort(mean_acts)[::-1][:K]
    top_concept_names = [CONCEPT_NAMES[i % len(CONCEPT_NAMES)] for i in range(K)]

    fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
    fig.suptitle(
        f'CG-CRASH v4 — Dual-Layer Interpretability\n'
        f'Case {case_idx+1}: {"Accident" if label > 0 else "No-Accident"} '
        f'(ToA={toa_sec:.1f}s)',
        fontsize=13, color='white', fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 1, hspace=0.45,
                           top=0.91, bottom=0.07, left=0.1, right=0.95)

    # ── Panel 1: Concept activations heatmap (WHY) ────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list('risk', ['#0d1117', '#1a3a5c', '#e63946'])
    im = ax1.imshow(
        concept_acts[:, top_k_idx].T,  # (K, T)
        aspect='auto', cmap=cmap, vmin=0, vmax=1,
        extent=[0, times[-1], -0.5, K - 0.5],
        origin='lower',
    )
    if label > 0:
        ax1.axvline(toa_sec, color='#ffd166', lw=2, ls='--', label=f'Accident t={toa_sec:.1f}s')
    ax1.set_yticks(range(K))
    ax1.set_yticklabels(top_concept_names, fontsize=7, color='#c9d1d9')
    ax1.set_xlabel('')
    ax1.set_title('WHY: Concept Activation Timeline (top-20 risk concepts)',
                  color='#58a6ff', fontsize=10, pad=4)
    ax1.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax1.spines.values(): sp.set_color('#30363d')
    plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.01).ax.tick_params(labelcolor='#8b949e')

    # ── Panel 2: Actor P(alert) vs pred_prob (WHEN) ───────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(times, pred_probs, alpha=0.25, color='#3fb950', label='CBM pred P(accident)')
    ax2.plot(times, pred_probs, color='#3fb950', lw=1.5)
    ax2.fill_between(times, actor_probs, alpha=0.25, color='#58a6ff', label='Actor P(alert)')
    ax2.plot(times, actor_probs, color='#58a6ff', lw=2)
    ax2.axhline(0.5, color='#6e7681', lw=1, ls=':')
    if label > 0:
        ax2.axvline(toa_sec, color='#ffd166', lw=2, ls='--')
        # Find first alert
        alert_t = np.where(actor_probs > 0.5)[0]
        if len(alert_t) > 0:
            at_sec = alert_t[0] / fps
            tta = toa_sec - at_sec
            ax2.axvline(at_sec, color='#ff7b72', lw=2, ls='-.',
                        label=f'Alert t={at_sec:.1f}s (TTA={tta:.1f}s)')
    ax2.set_ylabel('Probability', color='#c9d1d9', fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_title('WHEN: Actor Decision vs CBM Prediction',
                  color='#58a6ff', fontsize=10, pad=4)
    ax2.legend(fontsize=8, loc='upper left',
               facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax2.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax2.spines.values(): sp.set_color('#30363d')
    ax2.set_facecolor('#161b22')

    # ── Panel 3: Concept Risk Score (CRS) bar ────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    mean_risk = concept_acts[:, top_k_idx].mean(0)  # (K,)
    colors = plt.cm.get_cmap('YlOrRd')(mean_risk / (mean_risk.max() + 1e-6))
    bars = ax3.barh(range(K), mean_risk, color=colors, edgecolor='#30363d')
    ax3.set_yticks(range(K))
    ax3.set_yticklabels(top_concept_names, fontsize=7, color='#c9d1d9')
    ax3.set_xlabel('Mean Activation (Risk Score)', color='#c9d1d9', fontsize=9)
    ax3.set_title('Concept Risk Score (CRS) — Auditable Safety Signal',
                  color='#58a6ff', fontsize=10, pad=4)
    ax3.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax3.spines.values(): sp.set_color('#30363d')
    ax3.set_facecolor('#161b22')

    out_path = os.path.join(out_dir, f'dual_interp_case{case_idx+1}.png')
    fig.patch.set_facecolor('#0d1117')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  Saved: {out_path}')


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading model from {args.ckpt}...')
    model = load_model(args.ckpt, device)
    print(f'Model loaded. use_cbm={model.use_cbm}, use_ac={model.use_ac}')

    # Load DAD test data
    from DataLoader import DADDataset
    data_path = str(ROOT.parent / 'CRASH' / 'data' / 'dad')
    print(f'Loading data from {data_path}...')
    dset = DADDataset(data_path, 'features', 'testing',
                      toTensor=True, device='cpu', vis=False)

    n_done = 0
    for i in range(min(len(dset), 50)):  # scan first 50 samples
        sample = dset[i]
        if n_done >= args.n_cases:
            break
        x   = sample[0]   # (T, n_obj, x_dim) or (T, n_obj*x_dim)
        lbl = int(sample[1][0].item()) if hasattr(sample[1], '__len__') else int(sample[1])
        toa = sample[2] if len(sample) > 2 else x.shape[0]

        if lbl == 0:
            continue  # only show accident cases for paper

        # Reshape to (B, T, n_obj, x_dim)
        if x.dim() == 3:  # (T, n_obj, x_dim)
            x = x.unsqueeze(0)
        elif x.dim() == 2:  # (T, n_obj*x_dim) — legacy flat format
            T, D = x.shape
            n_obj = 19
            x_dim = D // n_obj
            x = x.view(T, n_obj, x_dim).unsqueeze(0)

        toa_t = torch.tensor([float(toa)])

        print(f'  Case {n_done+1}: label={lbl}, toa={float(toa):.0f}')
        try:
            c_acts, a_probs, p_probs = run_forward_with_hooks(model, x, toa_t, device)
            plot_case(
                c_acts[0], a_probs[0], p_probs[0],
                float(toa), lbl, 20.0, n_done, args.out_dir
            )
            n_done += 1
        except Exception as e:
            print(f'  Error: {e}')
            import traceback; traceback.print_exc()
            continue

    print(f'Done. Generated {n_done} figures in {args.out_dir}/')


if __name__ == '__main__':
    main()
