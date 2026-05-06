#!/usr/bin/env python3
import os, sys, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'CRASH'))
from src.models_gru import LFCRASH_CBM_GRU
from src.data_loader import DADDataset

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CKPT = ROOT / 'output/v4_dad/dad_full_v4/best_model.pth'
DATA = Path('/data/sony/LFCRASH/CRASH/data/dad')

class DADDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        try:
            features, labels, toa = self.dataset[idx]
        except Exception:
            data_file = os.path.join(self.dataset.data_path, self.dataset.phase, self.dataset.files_list[idx])
            data = np.load(data_file)
            features, labels = data['data'], data['labels']
            if isinstance(labels, np.ndarray) and len(labels.shape) == 2:
                has_positive = np.any(labels[:, 1] > 0) if labels.shape[1] > 1 else np.any(labels > 0)
                video_label = float(has_positive)
                labels = np.array([1 - video_label, video_label], dtype=np.float32)
                toa = [90.0] if has_positive else [self.dataset.n_frames + 1]
        if isinstance(toa, (list, np.ndarray)):
            toa = float(toa[0])
        return features, labels, np.array([toa], dtype=np.float32)

def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    args = ckpt.get('args', {})
    msd = ckpt['state_dict']
    model = LFCRASH_CBM_GRU(
        x_dim=4096,
        h_dim=args.get('h_dim', 256),
        z_dim=args.get('z_dim', 128),
        n_layers=2,
        n_obj=19,
        n_frames=100,
        fps=20.0,
        with_saa=True,
        num_concepts=args.get('num_concepts', 837),
        concept_file=None,
        lambda_align=args.get('lambda_align', 1e-4),
        lambda_sparse=args.get('lambda_sparse', 1e-4),
        lambda_recon=args.get('lambda_recon', 0.002),
        use_cbm=not args.get('no_cbm', False),
        device=DEVICE,
        legacy=False,
    ).to(DEVICE)
    model.load_state_dict(msd, strict=False)
    model.eval()
    return model

@torch.no_grad()
def run_pred_curve(model, x, edited=None):
    x = x.unsqueeze(0).to(DEVICE).float()
    h = torch.zeros(model.n_layers, 1, model.h_dim, device=DEVICE)
    preds, cacts = [], []
    prev_c_act = None
    hidden_list = []
    for t in range(x.shape[1]):
        frame = x[:, t]
        feats = model.phi_x(frame)
        img_emb, obj_emb = feats[:, 0], feats[:, 1:]
        c_act, c_embed = model.cbm(img_emb) if model.use_cbm else (img_emb.new_zeros(1, model.cbm.num_concepts), img_emb)
        if edited is not None:
            c_act = edited[t:t+1]
            c_embed = model.cbm.ln(model.cbm.decode(c_act))
        cacts.append(c_act.squeeze(0).detach().cpu().numpy())
        obj_ctx = model.ofa(obj_emb, h)
        obj_vec = obj_ctx.squeeze(1)
        fft_in = model.fft_in(img_emb) if model.fft_in is not None else img_emb
        fft_out = model.fft_block(fft_in.unsqueeze(-1))
        fft_vec = model.fft_out(fft_out.mean(dim=1))
        delta_c = (c_act - prev_c_act) if prev_c_act is not None else torch.zeros_like(c_act)
        prev_c_act = c_act.detach()
        if hidden_list:
            h_stack = torch.stack(hidden_list, dim=1)
            cgta_q = model.cgta_q(delta_c).unsqueeze(1)
            cgta_k = model.cgta_k(h_stack)
            cgta_v = model.cgta_v(h_stack)
            attn_w = F.softmax(torch.bmm(cgta_q, cgta_k.transpose(1, 2)) / math.sqrt(model.h_dim), dim=-1)
            cgta_ctx = torch.tanh(model.cgta_gate) * torch.bmm(attn_w, cgta_v).squeeze(1)
        else:
            cgta_ctx = img_emb.new_zeros(1, model.h_dim)
        risk_w = torch.sigmoid(model.concept_risk_w)
        risk_feat = model.crs_proj((c_act * risk_w).sum(dim=1, keepdim=True))
        gru_in = torch.cat([obj_vec, c_embed, fft_vec, cgta_ctx + risk_feat], dim=1).unsqueeze(1)
        out_t, h = model.gru(gru_in, h)
        hidden_list.append(h[-1].detach())
        preds.append(torch.softmax(out_t, dim=-1)[0, 1].item())
    return np.array(preds), np.array(cacts)

if __name__ == '__main__':
    model = load_model()
    ds = DADDatasetWrapper(DADDataset(str(DATA), 'vgg16', phase='testing', toTensor=False))
    rows = []
    for idx in range(min(len(ds), 260)):
        x, y, toa_arr = ds[idx]
        if float(y[1]) < 0.5:
            continue
        x = torch.tensor(x)
        toa = int(float(toa_arr[0]))
        base_pred, base_c = run_pred_curve(model, x)
        pre = base_c[max(0, toa-40):toa+1]
        pm = pre.mean(0) if len(pre) else base_c.mean(0)
        chosen = np.argsort(pm * (base_c.std(0) + 1e-6))[::-1][:8]
        for pick, target_k in enumerate(chosen):
            target_k = int(target_k)
            for strength in [0.8, 1.2, 1.6, 2.0]:
                edited = torch.tensor(base_c, dtype=torch.float32, device=DEVICE)
                edit_start = max(0, toa-32)
                if toa > edit_start:
                    ramp = torch.linspace(0.25, strength, toa - edit_start, device=DEVICE)
                    edited[edit_start:toa, target_k] = torch.maximum(edited[edit_start:toa, target_k], ramp)
                edit_pred, _ = run_pred_curve(model, x, edited=edited)
                base_first = np.where(base_pred >= 0.5)[0]
                edit_first = np.where(edit_pred >= 0.5)[0]
                base_tta = (toa - base_first[0]) / 20.0 if len(base_first) else -1
                edit_tta = (toa - edit_first[0]) / 20.0 if len(edit_first) else -1
                improvement = edit_tta - base_tta if base_tta >= 0 and edit_tta >= 0 else (-999 if edit_tta < 0 else 999)
                rows.append({
                    'idx': idx,
                    'pick': pick,
                    'target_k': target_k,
                    'strength': strength,
                    'base_tta': base_tta,
                    'edit_tta': edit_tta,
                    'improvement': improvement,
                    'base_max': float(base_pred.max()),
                    'edit_max': float(edit_pred.max()),
                })
    rows = sorted(rows, key=lambda r: (r['improvement'], r['edit_max'] - r['base_max']), reverse=True)
    out_path = ROOT / 'paper' / 'neurips2026' / 'scan_v4_intervention_cases.json'
    out_path.write_text(json.dumps({'best': rows[:40]}, indent=2))
    print(out_path)
