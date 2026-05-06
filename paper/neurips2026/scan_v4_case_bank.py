#!/usr/bin/env python3
import os, sys, json
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
CONCEPT_FILE = ROOT.parent / '000_all_concept_set.txt'
OUT_JSON = ROOT / 'paper' / 'neurips2026' / 'v4_case_bank_scan.json'

class DADDatasetWrapper:
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
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
    args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    msd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else (ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt)
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
    ret = model.load_state_dict(msd, strict=False)
    model.eval()
    return model, ret

@torch.no_grad()
def run_one(model, x):
    x = x.unsqueeze(0).to(DEVICE).float()
    h = torch.zeros(model.n_layers, 1, model.h_dim, device=DEVICE)
    preds, alerts, cacts = [], [], []
    prev_c_act = None
    hidden_list = []
    for t in range(x.shape[1]):
        frame = x[:, t]
        feats = model.phi_x(frame)
        img_emb, obj_emb = feats[:, 0], feats[:, 1:]
        if model.use_cbm:
            c_act, c_embed = model.cbm(img_emb)
        else:
            c_act = img_emb.new_zeros(1, model.cbm.num_concepts)
            c_embed = img_emb
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
            attn_w = F.softmax(torch.bmm(cgta_q, cgta_k.transpose(1, 2)) / np.sqrt(model.h_dim), dim=-1)
            cgta_ctx = torch.tanh(model.cgta_gate) * torch.bmm(attn_w, cgta_v).squeeze(1)
        else:
            cgta_ctx = img_emb.new_zeros(1, model.h_dim)
        risk_w = torch.sigmoid(model.concept_risk_w)
        risk_feat = model.crs_proj((c_act * risk_w).sum(dim=1, keepdim=True))
        gru_in = torch.cat([obj_vec, c_embed, fft_vec, cgta_ctx + risk_feat], dim=1).unsqueeze(1)
        out_t, h = model.gru(gru_in, h)
        hidden_list.append(h[-1].detach())
        preds.append(torch.softmax(out_t, dim=-1)[0, 1].item())
        if model.use_ac:
            al, _, _ = model.ac_module(h[-1], c_act)
            alerts.append(torch.softmax(al, dim=-1)[0, 1].item())
        else:
            alerts.append(preds[-1])
    return np.array(preds), np.array(alerts), np.array(cacts)

def clean_name(s):
    s = s.strip()
    for pre in ('A photo of a ', 'A photo of ', 'Photo of a ', 'Photo of '):
        if s.lower().startswith(pre.lower()):
            return s[len(pre):]
    return s

if __name__ == '__main__':
    names = [clean_name(x) for x in CONCEPT_FILE.read_text().splitlines()]
    model, ret = load_model()
    ds = DADDatasetWrapper(DADDataset(str(DATA), 'vgg16', phase='testing', toTensor=False))
    rows = []
    for idx in range(len(ds)):
        x, y, toa_arr = ds[idx]
        if float(y[1]) < 0.5:
            continue
        x = torch.tensor(x)
        toa = int(float(toa_arr[0]))
        preds, alerts, cacts = run_one(model, x)
        pred_first = np.where(preds >= 0.5)[0]
        alert_first = np.where(alerts >= 0.5)[0]
        pred_tta = (toa - pred_first[0]) / 20.0 if len(pred_first) else -1
        alert_tta = (toa - alert_first[0]) / 20.0 if len(alert_first) else -1
        pre = cacts[max(0, toa-40):toa+1]
        mean_pre = pre.mean(0) if len(pre) else cacts.mean(0)
        std_all = cacts.std(0)
        risk_score = mean_pre * (std_all + 1e-6)
        top = np.argsort(risk_score)[::-1][:8]
        rows.append({
            'idx': idx,
            'toa': toa,
            'pred_tta': pred_tta,
            'alert_tta': alert_tta,
            'pred_max': float(preds.max()),
            'alert_max': float(alerts.max()),
            'top_idx': [int(k) for k in top.tolist()],
            'top_names': [names[int(k)] for k in top.tolist()],
        })
    late = [r for r in rows if 0 <= r['alert_tta'] < 0.5]
    late = sorted(late, key=lambda r: (r['alert_max'], r['pred_max']), reverse=True)
    failed = [r for r in rows if r['alert_tta'] < 0]
    failed = sorted(failed, key=lambda r: (r['alert_max'], r['pred_max']), reverse=True)
    strong = sorted([r for r in rows if r['alert_tta'] >= 0], key=lambda r: r['alert_tta'], reverse=True)
    out = {
        'num_pos': len(rows),
        'num_late': len(late),
        'num_failed': len(failed),
        'num_missing': len(ret.missing_keys),
        'num_unexpected': len(ret.unexpected_keys),
        'late_top': late[:30],
        'failed_top': failed[:30],
        'strong_top': strong[:30],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({'saved': str(OUT_JSON), 'num_pos': len(rows), 'num_late': len(late), 'num_failed': len(failed)}, indent=2, ensure_ascii=False))
