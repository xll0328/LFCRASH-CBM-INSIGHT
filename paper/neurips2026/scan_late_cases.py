#!/usr/bin/env python3
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'CRASH'))
from train_best_params import BEST_PARAMS, DATASET_PARAMS
from src.models_gru import LFCRASH_CBM_GRU
from src.data_loader import DADDataset

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CKPT = ROOT / 'output/full_training_20260114_121308/best_dad/best_model.pth'
CONCEPT_FILE = str(ROOT.parent / '000_all_concept_set.txt')
OUT_JSON = ROOT / 'paper' / 'neurips2026' / 'late_case_scan.json'

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

params = BEST_PARAMS['dad']
ds = DATASET_PARAMS['dad']
base = DADDataset('/data/sony/LFCRASH/CRASH/data/dad', ds['feature'], phase=ds['phase_test'], toTensor=False)
dataset = DADDatasetWrapper(base)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

model = LFCRASH_CBM_GRU(
    x_dim=ds['x_dim'], h_dim=params['h_dim'], z_dim=params['z_dim'], n_layers=2,
    n_obj=ds['n_obj'], n_frames=ds['n_frames'], fps=ds['fps'], with_saa=True,
    num_concepts=837, concept_file=CONCEPT_FILE, lambda_align=params['lambda_align'],
    lambda_sparse=params['lambda_sparse'], device=DEVICE).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

names = Path(CONCEPT_FILE).read_text().splitlines()
risk_w = model.get_concept_risk_weights()
out = []
with torch.no_grad():
    for idx, (x, y, toa) in enumerate(loader):
        if y[0,1].item() < 0.5:
            continue
        x = x.to(DEVICE).float(); y = y.to(DEVICE).float(); toa = toa.to(DEVICE).float().flatten()
        losses, outputs, hiddens = model(x, y, toa)
        probs = torch.softmax(torch.stack(outputs, dim=1).squeeze(0), dim=-1)[:,1].cpu().numpy()
        acts = model.get_concept_activations(x).squeeze(0).cpu().numpy()
        hidden = torch.stack(hiddens, dim=1).squeeze(0)
        ac = []
        for t in range(hidden.shape[0]):
            logits, _, _ = model.ac_module(hidden[t:t+1], torch.tensor(acts[t:t+1], device=DEVICE, dtype=torch.float32))
            ac.append(torch.softmax(logits, dim=-1)[0,1].item())
        ac = np.array(ac)
        af = np.where(ac >= 0.5)[0]
        alert = int(af[0]) if len(af) else 999
        tta = (toa.item() - alert) / ds['fps'] if alert < 999 else -1
        pre = acts[max(0,int(toa.item())-40):int(toa.item())+1]
        top = np.argsort((pre.mean(0) if len(pre) else acts.mean(0)) * risk_w)[::-1][:8]
        top_names = []
        for k in top:
            s = names[k].strip()
            for prefx in ('A photo of a ','A photo of ','Photo of a ','Photo of '):
                if s.lower().startswith(prefx.lower()):
                    s = s[len(prefx):]
                    break
            top_names.append(s)
        out.append({
            'idx': idx,
            'alert': alert,
            'tta': tta,
            'max_ac': float(ac.max()),
            'max_prob': float(probs.max()),
            'top_idx': [int(k) for k in top.tolist()],
            'top_names': top_names,
        })

late = [o for o in out if 0 <= o['tta'] < 0.5]
late = sorted(late, key=lambda z: (z['max_ac'], z['max_prob']), reverse=True)
failed = [o for o in out if o['tta'] < 0]
failed = sorted(failed, key=lambda z: z['max_prob'], reverse=True)
strong = sorted(out, key=lambda z: z['tta'], reverse=True)[:20]
result = {
    'num_pos': len(out),
    'num_late': len(late),
    'num_failed': len(failed),
    'late_top': late[:25],
    'failed_top': failed[:25],
    'strong_top': strong,
}
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps({'saved': str(OUT_JSON), 'num_pos': len(out), 'num_late': len(late), 'num_failed': len(failed)}, indent=2, ensure_ascii=False))
