#!/usr/bin/env python3
"""eval_dad_sota.py - Eval best DAD checkpoints to get full metrics"""
import os, sys, json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

ROOT       = Path(__file__).resolve().parent
CRASH_ROOT = ROOT.parent / 'CRASH'
DATA_ROOT  = CRASH_ROOT / 'data'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRASH_ROOT))

from src.models_gru  import LFCRASH_CBM_GRU
from src.data_loader import DADDataset
from src.eval_tools  import evaluation

TARGETS = [
    {'name':'dad_z512',    'ckpt':'output/dad_sota_push/dad_z512/best_model.pt',
     'h_dim':256,'z_dim':256,'gpu':6},
    {'name':'dad_h512_v2', 'ckpt':'output/dad_sota_push/dad_h512_v2/best_model.pt',
     'h_dim':256,'z_dim':256,'gpu':6},
]

def collate_fn(batch):
    xs,ys,toas=zip(*batch)
    xs=torch.from_numpy(np.stack(xs)).float()
    ys=torch.from_numpy(np.stack(ys)).float()
    tf=[float(t[0]) if hasattr(t,'__len__') and len(t)>0 else float(t) for t in toas]
    return xs,ys,torch.tensor(tf,dtype=torch.float32)

@torch.no_grad()
def run(target):
    name=target['name']; gpu=target['gpu']
    ckpt_p=ROOT/target['ckpt']
    device=torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'\n=== {name} ===')
    ckpt=torch.load(str(ckpt_p),map_location=device,weights_only=False)
    state=ckpt.get('model_state_dict',ckpt.get('state_dict',ckpt))
    has_cgta=any('cgta' in k for k in state.keys())
    print(f'  epoch={ckpt.get("epoch","?")} AP_ckpt={ckpt.get("AP",0):.4f} has_cgta={has_cgta}')
    # Try to detect h_dim from state dict
    h_dim=target['h_dim']; z_dim=target['z_dim']
    if 'phi_x.0.weight' in state:
        h_dim2 = state['phi_x.0.weight'].shape[0]//2
        print(f'  detected h_dim={h_dim2} (from phi_x.0.weight)')
        h_dim=h_dim2
    if 'cbm.concept_proj.3.weight' in state:
        z_dim2 = state['cbm.concept_proj.3.weight'].shape[0]  
        # actually z_dim is concept bottleneck output
        pass
    model=LFCRASH_CBM_GRU(
        x_dim=4096,h_dim=h_dim,z_dim=z_dim,n_layers=2,
        n_obj=19,n_frames=100,fps=20.0,with_saa=True,
        num_concepts=837,concept_file=None,
        lambda_align=1e-6,lambda_sparse=5e-5,lambda_recon=1e-4,
        use_cbm=True,device=str(device),legacy=not has_cgta,
    ).to(device)
    missing,unexpected=model.load_state_dict(state,strict=False)
    if missing: print(f'  missing({len(missing)}): {missing[:3]}')
    model.eval()
    te=DADDataset(str(DATA_ROOT/'dad'),'vgg16',phase='testing',toTensor=False)
    loader=DataLoader(te,batch_size=32,shuffle=False,num_workers=2,collate_fn=collate_fn)
    print(f'  test samples: {len(te)}')
    all_pred,all_labels,all_toas=[],[],[]
    for xs,ys,toas in loader:
        xs=xs.to(device)
        _,outputs,_=model(xs,None,None)
        T=len(outputs)
        fp=np.zeros((xs.size(0),100),dtype=np.float32)
        for t_i,out_t in enumerate(outputs):
            fp[:,t_i]=torch.softmax(out_t,dim=-1)[:,1].cpu().numpy()
        if T<100: fp[:,T:]=fp[:,T-1:T]
        all_pred.append(fp)
        all_labels.append(ys[:,1].numpy())
        all_toas.append(toas.numpy())
    all_pred=np.concatenate(all_pred,0)
    all_labels=np.concatenate(all_labels,0)
    all_toas=np.concatenate(all_toas,0)
    AP,mTTA,TTA_R80,P_R80=evaluation(all_pred,all_labels,all_toas,fps=20.0)
    print(f'  AP={AP:.4f} mTTA={mTTA:.4f} TTA_R80={TTA_R80:.4f} P_R80={P_R80:.4f}')
    res={'AP':AP,'mTTA':mTTA,'TTA_R80':TTA_R80,'P_R80':P_R80,
         'epoch':ckpt.get('epoch','?'),'dataset':'dad','tag':name}
    out=ROOT/'output'/'dad_sota_push'/name/'results.json'
    with open(out,'w') as f: json.dump(res,f,indent=2)
    print(f'  Saved -> {out}')
    return res

if __name__=='__main__':
    os.chdir(ROOT)
    for t in TARGETS:
        try: run(t)
        except Exception as e:
            import traceback; print(f'ERROR {t["name"]}: {e}'); traceback.print_exc()
