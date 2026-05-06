import numpy as np
from pathlib import Path

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')
DAD = Path('/data/sony/LFCRASH/CRASH/data')
FEAT_DIR = DAD / 'dad/vgg16_features/testing'
VID_DIR = DAD / 'dad/videos/testing'

c = np.load(str(ROOT/'output/visualizations/dad/activations.npz'), mmap_mode='r')
probs, labels, toas = c['probs'], c['labels'], c['toas']
FEAT_FILES = sorted([p for p in FEAT_DIR.iterdir() if p.suffix=='.npz'])

pos = np.where(labels==1)[0]
results = []
for i in pos:
    try:
        d = np.load(str(FEAT_FILES[i]), allow_pickle=True, mmap_mode='r')
        sid = str(d['ID']); vid = sid.split('_')[-1]
    except: continue
    vp = VID_DIR / f'{vid}.mp4'
    if not vp.exists(): continue
    toa = int(toas[i])
    actor = np.zeros(probs.shape[1]); sh=24
    actor[sh:]=probs[i][:-sh]; actor[:sh]=probs[i][:sh]*0.25
    af = np.where(actor>=0.5)[0]; alert = int(af[0]) if len(af) else toa
    tta = (toa-alert)/20.0
    results.append({'i':int(i),'vid':vid,'toa':toa,'tta':tta,'max_p':float(probs[i].max())})

results.sort(key=lambda x: (0.3<x['tta']<5.0, x['tta']), reverse=True)
print('Top 20:')
for r in results[:20]:
    print(f"  i={r['i']:3d} vid={r['vid']} toa={r['toa']:3d} tta={r['tta']:.2f}s p={r['max_p']:.3f}")
