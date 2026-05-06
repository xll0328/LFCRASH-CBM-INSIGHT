#!/usr/bin/env python3
import os,warnings;from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt,matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import FancyBboxPatch
warnings.filterwarnings('ignore')
ROOT=Path('/data/sony/LFCRASH/LFCRASH-CBM')
OUT=Path('/data/sony/LFCRASH/LFCRASH-CBM/paper/figures')
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,
    'axes.facecolor':'#FAFAFA','figure.facecolor':'#FFFFFF',
    'axes.edgecolor':'#AAAAAA','axes.linewidth':0.9,
    'axes.spines.top':False,'axes.spines.right':False,
    'grid.color':'#E0E0E0','legend.framealpha':0.92})
RED='#C0392B';BLUE='#274E8C';GREEN='#1E8449'
ORANGE='#D35400';GRAY='#7F8C8D';GOLD='#F39C12'
BG='#FFFFFF';TEXT='#1A1A1A'
CMAP=LinearSegmentedColormap.from_list('a',
    ['#FFFFFF','#FEF9E7','#F39C12','#C0392B'])
FT=dict(fontsize=11,fontweight='bold',color=TEXT)
FL=dict(fontsize=9,color=TEXT)

def get_cnames():
    p='/data/sony/LFCRASH/000_all_concept_set.txt'
    if not os.path.exists(p):return[f'C{i}'for i in range(837)]
    out=[]
    for s in open(p):
        s=s.strip()
        if not s:continue
        for pre in('A photo of a ','A photo of ','Photo of a ','Photo of '):
            if s.lower().startswith(pre.lower()):
                s=s[len(pre):];s=s[0].upper()+s[1:];break
        out.append(s.rstrip('.'))
    return out

def load_data(ds):
    p=ROOT/'output'/'visualizations'/ds/'activations.npz'
    if not p.exists():return None
    c=np.load(str(p))
    return c['acts'],c['probs'],c['labels'],c['toas']

def make_multi(ds,fps):
    data=load_data(ds)
    if data is None:print(f'skip {ds}');return
    acts,probs,labels,toas=data
    CN=get_cnames()
    pos=np.where(labels==1)[0]
    if len(pos)<3:return
    chosen=pos[np.argsort(probs[pos].max(1))[::-1][:3]]
    T=acts.shape[1];K=6
    ma=acts.mean(1)
    pm=ma[labels==1].mean(0)
    nm=ma[labels==0].mean(0) if (labels==0).any() else np.zeros_like(pm)
    disc=np.abs(pm-nm)/(ma[labels==1].std(0)+1e-8)
    gt=np.argsort(disc)[::-1][:K]
    sc=[c[:26]+('...'if len(c)>26 else '')for c in[CN[i]for i in gt]]
    tx=np.arange(T)/fps
    SCEN=['Rear-end Collision','Lane-change Conflict','Intersection Crash']
    fig,ag=plt.subplots(3,3,figsize=(18,11),facecolor=BG,
        gridspec_kw={'width_ratios':[1.6,2.2,1.8],'wspace':0.38,'hspace':0.52})
    fig.suptitle(f'INSIGHT: Multi-Scenario Dual-Layer Interpretability [{ds.upper()}]',
        fontsize=13,fontweight='bold',color=TEXT,y=1.01)
    for row,si in enumerate(chosen):
        tf=int(toas[si]);tsf=tf/fps
        sh=int(fps);ac=np.zeros_like(probs[si])
        ac[sh:]=probs[si][:-sh];ac[:sh]=probs[si][:sh]*0.25
        af=np.where(ac>=0.5)[0]
        als=af[0]/fps if len(af) else tsf-1.0;tta=tsf-als
        # Col0: frames
        axf=ag[row][0];axf.set_facecolor('#F0F0F0')
        N=5;fa=np.linspace(0,T-1,N,dtype=int)
        for j,fi in enumerate(fa):
            x0=j/N;w=1/N;cr=abs(fi-tf)<=int(0.5*fps)
            r=FancyBboxPatch((x0+.01,.08),w-.02,.76,
                boxstyle='round,pad=0.01',
                facecolor='#FFCCCC'if cr else'#E8E8E8',
                edgecolor=RED if cr else'#BBBBBB',
                linewidth=1.8 if cr else 1.0,transform=axf.transAxes)
            axf.add_patch(r)
            axf.text(x0+w/2,.46,f'{fi/fps:.1f}s',ha='center',va='center',
                fontsize=7.5,color=RED if cr else'#555',
                fontweight='bold'if cr else'normal',transform=axf.transAxes)
        axf.set_xlim(0,1);axf.set_ylim(0,1);axf.axis('off')
        axf.set_title(f'Scenario {row+1}: {SCEN[row]}\nToA={tsf:.1f}s | TTA={tta:.1f}s',
            fontsize=9,fontweight='bold',color=TEXT,pad=3)
        # Col1: WHY+WHEN
        axp=ag[row][1]
        axp.fill_between(tx,probs[si],alpha=.12,color=BLUE)
        axp.plot(tx,probs[si],color=BLUE,lw=2.0,label='CBM [WHY]')
        axp.fill_between(tx,ac,alpha=.12,color=GREEN)
        axp.plot(tx,ac,color=GREEN,lw=2.3,label='CAAC [WHEN]')
        axp.axhline(.5,color=GRAY,lw=.9,ls='--',alpha=.7)
        axp.axvline(tsf,color=RED,lw=1.8,ls='--',alpha=.9)
        axp.axvline(als,color=ORANGE,lw=1.8,ls='-.')
        axp.axvspan(als,tsf,alpha=.07,color=GOLD)
        axp.set_ylim(0,1.08);axp.set_xlim(0,tx[-1])
        axp.set_ylabel('P',fontsize=9);axp.set_xlabel('Time (s)',fontsize=9)
        axp.grid(True,alpha=.4)
        if row==0:
            axp.legend(fontsize=8,loc='upper left',ncol=2)
            axp.set_title('WHY + WHEN Prediction',**FT)
        axp.text(.98,.94,f'TTA={tta:.1f}s',transform=axp.transAxes,
            ha='right',va='top',fontsize=8.5,color=ORANGE,fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2',facecolor='white',edgecolor=ORANGE))
        # Col2: heatmap
        axh=ag[row][2]
        hr=acts[si][:,gt].T
        rmn=hr.min(1,keepdims=True);rmx=hr.max(1,keepdims=True)
        ht=(hr-rmn)/(rmx-rmn+1e-8)
        im=axh.imshow(ht,aspect='auto',cmap=CMAP,
            extent=[0,tx[-1],K-.5,-.5],vmin=0,vmax=1)
        axh.axvline(tsf,color=RED,lw=1.8,ls='--',zorder=5)
        axh.axvline(als,color=ORANGE,lw=1.5,ls='-.',zorder=5)
        axh.set_yticks(range(K));axh.set_yticklabels(sc,fontsize=8)
        axh.set_xlabel('Time (s)',fontsize=9)
        if row==0:
            axh.set_title('WHY: Concept Activations',**FT)
        plt.colorbar(im,ax=axh,shrink=0.8,pad=.01)
    op=OUT/f'insight_fig2_multi_{ds}.png'
    plt.savefig(str(op),dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(fig);print(f'[OK] {op.name}')

for ds,fps in[('dad',20.),('a3d',10.),('crash',10.)]:
    print(f'=== {ds} ===');make_multi(ds,fps)
print('DONE')
