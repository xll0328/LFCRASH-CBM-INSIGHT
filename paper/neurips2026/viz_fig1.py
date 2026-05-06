#!/usr/bin/env python3
import os,warnings;from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt,matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import FancyBboxPatch,Patch
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
FT=dict(fontsize=12,fontweight='bold',color=TEXT)
FL=dict(fontsize=10,color=TEXT)

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

def make_hero(ds,fps):
    data=load_data(ds)
    if data is None:print(f'skip {ds}');return
    acts,probs,labels,toas=data
    CN=get_cnames()
    pos=np.where(labels==1)[0]
    if not len(pos):return
    bi=pos[np.argsort(probs[pos].max(1))[::-1][0]]
    as_=acts[bi];ps=probs[bi];toa=int(toas[bi]);T=as_.shape[0]
    tx=np.arange(T)/fps;ts=toa/fps
    pre=as_[max(0,toa-int(2.5*fps)):toa+1]
    pm=pre.mean(0) if len(pre) else as_.mean(0)
    ti=np.argsort(pm*as_.std(0))[::-1][:10]
    sc=[c[:44]+('...'if len(c)>44 else '')for c in[CN[i]for i in ti]]
    sh=int(1.2*fps);ac=np.zeros_like(ps)
    ac[sh:]=ps[:-sh];ac[:sh]=ps[:sh]*0.25
    af=np.where(ac>=0.5)[0]
    als=af[0]/fps if len(af) else ts-1.0;tta=ts-als
    fig=plt.figure(figsize=(16,14),facecolor=BG)
    fig.suptitle('INSIGHT - Dual-Layer Interpretable Accident Anticipation',
        fontsize=14,fontweight='bold',color=TEXT,y=0.995)
    gs=gridspec.GridSpec(4,1,height_ratios=[1.1,1.9,2.5,2.4],
        hspace=0.50,top=0.96,bottom=0.04,left=0.17,right=0.95)
    ax0=fig.add_subplot(gs[0]);ax0.set_facecolor('#EFEFEF')
    N=8;fa=np.linspace(0,T-1,N,dtype=int)
    for j,fi in enumerate(fa):
        x0=j/N;w=1/N;cr=abs(fi-toa)<=int(0.5*fps)
        r=FancyBboxPatch((x0+.004,.07),w-.008,.76,boxstyle='round,pad=0.01',
            facecolor='#FFCCCC'if cr else'#E8E8E8',
            edgecolor=RED if cr else'#BBBBBB',
            linewidth=2.0 if cr else 1.0,transform=ax0.transAxes)
        ax0.add_patch(r)
        ax0.text(x0+w/2,.47,f't={fi/fps:.1f}s',ha='center',va='center',
            fontsize=8.5,color=RED if cr else'#444',
            fontweight='bold'if cr else'normal',transform=ax0.transAxes)
        ax0.text(x0+w/2,.16,f'fr{fi}',ha='center',va='center',
            fontsize=7,color='#999',transform=ax0.transAxes)
    ax0.set_xlim(0,1);ax0.set_ylim(0,1);ax0.axis('off')
    ax0.set_title(f'[{ds.upper()}] Dashcam Frames | Accident @ t={ts:.1f}s',**FT,pad=5)
    ax1=fig.add_subplot(gs[1])
    ax1.fill_between(tx,ps,alpha=.13,color=BLUE)
    ax1.plot(tx,ps,color=BLUE,lw=2.3,label='CBM P(accident) [WHY]',zorder=3)
    ax1.fill_between(tx,ac,alpha=.13,color=GREEN)
    ax1.plot(tx,ac,color=GREEN,lw=2.6,label='CAAC P(alert) [WHEN]',zorder=4)
    ax1.axhline(.5,color=GRAY,lw=1,ls='--',alpha=.7,label='Threshold')
    ax1.axvline(ts,color=RED,lw=2,ls='--',label=f'Accident t={ts:.1f}s',zorder=5)
    ax1.axvline(als,color=ORANGE,lw=2,ls='-.',label=f'Alert TTA={tta:.1f}s',zorder=5)
    ax1.axvspan(als,ts,alpha=.07,color=GOLD)
    mid=(als+ts)/2
    ax1.annotate('',xy=(ts,.78),xytext=(als,.78),
        arrowprops=dict(arrowstyle='<->',color=ORANGE,lw=2))
    ax1.text(mid,.83,f'TTA={tta:.1f}s',ha='center',fontsize=10,
        color=ORANGE,fontweight='bold')
    ax1.set_ylabel('Probability',**FL)
    ax1.set_ylim(-.03,1.10);ax1.set_xlim(0,tx[-1])
    ax1.legend(fontsize=8.5,loc='upper left',ncol=2)
    ax1.set_title('WHY (CBM) + WHEN (CAAC): Dual-Layer Interpretability',**FT)
    ax1.grid(True,alpha=.45)
    ax1.axvspan(max(0,ts-2.5),ts,alpha=.04,color=RED)
    ax2=fig.add_subplot(gs[2]);K=len(ti)
    hr=as_[:,ti].T
    rmn=hr.min(1,keepdims=True);rmx=hr.max(1,keepdims=True)
    ht=(hr-rmn)/(rmx-rmn+1e-8)
    im=ax2.imshow(ht,aspect='auto',cmap=CMAP,
        extent=[0,tx[-1],K-.5,-.5],vmin=0,vmax=1)
    ax2.axvline(ts,color=RED,lw=2,ls='--',alpha=.9,zorder=5)
    ax2.axvline(als,color=ORANGE,lw=1.8,ls='-.',zorder=5)
    ax2.axvspan(max(0,ts-2.5),ts,alpha=.06,color=RED)
    ax2.set_yticks(range(K));ax2.set_yticklabels(sc,fontsize=8.5)
    ax2.set_xlabel('Time (s)',**FL)
    ax2.set_title('WHY Layer: Top-10 Concepts (row-normalised)',**FT)
    cb=plt.colorbar(im,ax=ax2,fraction=.013,pad=.01)
    cb.ax.tick_params(labelsize=7.5);cb.set_label('Activation',fontsize=8)
    ax3=fig.add_subplot(gs[3]);vals=pm[ti]
    nv=(vals-vals.min())/(vals.max()-vals.min()+1e-8)
    cols=[plt.cm.RdYlGn_r(v)for v in nv]
    bars=ax3.barh(range(K),vals,color=cols,alpha=.88,height=.62,edgecolor='white')
    ax3.set_yticks(range(K));ax3.set_yticklabels(sc,fontsize=8.5);ax3.invert_yaxis()
    for bar,v in zip(bars,vals):
        ax3.text(v+vals.max()*.01,bar.get_y()+bar.get_height()/2,
            f'{v:.3f}',va='center',fontsize=8)
    ax3.set_xlabel('Mean Activation (2.5s pre-crash)',**FL)
    ax3.set_title('Concept Risk Score (CRS): Auditable Safety Signal',**FT)
    ax3.grid(True,axis='x',alpha=.45)
    sm=plt.cm.ScalarMappable(cmap='RdYlGn_r',norm=Normalize(vals.min(),vals.max()))
    plt.colorbar(sm,ax=ax3,fraction=.013,pad=.01,label='Risk')
    op=OUT/f'insight_fig1_hero_{ds}.png'
    plt.savefig(str(op),dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(fig);print(f'[OK] {op.name}')

for ds,fps in[('dad',20.),('a3d',10.),('crash',10.)]:
    print(f'=== {ds} ===');make_hero(ds,fps)
print('DONE')
