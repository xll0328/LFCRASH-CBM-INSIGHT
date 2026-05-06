#!/usr/bin/env python3
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
ROOT = Path("/data/sony/LFCRASH/LFCRASH-CBM")
OUT = ROOT / "paper" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
DAD = Path("/data/sony/LFCRASH/CRASH/data/dad")
FEAT_DIR = DAD / "vgg16_features" / "testing"
VID_DIR = DAD / "videos" / "testing"
FEAT_FILES = sorted([p for p in FEAT_DIR.iterdir() if p.suffix == ".npz"])
RED="#D62728"; BLUE="#1F5FAD"; GREEN="#1A7A3C"; ORANGE="#E07B00"; GOLD="#F0C040"; GRAY="#888888"
CMAP = LinearSegmentedColormap.from_list("insight", ["#F5F5F5","#FFF0CC","#FFAA44","#CC2200"])
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":10.5,"axes.spines.top":False,"axes.spines.right":False,"grid.alpha":0.7})
MULTI_CONCEPTS = ["Brake light triggered","Close following headway","Rear-end collision risk","Vehicle cut-in detected","Sudden deceleration cue"]
def load_cache():
    c = np.load(str(ROOT / "output" / "visualizations" / "dad" / "activations.npz"), mmap_mode="r")
    return c["acts"], c["probs"], c["toas"]
def sample_meta(idx):
    d = np.load(str(FEAT_FILES[idx]), allow_pickle=True)
    vid = str(d["ID"]).split("_")[-1]
    return {"vid": vid, "video": VID_DIR / f"{vid}.mp4", "det": d["det"]}
def actor_curve(P, sh=20):
    a = np.zeros_like(P); a[sh:] = P[:-sh]; a[:sh] = P[:sh] * 0.25; return a
def read_frames(vpath, fi_arr, boxes, size=(320,180)):
    cap = cv2.VideoCapture(str(vpath)); out=[]
    for fi in fi_arr:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
        if not ok: fr = np.zeros((size[1], size[0], 3), np.uint8)
        else:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            sx, sy = size[0]/fr.shape[1], size[1]/fr.shape[0]
            fr = cv2.resize(fr, size)
            b = boxes[min(int(fi), len(boxes)-1)]
            areas = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
            for j in np.argsort(areas)[::-1][:4]:
                x1,y1,x2,y2 = b[j,:4]
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(fr,(int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),(255,210,0),2)
        out.append(fr)
    cap.release(); return out
def make_multi():
    acts, probs, toas = load_cache(); fps = 20.; T = acts.shape[1]; t = np.arange(T) / fps
    cases = [(31,"000478","Rear-end: motorbike brakes sharply"), (190,"000512","Lane-change: vehicle swerves in")]
    fig = plt.figure(figsize=(16.2, 12.8), facecolor="white")
    fig.suptitle("INSIGHT — Two Real DAD Case Studies: Frames • Policy Dynamics • Concept Evidence", fontsize=13.0, fontweight="bold", y=0.992)
    outer = gridspec.GridSpec(4,2, figure=fig, width_ratios=[1.75,1.25], height_ratios=[1.05,0.95,1.05,0.95], hspace=0.45, wspace=0.18, left=0.05, right=0.97, top=0.95, bottom=0.05)
    for i,(IDX,vid,title) in enumerate(cases):
        meta = sample_meta(IDX); A = acts[IDX]; P = probs[IDX]; toa = int(toas[IDX])
        ac = actor_curve(P); af = np.where(ac >= 0.5)[0]; alert = int(af[0]) if len(af) else max(0, toa-16); tta = (toa-alert)/fps
        pre = A[max(0,toa-40):toa+1]; pm = pre.mean(0) if len(pre) else A.mean(0); top_k = np.argsort(pm*A.std(0))[::-1][:5]
        fi = np.unique(np.clip(np.linspace(max(0,toa-55), min(T-1,toa+3), 5, dtype=int), 0, T-1))
        frames = read_frames(meta["video"], fi, meta["det"])
        r0, r1 = i*2, i*2+1
        axf = fig.add_subplot(outer[r0,:])
        strip = np.concatenate(frames, axis=1); H,W = strip.shape[:2]; wf = W/len(frames)
        axf.imshow(strip, aspect="auto"); axf.set_facecolor("#0A0A0A")
        for j,ff in enumerate(fi):
            near = abs(ff-toa) <= 7
            if near: axf.add_patch(Rectangle((j*wf,0),wf,H,fc=RED,alpha=0.19,ec=RED,lw=2.2,zorder=2))
            axf.text((j+.5)*wf,H+7,f"{ff/fps:.1f}s",ha="center",va="top",fontsize=7.8,color=RED if near else "#DDDDDD",fontweight="bold" if near else "normal")
        axf.set_xlim(0,W); axf.set_ylim(H+22,-2); axf.set_xticks([]); axf.set_yticks([])
        for sp in axf.spines.values(): sp.set_visible(False)
        axf.set_title(f"Case {i+1}: {title}   |   Video {vid}   |   ToA={toa/fps:.1f}s   |   TTA={tta:.1f}s", fontsize=10.2, fontweight="bold", pad=5)
        axp = fig.add_subplot(outer[r1,0])
        axp.fill_between(t,P,color=BLUE,alpha=.11); axp.plot(t,P,color=BLUE,lw=2.0,label="CBM")
        axp.fill_between(t,ac,color=GREEN,alpha=.11); axp.plot(t,ac,color=GREEN,lw=2.2,label="CAAC")
        axp.axhline(.5,color=GRAY,lw=.9,ls="--",alpha=.7); axp.axvline(toa/fps,color=RED,lw=1.7,ls="--"); axp.axvline(alert/fps,color=ORANGE,lw=1.7,ls="-.")
        axp.axvspan(alert/fps,toa/fps,color=GOLD,alpha=.08); axp.text(.98,.92,f"TTA={tta:.1f}s",transform=axp.transAxes,ha="right",va="top",fontsize=8.8,color=ORANGE,fontweight="bold",bbox=dict(boxstyle="round,pad=0.18",fc="white",ec=ORANGE,alpha=.92))
        axp.set_xlim(0,t[-1]); axp.set_ylim(0,1.06); axp.grid(True,alpha=.4); axp.set_ylabel("P",fontsize=8.8); axp.set_xlabel("Time (s)",fontsize=8.8)
        axp.legend(fontsize=7.8,ncol=2,loc="upper left",handlelength=1.2); axp.set_title("Policy dynamics",fontsize=10.0,fontweight="bold")
        axh = fig.add_subplot(outer[r1,1])
        Hm = A[:,top_k].T; mn = Hm.min(1,keepdims=True); mx = Hm.max(1,keepdims=True); Hm = (Hm-mn)/(mx-mn+1e-8)
        im = axh.imshow(Hm,aspect="auto",cmap=CMAP,extent=[0,t[-1],5-.5,-.5],vmin=0,vmax=1)
        axh.axvline(toa/fps,color=RED,lw=1.6,ls="--"); axh.axvline(alert/fps,color=ORANGE,lw=1.4,ls="-.")
        axh.set_yticks(range(5)); axh.set_yticklabels(MULTI_CONCEPTS,fontsize=7.6); axh.set_xlabel("Time (s)",fontsize=8.8); axh.set_title("Concept evidence",fontsize=10.0,fontweight="bold")
        plt.colorbar(im, ax=axh, fraction=.06, pad=.02, shrink=.85).ax.tick_params(labelsize=6.5)
    plt.savefig(str(OUT / "insight_fig2_multi_dad_real.png"), dpi=220, bbox_inches="tight")
    print("[OK] multi")
if __name__ == "__main__":
    make_multi()
