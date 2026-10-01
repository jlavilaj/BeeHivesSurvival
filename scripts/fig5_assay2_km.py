"""fig5_assay2_km.py - Figure 5 (Assay 2): KM curves of the four groups, first-observed-dead convention, 28 plates;
4 larvae without any record counted as dead at D3. Bands: 95% percentile CI, plate-level bootstrap (2000)."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
root=Path(__file__).resolve().parents[1]
d=pd.read_csv(root/"data/processed/assay2_lab_paper_transfer.csv")
days=[3,4,5,6,7,8,15,22]; cols=[f"day_{x}" for x in days]
d["hum"]=d.humidity_regime_code.map({1:"95/80/60",2:"70/70/60"}); d["P"]=(d.transferred_code==2).astype(int)
d["grp"]=d.hum+np.where(d.P==1," · P"," · WP"); d["pid"]=d.run.astype(str)+d.plate+d.grp
A=(d[cols]==1).values; fd=np.where((~A).any(axis=1),(~A).argmax(axis=1),-1)
d["event"]=(fd>=0).astype(int); d["time"]=np.where(fd>=0,np.array(days)[np.clip(fd,0,None)],22)
grid=[0]+days
def km(g):
    S=1.0;n=len(g);o=[1.0];t=g.time.values;e=g.event.values
    for x in days:
        k=((t==x)&(e==1)).sum(); S*=1-k/n; o.append(S); n-=(t==x).sum()
    return np.array(o)
rng=np.random.default_rng(2024); fig,ax=plt.subplots(figsize=(9,6.5),dpi=200)
for lab,col in (("95/80/60 · WP","#1f77b4"),("95/80/60 · P","#87ceeb"),("70/70/60 · WP","#d62728"),("70/70/60 · P","#ffa07a")):
    g=d[d.grp==lab]; S=km(g); gl=[x for _,x in g.groupby("pid")]; B=[]
    for _ in range(2000):
        i=rng.integers(0,len(gl),len(gl)); B.append(km(pd.concat([gl[j] for j in i])))
    lo,hi=np.percentile(B,[2.5,97.5],axis=0); gx=grid+[23]
    ax.step(gx,np.append(S,S[-1]),where="post",color=col,lw=2,label=lab)
    ax.fill_between(gx,np.append(lo,lo[-1]),np.append(hi,hi[-1]),step="post",color=col,alpha=.15)
    print(lab,np.round(S[-3:],4))
ax.set_xlabel("Days"); ax.set_ylabel("Survival probability"); ax.set_ylim(0,1.03); ax.grid(alpha=.3); ax.legend(loc="lower left",fontsize=9)
fig.tight_layout(); fig.savefig(root/"outputs/figure5_assay2_km.png")
