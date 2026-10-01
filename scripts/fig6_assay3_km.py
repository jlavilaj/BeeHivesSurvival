"""fig6_assay3_km.py - Figure 6 (Assay 3): KM curves, common schedule D4/D8/D15/D22, first-observed-dead convention.
Laboratory groups: 95% plate-level bootstrap bands (2000). Hive: pooled curve (thick) + the three colony curves (thin dashed); no band
because the hive reference has only three independent colonies."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
root=Path(__file__).resolve().parents[1]
H=pd.read_csv(root/"data/processed/assay3_harmonised_D4.csv"); D=["D4","D8","D15","D22"]; dayn=[4,8,15,22]
A=(H[D]==1).values; fd=np.where((~A).any(axis=1),(~A).argmax(axis=1),-1)
H["event"]=(fd>=0).astype(int); H["time"]=np.where(fd>=0,np.array(dayn)[np.clip(fd,0,None)],22)
grid=[0]+dayn
def km(g):
    S=1.0;n=len(g);o=[1.0];t=g.time.values;e=g.event.values
    for x in dayn:
        k=((t==x)&(e==1)).sum(); S*=1-k/n; o.append(S); n-=(t==x).sum()
    return np.array(o)
gx=grid+[23]; ext=lambda a:np.append(a,a[-1])
rng=np.random.default_rng(2024); fig,ax=plt.subplots(figsize=(9,6.5),dpi=200)
hv=H[H.group=="A3-HIVE"]
ax.step(gx,ext(km(hv)),where="post",color="green",lw=2.4,label="Hive (3 colonies, pooled)")
for c in (1,2,3): ax.step(gx,ext(km(hv[hv.colony==c])),where="post",color="green",lw=.9,ls="--",alpha=.7,label="Hive, individual colonies" if c==1 else None)
for lab,col in (("A3-95/80/60-WP","#1f77b4"),("A3-95/80/60-P","#87ceeb"),("A3-70/70/60-WP","#d62728"),("A3-70/70/60-P","#ffa07a")):
    g=H[H.group==lab]; S=km(g); gl=[x for _,x in g.groupby("pid")]; B=[]
    for _ in range(2000):
        i=rng.integers(0,len(gl),len(gl)); B.append(km(pd.concat([gl[j] for j in i])))
    lo,hi=np.percentile(B,[2.5,97.5],axis=0)
    ax.step(gx,ext(S),where="post",color=col,lw=2,label=lab.replace("A3-","").replace("-"," · ",1) if False else lab[3:].replace("-"," · "))
    ax.fill_between(gx,ext(lo),ext(hi),step="post",color=col,alpha=.15)
ax.set_xlabel("Days"); ax.set_ylabel("Survival probability"); ax.set_ylim(0,1.03); ax.grid(alpha=.3); ax.legend(loc="lower left",fontsize=8.5)
fig.tight_layout(); fig.savefig(root/"outputs/figure6_assay3_km.png")
