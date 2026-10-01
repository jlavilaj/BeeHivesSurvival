"""figS1_assay2_km_with_risktable.py - Figure S1: Figure 5 curves (first-dead convention, plate bootstrap bands)
plus number at risk / cumulative events below (from tableS6_assay2.csv)."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
root=Path(__file__).resolve().parents[1]
d=pd.read_csv(root/"data/processed/assay2_lab_paper_transfer.csv"); S6=pd.read_csv(root/"outputs/tableS6_assay2.csv")
days=[3,4,5,6,7,8,15,22]; cols=[f"day_{x}" for x in days]
d["hum"]=d.humidity_regime_code.map({1:"95/80/60",2:"70/70/60"}); d["P"]=(d.transferred_code==2).astype(int)
d["grp"]=d.hum+np.where(d.P==1," P"," WP"); d["pid"]=d.run.astype(str)+d.plate+d.grp
A=(d[cols]==1).values; fd=np.where((~A).any(axis=1),(~A).argmax(axis=1),-1)
d["event"]=(fd>=0).astype(int); d["time"]=np.where(fd>=0,np.array(days)[np.clip(fd,0,None)],22)
grid=[0]+days
def km(g):
    S=1.0;n=len(g);o=[1.0];t=g.time.values;e=g.event.values
    for x in days:
        k=((t==x)&(e==1)).sum(); S*=1-k/n; o.append(S); n-=(t==x).sum()
    return np.array(o)
rng=np.random.default_rng(2024); fig=plt.figure(figsize=(9.5,8),dpi=170)
ax=fig.add_axes([.09,.46,.88,.50]); tab=fig.add_axes([.09,.03,.88,.34]); tab.axis("off")
G=[("95/80/60 WP","#1f77b4"),("95/80/60 P","#87ceeb"),("70/70/60 WP","#d62728"),("70/70/60 P","#ff7f0e")]
for lab,col in G:
    g=d[d.grp==lab]; S=km(g); gl=[x for _,x in g.groupby("pid")]; B=[]
    for _ in range(1000):
        i=rng.integers(0,len(gl),len(gl)); B.append(km(pd.concat([gl[j] for j in i])))
    lo,hi=np.percentile(B,[2.5,97.5],axis=0); gx=grid+[23]
    ax.step(gx,np.append(S,S[-1]),where="post",color=col,lw=2,label=lab)
    ax.fill_between(gx,np.append(lo,lo[-1]),np.append(hi,hi[-1]),step="post",color=col,alpha=.15)
ax.set_title("Kaplan–Meier survival curves, four experimental groups (Assay 2)"); ax.set_xlabel("Days"); ax.set_ylabel("Survival probability")
ax.set_ylim(0,1.03); ax.set_xlim(-.5,23.5); ax.grid(alpha=.3); ax.legend(title="Group (95% bootstrap CI shaded)",loc="lower left",fontsize=9)
show=[0,4,8,15,22]; y=1.0
tab.set_xlim(-.5,23.5); tab.set_ylim(0,1)
for k,(lab,col) in enumerate(G):
    s=S6[S6.group==lab].set_index("day"); y0=1-k*.25
    tab.text(-.5,y0,lab,color=col,fontsize=9,fontweight="bold",va="top")
    for x in show:
        tab.text(x,y0-.07,f"{int(s.loc[x,'at_risk'])}",ha="center",fontsize=9,va="top")
        tab.text(x,y0-.14,f"{int(s.loc[x,'cum_events'])}",ha="center",fontsize=9,va="top")
    tab.text(23.6,y0-.07,"at risk",fontsize=8,va="top",ha="right",alpha=.0)
tab.text(-.5,0.0,"Rows per group: number at risk (upper), cumulative deaths (lower).",fontsize=8,va="bottom")
fig.savefig(root/"outputs/figureS1_assay2_km.png")
