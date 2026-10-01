"""
fig4_assay1_km.py - Figure 4 (Assay 1): Kaplan-Meier curves, first-observed-dead convention,
30 plates/regime. Shaded band = 95% percentile CI from a plate-level (cluster) bootstrap (2000 resamples).
Output: outputs/figure4_assay1_km_30plates.png
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root/"outputs/assay1_30plates_survival_times.csv")
df["plate_id"] = df.run.astype(str)+"_"+df.plate
days = [3,4,5,6,7,8,15,22]; grid = [0]+days
def km(g):
    S=1.0; n=len(g); out=[1.0]
    t=g.time_first_dead_or_censor.values; e=g.event_death.values
    for d in days:
        dd=((t==d)&(e==1)).sum(); S*=1-dd/n if n else 1; out.append(S); n-=((t==d)).sum()
    return np.array(out)
rng=np.random.default_rng(2024); fig,ax=plt.subplots(figsize=(8,6),dpi=200)
for code,lab,col in ((1,"Humidity 95/80/60","#1f77b4"),(2,"Humidity 70/70/60","#d62728")):
    g=df[df.humidity_regime_code==code]; S=km(g)
    groups=[x for _,x in g.groupby("plate_id")]; B=[]
    for _ in range(2000):
        idx=rng.integers(0,len(groups),len(groups)); B.append(km(pd.concat([groups[i] for i in idx])))
    lo,hi=np.percentile(B,[2.5,97.5],axis=0)
    gx=grid+[23]; S2=np.append(S,S[-1]); lo2=np.append(lo,lo[-1]); hi2=np.append(hi,hi[-1])  # last level drawn to D23 only for readability
    ax.step(gx,S2,where="post",color=col,lw=2,label=lab); ax.fill_between(gx,lo2,hi2,step="post",color=col,alpha=.15)
    print(lab, np.round(S,4), np.round(lo[-1],3), np.round(hi[-1],3))
ax.set_xlabel("Days"); ax.set_ylabel("Survival probability"); ax.set_ylim(0,1.03); ax.grid(alpha=.3); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(root/"outputs/figure4_assay1_km_30plates.png")
