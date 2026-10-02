"""
assay3_survival_d5.py - Assay 3 (laboratory groups vs original hive) with a common assessment schedule D5, D8, D15, D22.

* Hive: brood combs photographed at D1, D4, D8, D15, D22 (3 colonies x 300 larvae). In assay3_lab_vs_hive.csv the hive D4 photograph is stored in column day_5.
* Laboratory groups = the run-6 plates of Assay 2 (colonies 1-3; 4 plates x 36 larvae per group). In assay3_lab_vs_hive.csv the column day_5 holds the
  laboratory D5 assessment, so the laboratory D4 assessment is taken from Assay 2 (day_4) for the same larvae. D8, D15, D22 are identical in both files.
* Time of death = first assessment (D4, D8, D15, D22) at which the larva is not recorded alive; survivors at D22 are right-censored.
  The 4 larvae without any record (plate B1, 95/80/60 + paper) are counted as dead at the first assessment.
Outputs -> outputs/assay3_*.csv , data/processed/assay3_harmonised_D5.csv
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from pathlib import Path
root=Path(__file__).resolve().parents[1]; out=root/"outputs"
a2=pd.read_csv(root/"data/processed/assay2_lab_paper_transfer.csv"); a3=pd.read_csv(root/"data/processed/assay3_lab_vs_hive.csv")
lab=a3[a3.assay_type_code==1].copy(); k=["plate","colony","larva_id","humidity_regime_code","transferred_code"]
lab["source"]="lab"; lab["hum"]=lab.humidity_regime_code.map({1:"95/80/60",2:"70/70/60"}); lab["paper"]=lab.transferred_code.map({1:"WP",2:"P"})
lab["group"]="A3-"+lab.hum+"-"+lab.paper; lab["pid"]=lab.plate+"_"+lab.group
lab=lab.rename(columns={"day_5":"D5","day_8":"D8","day_15":"D15","day_22":"D22"})
hv=a3[a3.assay_type_code==2].copy().rename(columns={"day_5":"D5","day_8":"D8","day_15":"D15","day_22":"D22"})
hv["source"]="hive"; hv["group"]="A3-HIVE"; hv["pid"]="hive_c"+hv.colony.astype(str); hv["hum"]=None; hv["paper"]=None
D=["D5","D8","D15","D22"]; dayn=[5,8,15,22]
H=pd.concat([lab[["source","group","hum","paper","pid","plate","colony","larva_id"]+D],hv[["source","group","hum","paper","pid","plate","colony","larva_id"]+D]],ignore_index=True)
H.to_csv(root/"data/processed/assay3_harmonised_D5.csv",index=False)
print("larvae",len(H),"| lab",len(lab),"| hive",len(hv))
A=(H[D]==1).values; fd=np.where((~A).any(axis=1),(~A).argmax(axis=1),-1)
H["event"]=(fd>=0).astype(int); H["time"]=np.where(fd>=0,np.array(dayn)[np.clip(fd,0,None)],22)
print("events",H.event.sum(),"censored",(H.event==0).sum())
order=["A3-HIVE","A3-95/80/60-WP","A3-95/80/60-P","A3-70/70/60-WP","A3-70/70/60-P"]

# ---- Table 4 ----
L=H[H.source=="lab"]; rows=[]
for g in order[1:]:
    for d in D:
        pl=L[L.group==g].groupby("pid")[d].apply(lambda s:(s==1).mean()*100)
        rows.append(dict(group=g,colony="ALL (mean of plates)",day=d,mean=pl.mean(),sd=pl.std(ddof=1),n=len(pl)))
        for c in (1,2,3):
            cp=L[(L.group==g)&(L.colony==c)].groupby("pid")[d].apply(lambda s:(s==1).mean()*100)
            rows.append(dict(group=g,colony=f"colony {c}",day=d,mean=cp.mean(),sd=cp.std(ddof=1),n=len(cp)))
for d in D:
    cv=[(H[(H.source=="hive")&(H.colony==c)][d]==1).mean()*100 for c in (1,2,3)]
    for c,v in zip((1,2,3),cv): rows.append(dict(group="A3-HIVE",colony=f"colony {c}",day=d,mean=v,sd=np.nan,n=300))
    rows.append(dict(group="A3-HIVE",colony="ALL (mean of colonies)",day=d,mean=np.mean(cv),sd=np.std(cv,ddof=1),n=3))
T4=pd.DataFrame(rows)
# letters among the four laboratory groups (plate level, n=4 per group): Kruskal-Wallis then pairwise Mann-Whitney (p<=0.05)
let=[]
for d in D:
    pl={g:L[L.group==g].groupby("pid")[d].apply(lambda s:(s==1).mean()*100).values for g in order[1:]}
    kw=stats.kruskal(*pl.values()).pvalue
    ranked=sorted(pl,key=lambda g:pl[g].mean())
    # simple compact letter display from pairwise MW p<=0.05 (greedy)
    sig={(a,b):stats.mannwhitneyu(pl[a],pl[b]).pvalue<=0.05 for a in pl for b in pl if a!=b}
    letters={g:"" for g in pl}; cur=0; groups=[]
    for g in ranked:
        placed=False
        for gi,mem in enumerate(groups):
            if all(not sig[(g,m)] for m in mem): mem.append(g); letters[g]+="abcdef"[gi]; placed=True
        if not placed: groups.append([g]); letters[g]+="abcdef"[len(groups)-1]
    for g in pl: let.append(dict(day=d,group=g,letters="".join(sorted(set(letters[g]))),kruskal_p=kw))
LT=pd.DataFrame(let); T4.round(3).to_csv(out/"assay3_d5_table4_values.csv",index=False); LT.round(4).to_csv(out/"assay3_d5_table4_letters.csv",index=False)
print(T4[T4.colony.str.startswith("ALL")].round(2).to_string(index=False)); print(LT.round(3).to_string(index=False))

# ---- S6 (first-dead, common schedule) ----
s6=[]
for g in order:
    gd=H[H.group==g]; N0=len(gd); at=N0; cum=0
    s6.append(dict(group=g,day=0,at_risk=N0,cum_events=0,cum_censored=0))
    for x in dayn:
        dd=int(((gd.time==x)&(gd.event==1)).sum()); ce=int(((gd.time==x)&(gd.event==0)).sum()); cum+=dd
        s6.append(dict(group=g,day=x,at_risk=at,cum_events=cum,cum_censored=ce if x==22 else 0)); at-=dd+ce
S6=pd.DataFrame(s6); S6.to_csv(out/"assay3_d5_tableS6.csv",index=False); print(S6.to_string(index=False))

# ---- lab-only Cox (Efron, SE clustered by plate), reference A3-95/80/60-WP ----
Lb=H[H.source=="lab"].copy()
for g in order[2:]: Lb[g]=(Lb.group==g).astype(int)
def cox(df,X,label,names):
    m=PHReg(df.time.values,df[X].values.astype(float),status=df.event.values,ties="efron").fit(groups=df.pid.values)
    return [dict(model=label,term=n,coef=m.params[i],se=m.bse[i],z=m.params[i]/m.bse[i],p=2*stats.norm.sf(abs(m.params[i]/m.bse[i])),hr=np.exp(m.params[i]),lo=np.exp(m.params[i]-1.96*m.bse[i]),hi=np.exp(m.params[i]+1.96*m.bse[i]),n=len(df),events=int(df.event.sum())) for i,n in enumerate(names)]
res=cox(Lb,order[2:],"lab groups vs A3-95/80/60-WP (plate-clustered)",order[2:])
Lb["h70"]=(Lb.hum=="70/70/60").astype(int); Lb["pp"]=(Lb.paper=="P").astype(int); Lb["inter"]=Lb.h70*Lb.pp
res+=cox(Lb,["h70","pp","inter"],"lab factorial (plate-clustered)",["humidity 70 (WP)","paper (at 95/80/60)","interaction"])
for c in (2,3):
    Lb[f"c{c}"]=(Lb.colony==c).astype(int)
res+=cox(Lb,["c2","c3","h70","pp","inter"],"lab factorial + colony (colony 1 = reference)",["colony 2","colony 3","humidity 70 (WP)","paper (at 95/80/60)","interaction"])
for nm,(g1,g2) in {"hive vs lab pooled":None}.items() if False else {}.items(): pass
CX=pd.DataFrame(res); CX.round(4).to_csv(out/"assay3_d5_cox_lab.csv",index=False); print(CX.round(3).to_string(index=False))

# ---- descriptive hive vs lab at D22 by colony ----
rows=[]
for c in (1,2,3):
    r=dict(colony=c,hive=(H[(H.source=="hive")&(H.colony==c)]["D22"]==1).mean()*100)
    for g in order[1:]: r[g]=(H[(H.group==g)&(H.colony==c)]["D22"]==1).mean()*100
    rows.append(r)
DC=pd.DataFrame(rows).round(2); DC.to_csv(out/"assay3_d5_D22_by_colony.csv",index=False); print(DC.to_string(index=False))
print("rank order of colonies by D22 survival (hive):",list(DC.sort_values("hive",ascending=False).colony))
for g in order[1:]: print(g,list(DC.sort_values(g,ascending=False).colony))

# ---- Cohen's f (eta^2-based) on colony x plate cells of the 4 laboratory groups ----
f=[]
for d in D:
    c=Lb.assign(h=Lb.hum,p=Lb.paper).groupby(["h","p","colony","plate"])[d].apply(lambda s:(s==1).mean()*100).reset_index(name="y")
    A_=sm.stats.anova_lm(smf.ols("y ~ C(colony)+C(h)*C(p)",data=c).fit(),typ=2); tot=A_.sum_sq.sum()
    for kx in A_.index[:-1]:
        e=A_.loc[kx,"sum_sq"]/tot; f.append(dict(day=d,effect=kx,eta2=e,cohen_f=np.sqrt(e/(1-e))))
F=pd.DataFrame(f); F.round(3).to_csv(out/"assay3_d5_cohen_f.csv",index=False); print(F.round(2).pivot(index="day",columns="effect",values="cohen_f"))
