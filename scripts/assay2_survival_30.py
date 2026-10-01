"""
assay2_survival.py - Assay 2 (humidity x transfer-plus-paper), larva-level CSV, 7 plates (36 larvae) per group.
Unit = plate (run + plate label + group). Outputs to outputs/:
  table3_assay2_plates.csv       plate-level summary (mean, SD, bounded t-CI, min, max, Mann-Whitney vs paper / vs humidity)
  tableS6_assay2.csv             at risk / cumulative events / cumulative censored, first-dead convention
  cox_assay2.csv                 HR for transfer within each humidity regime (Efron, SE clustered by plate), windows, interaction
  d22_assay2_plate_models.csv    plate-level binomial GLMs at D22 (factorial, with/without run)
Blank larvae: 4 larvae (run 6, plate B1, 95/80/60 + paper, larvae 10-13) have NO record at any visit.
Primary analysis counts them as dead at D3 (decision of the authors; consistent with Table S5, 248/252 alive at D3); sensitivity analysis excludes them.
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from pathlib import Path
root = Path(__file__).resolve().parents[1]; out = root/"outputs"
raw = pd.read_csv(root/"data/processed/assay2_lab_paper_transfer.csv")
days=[3,4,5,6,7,8,15,22]; cols=[f"day_{d}" for d in days]
raw["hum"]=raw.humidity_regime_code.map({1:"95/80/60",2:"70/70/60"}); raw["paper"]=(raw.transferred_code==2).astype(int)
raw["grp"]=raw.hum+raw.paper.map({0:" WP",1:" P"})
raw["pid"]=raw.run.astype(str)+"_"+raw.plate+"_"+raw.grp
blank = raw[cols].isna().all(axis=1)
print("blank larvae (no record at any visit):", int(blank.sum()), "| plates:", raw.pid.nunique(), "| per group:", raw.groupby("grp").pid.nunique().to_dict())

def build(df):
    A=(df[cols]==1).values; fd=np.where((~A).any(axis=1),(~A).argmax(axis=1),-1)
    df=df.copy(); df["event"]=(fd>=0).astype(int); df["time"]=np.where(fd>=0,np.array(days)[np.clip(fd,0,None)],22); return df
prim = build(raw); sens = build(raw[~blank])

# ---- Table 3 (plate-level) ----
def plate_pct(df, d):
    return df.groupby(["grp","pid"])[f"day_{d}"].apply(lambda s:(s==1).sum()/len(s)*100).reset_index(name="pct")
rows=[]
for label,df in (("blank_dead",raw),("exclude_blank",raw[~blank])):
    for d in days:
        P=plate_pct(df,d); g={k:v.pct.values for k,v in P.groupby("grp")}
        for hum in ("95/80/60","70/70/60"):
            a,b=g[hum+" WP"],g[hum+" P"]; p_tr=stats.mannwhitneyu(a,b).pvalue
            for tr,x in (("WP",a),("P",b)):
                other=g[("70/70/60" if hum=="95/80/60" else "95/80/60")+" "+tr]
                m,s=x.mean(),x.std(ddof=1); h=stats.t.ppf(.975,len(x)-1)*s/np.sqrt(len(x))
                rows.append(dict(handling_blank=label,day=d,humidity=hum,transfer=tr,n_plates=len(x),mean=m,sd=s,ci_low=max(m-h,0),ci_high=min(m+h,100),
                  min=x.min(),max=x.max(),p_transfer_within_regime=p_tr,p_humidity_within_transfer=stats.mannwhitneyu(x,other).pvalue))
T3=pd.DataFrame(rows); T3.round(4).to_csv(out/"table3_assay2_plates.csv",index=False)
print(T3[(T3.handling_blank=="blank_dead")&T3.day.isin([3,15,22])].round(2).to_string(index=False))

# ---- S6 ----
s6=[]
for g,gd in prim.groupby("grp"):
    N0=len(gd); at=N0; cum=0
    s6.append(dict(group=g,day=0,at_risk=N0,cum_events=0,cum_censored=0))
    for d in days:
        dd=int(((gd.time==d)&(gd.event==1)).sum()); ce=int(((gd.time==d)&(gd.event==0)).sum()); cum+=dd
        s6.append(dict(group=g,day=d,at_risk=at,cum_events=cum,cum_censored=ce if d==22 else 0)); at-=dd+ce
pd.DataFrame(s6).to_csv(out/"tableS6_assay2.csv",index=False)

# ---- Cox (Efron, SE clustered by plate) ----
def cox(df,X,label,names):
    m=PHReg(df.time.values,df[X].values,status=df.event.values,ties="efron").fit(groups=df.pid.values)
    r=[]
    for i,nm in enumerate(names):
        b,se=m.params[i],m.bse[i]; r.append(dict(model=label,term=nm,coef=b,se=se,z=b/se,p=2*stats.norm.sf(abs(b/se)),hr=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),n=len(df),events=int(df.event.sum())))
    return r
res=[]
for tag,df in (("blank_dead",prim),("exclude_blank",sens)):
    for hum in ("95/80/60","70/70/60"):
        g=df[df.hum==hum]
        res+=cox(g,["paper"],f"[{tag}] {hum}: P vs WP, overall",["paper"])
        w1=g.copy(); w1.loc[w1.time>8,["time","event"]]=[8,0]; res+=cox(w1,["paper"],f"[{tag}] {hum}: P vs WP, D0-D8",["paper"])
        w2=g[g.time>8].copy(); w2["time"]-=8; res+=cox(w2,["paper"],f"[{tag}] {hum}: P vs WP, D8-D22",["paper"])
    df=df.copy(); df["h70"]=(df.hum=="70/70/60").astype(int); df["inter"]=df.h70*df.paper
    res+=cox(df,["h70","paper","inter"],f"[{tag}] factorial overall",["humidity 70 vs 95 (WP)","paper (at 95/80/60)","interaction humidity x paper"])
CX=pd.DataFrame(res); CX.round(4).to_csv(out/"cox_assay2.csv",index=False); print(CX[CX.model.str.contains("blank_dead")].round(3).to_string(index=False))

# ---- D22 plate-level GLMs ----
P=prim.groupby(["pid","run","hum","paper"]).agg(n=("larva_id","size"),alive=("day_22",lambda s:(s==1).sum())).reset_index(); P["h70"]=(P.hum=="70/70/60").astype(int)
mods=[]
for name,f in (("factorial",'alive + I(n-alive) ~ h70*paper'),("factorial + run",'alive + I(n-alive) ~ h70*paper + C(run)')):
    g=smf.glm(f,data=P,family=sm.families.Binomial()).fit(cov_type="cluster",cov_kwds={"groups":P.pid.factorize()[0]}) if False else smf.glm(f,data=P,family=sm.families.Binomial()).fit(cov_type="HC1")
    ci=g.conf_int()
    for t in g.params.index:
        if t=="Intercept" or t.startswith("C("): continue
        mods.append(dict(model=name,term=t,logOR=g.params[t],se=g.bse[t],OR=np.exp(g.params[t]),lo=np.exp(ci.loc[t,0]),hi=np.exp(ci.loc[t,1]),p=g.pvalues[t]))
M=pd.DataFrame(mods); M.round(4).to_csv(out/"d22_assay2_plate_models.csv",index=False); print(M.round(3).to_string(index=False))
# plate-level two-way ANOVA on percentages
import statsmodels.stats.anova as sa
P["pct"]=P.alive/P.n*100
print(sa.anova_lm(smf.ols("pct ~ h70*paper",data=P).fit(),typ=2).round(4))
print(sa.anova_lm(smf.ols("pct ~ h70*paper + C(run)",data=P).fit(),typ=2).round(4))
print("D22 plate means:\n",P.groupby(["hum","paper"]).pct.agg(["mean","std","count"]).round(2))
