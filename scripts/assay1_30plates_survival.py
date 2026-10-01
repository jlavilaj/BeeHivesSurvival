"""
assay1_30plates_survival.py
Assay 1 survival analysis from the larva-level CSV (30 plates x 36 larvae per regime):
 - time = first visit at which the larva is NOT recorded alive (first-dead / interval-censoring convention);
   larvae alive at D22 are administratively right-censored at D22
 - Table S6 (at risk, deaths first recorded, cumulative events/censored, KM)
 - Cox PH with cluster-robust SE (plate) overall and by window (D0-D8, D8-D22); run-stratified version
 - D22 analysis with the plate as unit
Outputs -> outputs/
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from pathlib import Path
root = Path(__file__).resolve().parents[1]; out = root/"outputs"
df = pd.read_csv(root/"data/processed/assay1_humidity_regimes.csv")
days = [3,4,5,6,7,8,15,22]; cols = [f"day_{d}" for d in days]
df["plate_id"] = df.run.astype(str)+"_"+df.plate
df["reg"] = (df.humidity_regime_code==2).astype(int)
A = (df[cols]==1).values            # alive at visit (blank/0 = not recorded alive)
nonmono = int(((~A[:, :-1]) & A[:, 1:]).any(axis=1).sum())
first_dead = np.where((~A).any(axis=1), (~A).argmax(axis=1), -1)
df["event"] = (first_dead >= 0).astype(int)
df["time"] = np.where(first_dead >= 0, np.array(days)[np.clip(first_dead,0,None)], 22)
print("larvae", len(df), "plates", df.plate_id.nunique(), "non-monotone records", nonmono,
      "| events", df.event.sum(), "censored", (df.event==0).sum())
df[["run","plate","block","humidity_regime_code","colony","larva_id","time","event"]].rename(
   columns={"time":"time_first_dead_or_censor","event":"event_death"}).to_csv(root/"outputs/assay1_30plates_survival_times.csv", index=False)

# ---- Table S6 ----
s6 = []
for code, lab in ((1,"95/80/60"),(2,"70/70/60")):
    g = df[df.humidity_regime_code==code]; N0 = len(g); atrisk = N0; cum = 0; S = 1.0
    s6.append(dict(regime=lab, day=0, at_risk=N0, deaths_first_recorded=0, cum_events=0, cum_censored=0, km_survival=1.0))
    for d in days:
        dd = int(((g.time==d)&(g.event==1)).sum()); cens = int(((g.time==d)&(g.event==0)).sum()); cum += dd; S *= 1-dd/atrisk
        s6.append(dict(regime=lab, day=d, at_risk=atrisk, deaths_first_recorded=dd, cum_events=cum, cum_censored=cens if d==22 else 0, km_survival=S))
        atrisk -= dd + cens
S6 = pd.DataFrame(s6); S6.round(4).to_csv(out/"tableS6_assay1_30plates.csv", index=False); print(S6.round(4).to_string(index=False))

# ---- Cox ----
def cox(d, label, strata=None):
    m = PHReg(d.time.values, d[["reg"]].values, status=d.event.values, ties="efron", strata=strata).fit(groups=d.plate_id.values)
    b, se = m.params[0], m.bse[0]
    return dict(model=label, coef=b, se=se, z=b/se, p=2*stats.norm.sf(abs(b/se)), hr=np.exp(b), lo=np.exp(b-1.96*se), hi=np.exp(b+1.96*se), n=len(d), events=int(d.event.sum()))
res = [cox(df,"Overall D3-D22 (cluster-robust, plate)")]
res.append(cox(df,"Overall, stratified by run", strata=df.run.values))
w1 = df.copy(); w1.loc[w1.time>8, ["time","event"]] = [8,0]; res.append(cox(w1,"Window D0-D8"))
w2 = df[df.time>8].copy(); w2["time"] -= 8; res.append(cox(w2,"Window D8-D22 (alive at D8)"))
R = pd.DataFrame(res); R.round(4).to_csv(out/"cox_assay1_30plates.csv", index=False); print(R.round(3).to_string(index=False))
# naive (larva-independent) SE for comparison
m = PHReg(df.time.values, df[["reg"]].values, status=df.event.values, ties="efron").fit()
print("naive HR %.2f (%.2f-%.2f)" % (np.exp(m.params[0]), *np.exp(m.params[0]+np.array([-1.96,1.96])*m.bse[0])))

# ---- D22 at plate level ----
P = df.groupby(["plate_id","reg","run"]).agg(n=("larva_id","size"), alive=("day_22", lambda s:(s==1).sum())).reset_index()
P["pct"] = P.alive/P.n*100
x = P[P.reg==1].pct; y = P[P.reg==0].pct
print("D22 diff %.2f pts; Welch CI %s; MW p=%.2g" % (x.mean()-y.mean(), np.round(stats.ttest_ind(x,y,equal_var=False).confidence_interval(.95),2), stats.mannwhitneyu(x,y).pvalue))
g = smf.glm("alive + I(n-alive) ~ reg", data=P, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups":P.plate_id.factorize()[0]})
print("plate-level binomial GLM: OR %.2f (%.2f-%.2f)" % (np.exp(g.params["reg"]), *np.exp(g.conf_int().loc["reg"])))
g2 = smf.glm("alive + I(n-alive) ~ reg + C(run)", data=P, family=sm.families.Binomial()).fit(cov_type="HC1")
print("  + run as fixed effect: OR %.2f (%.2f-%.2f)" % (np.exp(g2.params["reg"]), *np.exp(g2.conf_int().loc["reg"])))
print("plates per run x regime:\n", P.groupby(["run","reg"]).size().unstack())
