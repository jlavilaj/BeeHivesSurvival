"""
assay1_from_original_workbook.py
Assay 1 analysis from the original SPSS workbook (sheet 'ARCHIVO SPSS'):
10 plates (36 larvae) per humidity regime, plate-level counts of larvae alive at each visit.

Outputs (to outputs/): plate counts CSV, Table 2, Table S6 (first-dead convention), survival analyses.
Usage: python scripts/assay1_from_original_workbook.py <path to RESULTADOS_..._ENSAYO_1.xlsx>
"""
import sys, numpy as np, pandas as pd, openpyxl
from scipy import stats
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
from pathlib import Path

root = Path(__file__).resolve().parents[1]; out = root / "outputs"; out.mkdir(exist_ok=True)
ws = openpyxl.load_workbook(sys.argv[1], data_only=True).worksheets[0]
rows = [r for r in ws.iter_rows(values_only=True)][2:]
rows = [r for r in rows if any(v is not None for v in r)]
days = [3,4,5,6,7,8,15,22]
cols = [11,13,15,17,19,21,23,25]
recs = []; raw = []
for i, r in enumerate(rows, 1):
    reg = "95/80/60" if r[7] in (1,3) else "70/70/60"   # TREATMENT 1 and 3 = 95/80/60 (verified against TRATAMIENTO label)
    assert (("95" in r[6]) == (reg == "95/80/60"))
    recs.append(dict(plate=i, regime=reg, n0=r[10], **{f"d{d}": r[c] for d, c in zip(days, cols)}))
    # raw record exactly as in the workbook sheet 'ARCHIVO SPSS' (counts only; the PER_SUP_* percentage columns are derived)
    raw.append(dict(plate_row=i, datos_utiles=r[2], trabajo_entre_humedades=r[3], run=r[4], repeticion=r[5],
                    tratamiento=r[6], treatment_code=r[7], papel=r[8], papel_numero=r[9], humidity_regime=reg,
                    larvas_iniciales=r[10], **{f"alive_d{d}": r[c] for d, c in zip(days, cols)}))
P = pd.DataFrame(recs); P.to_csv(out/"assay1_original_plate_counts.csv", index=False)
proc = root / "data" / "processed"; proc.mkdir(parents=True, exist_ok=True)
pd.DataFrame(raw).to_csv(proc/"assay1_original10_plate_counts_raw.csv", index=False)
# monotonic check
cnt = P[[f"d{d}" for d in days]].values
assert (np.diff(cnt, axis=1) <= 0).all(), "non-monotone counts"
print("plates per regime:", P.groupby("regime").size().to_dict(), "larvae:", P.groupby("regime").n0.sum().to_dict())

# ---- Table 2 (plate-level, as in the paper) ----
t2 = []
for d in days:
    a = (P[P.regime=="95/80/60"][f"d{d}"]/36*100).values; b = (P[P.regime=="70/70/60"][f"d{d}"]/36*100).values
    p = stats.mannwhitneyu(a, b).pvalue
    for lab, x in (("95/80/60", a), ("70/70/60", b)):
        m, s = x.mean(), x.std(ddof=1); h = stats.t.ppf(.975, len(x)-1)*s/np.sqrt(len(x))
        t2.append(dict(day=d, regime=lab, n_plates=len(x), mean=m, sd=s, ci_low=m-h, ci_high=min(m+h,100), min=x.min(), max=x.max(), mw_p=p))
T2 = pd.DataFrame(t2).round(2); T2.to_csv(out/"table2_assay1_original10.csv", index=False)

# ---- Table S6: first-dead convention ----
s6 = []
for lab in ("95/80/60", "70/70/60"):
    g = P[P.regime==lab]; N0 = int(g.n0.sum()); prev = N0
    s6.append(dict(regime=lab, day=0, at_risk=N0, deaths_first_recorded=0, cum_events=0, alive_at_visit=N0, cum_censored=0, km_survival=1.0))
    for d in days:
        alive = int(g[f"d{d}"].sum())
        s6.append(dict(regime=lab, day=d, at_risk=prev, deaths_first_recorded=prev-alive, cum_events=N0-alive,
                       alive_at_visit=alive, cum_censored=alive if d == 22 else 0, km_survival=alive/N0))
        prev = alive
S6 = pd.DataFrame(s6); S6.to_csv(out/"tableS6_assay1_original10.csv", index=False)
print(S6.round(4).to_string(index=False))

# ---- reconstruct larva-level (time = first visit with death recorded; survivors censored at 22) ----
L = []
for _, r in P.iterrows():
    prev = int(r.n0)
    for d in days:
        a = int(r[f"d{d}"]); L += [dict(plate=r.plate, regime=r.regime, time=d, event=1)]*(prev-a); prev = a
    L += [dict(plate=r.plate, regime=r.regime, time=22, event=0)]*prev
L = pd.DataFrame(L)
L.insert(0, "larva_id", range(1, len(L)+1))   # reconstructed from plate counts: colony/individual identity is NOT in the workbook
L.rename(columns={"time": "time_first_dead_or_censor", "event": "event_death"}).to_csv(proc/"assay1_original10_larva_level_reconstructed.csv", index=False)
L["reg"] = (L.regime=="70/70/60").astype(int)
print("larvae:", len(L), "events:", L.event.sum(), "censored:", (L.event==0).sum())
def cox(df, label):
    m = PHReg(df.time.values, df[["reg"]].values, status=df.event.values, ties="breslow").fit(groups=df.plate.values)
    b, se = m.params[0], m.bse[0]
    print(f"{label}: HR={np.exp(b):.2f} (95% CI {np.exp(b-1.96*se):.2f}-{np.exp(b+1.96*se):.2f}), cluster-robust SE={se:.3f}, n={len(df)}, events={int(df.event.sum())}")
    return dict(model=label, coef=b, se=se, z=b/se, hr=np.exp(b), lo=np.exp(b-1.96*se), hi=np.exp(b+1.96*se), n=len(df), events=int(df.event.sum()))
res = [cox(L, "Overall D3-D22")]
w1 = L.copy(); w1.loc[w1.time > 8, ["time","event"]] = [8, 0]; res.append(cox(w1, "Window D0-D8"))
surv8 = L[L.time > 8].copy(); surv8["time"] -= 8; res.append(cox(surv8, "Window D8-D22 (alive at D8)"))
pd.DataFrame(res).round(4).to_csv(out/"cox_assay1_original10.csv", index=False)

# ---- D22 at plate level ----
L22 = P.assign(alive=P.d22, dead=P.n0-P.d22, reg=(P.regime=="70/70/60").astype(int))
q = smf.glm("alive + dead ~ reg", data=L22, family=sm.families.Binomial()).fit(cov_type="HC1")
print("D22 plate-level binomial GLM (robust): OR=%.2f (%.2f-%.2f)" % (np.exp(q.params["reg"]), *np.exp(q.conf_int().loc["reg"])))
x = P[P.regime=="70/70/60"].d22/36*100; y = P[P.regime=="95/80/60"].d22/36*100
print("D22 difference (percentage points): %.2f; Welch 95%% CI %s; Mann-Whitney p=%.2g" %
      (x.mean()-y.mean(), np.round(stats.ttest_ind(x, y, equal_var=False).confidence_interval(0.95), 2), stats.mannwhitneyu(x, y).pvalue))
