"""
make_table2_assay1.py
Regenerates Table 2 (Assay 1) from the larva-level CSV, using the plate
(run + plate label, 36 larvae) as the unit: 30 plates per humidity regime.

Survival at visit d = share of the 36 larvae of a plate recorded alive (value 1) at d.
Summary = mean +/- s.d., 95% t-interval, min and max across the 30 plates.
Between-regime comparison per day: Mann-Whitney U on plate percentages (30 vs 30).
"""
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data/processed/assay1_humidity_regimes.csv")
days = ["day_3","day_4","day_5","day_6","day_7","day_8","day_15","day_22"]
plates = (df.assign(plate_id=df.run.astype(str) + "_" + df.plate)
            .groupby(["humidity_regime_code","plate_id"])[days]
            .apply(lambda d: (d == 1).mean() * 100))
rows = []
for d in days:
    a = plates.loc[1][d].values; b = plates.loc[2][d].values
    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    for code, x, lab in ((1, a, "95/80/60"), (2, b, "70/70/60")):
        m, s = x.mean(), x.std(ddof=1)
        h = stats.t.ppf(0.975, len(x)-1) * s / np.sqrt(len(x))
        rows.append(dict(day=int(d.split("_")[1]), regime=lab, n_plates=len(x),
                         mean=m, sd=s, ci_low=m-h, ci_high=min(m+h, 100.0),
                         min=x.min(), max=x.max(), mw_p=p))
t = pd.DataFrame(rows)
t["letter"] = np.where(t.mw_p < 0.05, np.where(t.regime == "95/80/60", "a", "b"), "a")
t.round(2).to_csv(root / "outputs/table2_assay1_30plates.csv", index=False)
print(t.round(2).to_string(index=False))
