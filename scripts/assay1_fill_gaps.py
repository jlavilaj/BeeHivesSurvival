"""
assay1_fill_gaps.py
Assay 1 (30 plates per regime): larvae with an isolated missing record between two
'alive' records are treated as alive at the missing visit (a larva recorded alive at a
later visit cannot have been dead earlier; the blank is a missing observation).

Rule: for visit j, if the larva is not recorded alive at j but is recorded alive at some
earlier visit AND at some later visit, set day_j = 1.
Affects 5 larvae of 2,160 (D6: 1, D8: 1, D15: 3). Raw file is left untouched; the
gap-filled copy is written next to it and used by all Assay 1 scripts.
"""
import numpy as np, pandas as pd
from pathlib import Path
root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root/"data/processed/assay1_humidity_regimes.csv")
days = [3,4,5,6,7,8,15,22]; cols = [f"day_{d}" for d in days]
A = (df[cols] == 1).values
before = np.maximum.accumulate(A, axis=1)                 # alive at this or an earlier visit
after = np.maximum.accumulate(A[:, ::-1], axis=1)[:, ::-1]  # alive at this or a later visit
fill = (~A) & before & after
print("cells filled:", int(fill.sum()), "| larvae:", int(fill.any(axis=1).sum()))
print(df.loc[fill.any(axis=1), ["run","plate","humidity_regime_code","larva_id"]].assign(
      visits=[",".join(str(d) for d, f in zip(days, row) if f) for row in fill[fill.any(axis=1)]]).to_string(index=False))
out = df.copy()
for j, c in enumerate(cols):
    out.loc[fill[:, j], c] = 1.0
out.to_csv(root/"data/processed/assay1_humidity_regimes_gapfilled.csv", index=False)
