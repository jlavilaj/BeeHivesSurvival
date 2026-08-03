"""
assay3_pooled_all_colonies.py
--------------------------------
Assay 3: pooled analysis across all three source colonies, comparing
the original hive against the four laboratory treatment combinations
(humidity regime x lab-paper transfer). Reproduces the main "Kaplan
Meier" figure (Figure 6) and the overall Cox model reported in Section
3.3 of the manuscript (concordance index, likelihood-ratio test, and
hazard ratios by treatment).

Input data: data/processed/assay3_lab_vs_hive.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from survival_utils import add_time_and_event_columns

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "assay3_lab_vs_hive.csv"

DAY_COLUMNS = ["day_3", "day_5", "day_8", "day_15", "day_22"]
DAY_TO_TIME = {"day_3": 3, "day_5": 5, "day_8": 8, "day_15": 15, "day_22": 22}

GROUP_ORDER = [
    "Hive",
    "Treatment 1 + Transfer 1",
    "Treatment 1 + Transfer 2",
    "Treatment 2 + Transfer 1",
    "Treatment 2 + Transfer 2",
]
GROUP_COLORS = {
    "Hive": "green",
    "Treatment 1 + Transfer 1": "blue",
    "Treatment 1 + Transfer 2": "skyblue",
    "Treatment 2 + Transfer 1": "red",
    "Treatment 2 + Transfer 2": "orange",
}
# Labels matching the manuscript's Figure 6 legend.
PLOT_LABELS = {
    "Hive": "Hive",
    "Treatment 1 + Transfer 1": "95/80/60 - without Transference",
    "Treatment 1 + Transfer 2": "95/80/60 + Transference with lab paper",
    "Treatment 2 + Transfer 1": "70/70/60 - without Transference",
    "Treatment 2 + Transfer 2": "70/70/60 + Transference with lab paper",
}


def assign_group(row):
    if row["assay_type_code"] == 2:
        return "Hive"
    if row["assay_type_code"] == 1:
        humidity = int(row["humidity_regime_code"]) if not pd.isna(row["humidity_regime_code"]) else None
        transfer = int(row["transferred_code"]) if not pd.isna(row["transferred_code"]) else None
        if humidity in (1, 2) and transfer in (1, 2):
            return f"Treatment {humidity} + Transfer {transfer}"
    return None


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["colony", "assay_type_code"])
    df = add_time_and_event_columns(df, DAY_COLUMNS, DAY_TO_TIME, time_if_dead_before_first_check=3)
    df["group"] = df.apply(assign_group, axis=1)
    df = df[df["group"].notna()]
    return df


def plot_kaplan_meier(df):
    plt.figure(figsize=(12, 7))
    kmf = KaplanMeierFitter()
    for group in GROUP_ORDER:
        subgroup = df[df["group"] == group]
        if len(subgroup) == 0:
            continue
        kmf.fit(subgroup["time"], subgroup["event"], label=PLOT_LABELS[group])
        kmf.plot_survival_function(ci_show=True, color=GROUP_COLORS[group])

    plt.title("Kaplan Meier")
    plt.xlabel("Days")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fit_cox_model(df):
    cox_data = df[["time", "event", "group"]].dropna()
    cox_data["group"] = pd.Categorical(cox_data["group"], categories=GROUP_ORDER)
    if cox_data["event"].sum() == 0:
        print("Cannot fit a Cox model: no events observed.")
        return None

    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="time", event_col="event", formula="C(group)")
    print("\nPooled Cox model results by group (hazard ratios) - Section 3.3 of the manuscript:")
    cph.print_summary()
    return cph


if __name__ == "__main__":
    data = load_data()
    print(f"Total observations: {len(data)}")
    plot_kaplan_meier(data)
    fit_cox_model(data)
