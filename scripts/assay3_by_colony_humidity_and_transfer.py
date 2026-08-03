"""
assay3_by_colony_humidity_and_transfer.py
--------------------------------------------
Assay 3: full breakdown of brood survival by source colony, humidity
regime (1 = 95/80/60, 2 = 70/70/60), and lab-paper transfer protocol
(1 = without transfer, 2 = transferred), compared against the original
hive. This is the analysis underlying the per-colony hazard ratios
reported in Table 4 of the manuscript.

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


def analyse_colony(df, colony):
    print(f"\n{'=' * 20}\nSOURCE COLONY: {colony}\n{'=' * 20}")
    colony_data = df[df["colony"] == colony]
    groups_present = colony_data["group"].unique()
    print(f"Groups present: {groups_present}")
    if len(groups_present) < 2:
        print("Not enough groups to compare for this colony, skipping.")
        return

    # --- Kaplan-Meier ---
    plt.figure(figsize=(12, 7))
    kmf = KaplanMeierFitter()
    for group in GROUP_ORDER:
        if group in groups_present:
            subgroup = colony_data[colony_data["group"] == group]
            kmf.fit(subgroup["time"], subgroup["event"], label=group)
            kmf.plot_survival_function(ci_show=True, color=GROUP_COLORS[group])

    plt.title(f"Survival curves - Colony {colony}")
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Cox model ---
    cox_data = colony_data[["time", "event", "group"]].dropna()
    cox_data["group"] = pd.Categorical(cox_data["group"], categories=GROUP_ORDER)
    if cox_data["event"].sum() == 0:
        print("Cannot fit a Cox model: no events observed.")
        return

    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="time", event_col="event", formula="C(group)")
    print("\nCox model results by group (hazard ratios) - feeds into manuscript Table 4:")
    cph.print_summary()


if __name__ == "__main__":
    data = load_data()
    for colony in sorted(data["colony"].unique()):
        analyse_colony(data, colony)
