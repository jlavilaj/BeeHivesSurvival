"""
assay3_by_colony.py
---------------------
Assay 3: compares brood survival under laboratory conditions against
survival recorded in the original hives, analysed separately for each
source colony (Colony 1, 2 and 3).

This is the coarsest breakdown of Assay 3: it only distinguishes
"Laboratory" vs "Hive" (it does not yet split the laboratory group by
humidity regime or transfer protocol - see assay3_by_colony_and_humidity.py
and assay3_by_colony_humidity_and_transfer.py for that).

Input data: data/processed/assay3_lab_vs_hive.csv
(one row per larva; day_3, day_5, day_8, day_15, day_22 = 1 while alive,
blank after death)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from survival_utils import add_time_and_event_columns

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "assay3_lab_vs_hive.csv"

DAY_COLUMNS = ["day_3", "day_5", "day_8", "day_15", "day_22"]
DAY_TO_TIME = {"day_3": 3, "day_5": 5, "day_8": 8, "day_15": 15, "day_22": 22}

# assay_type_code: 1 = reared in the laboratory, 2 = original hive.
ASSAY_TYPE_LABELS = {1: "Laboratory", 2: "Hive"}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["colony", "assay_type_code"])
    # A larva that died before the first check (D3) is treated as dead at
    # day 3, matching the convention used throughout this repository.
    df = add_time_and_event_columns(df, DAY_COLUMNS, DAY_TO_TIME, time_if_dead_before_first_check=3)
    return df


def analyse_colony(df, colony):
    print(f"\n{'=' * 20}\nSOURCE COLONY: {colony}\n{'=' * 20}")
    colony_data = df[df["colony"] == colony]

    groups_present = colony_data["assay_type_code"].unique()
    print(f"Groups present: {[ASSAY_TYPE_LABELS[g] for g in groups_present]}")
    if len(groups_present) < 2:
        print("Both groups (Laboratory / Hive) are not present for this colony, skipping.")
        return

    # --- Kaplan-Meier ---
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for group_code in sorted(groups_present):
        label = ASSAY_TYPE_LABELS[group_code]
        subgroup = colony_data[colony_data["assay_type_code"] == group_code]
        kmf.fit(subgroup["time"], subgroup["event"], label=label)
        kmf.plot_survival_function(ci_show=True)

    plt.title(f"Survival curve - Colony {colony}")
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Cox model ---
    cox_data = colony_data[["time", "event", "assay_type_code"]].dropna()
    events_per_group = cox_data.groupby("assay_type_code")["event"].sum()
    if (events_per_group == 0).any():
        print("Cannot fit a Cox model: no events observed in at least one group.")
        return

    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="time", event_col="event")
    print("\nCox model results for Laboratory vs Hive (hazard ratio):")
    cph.print_summary()


if __name__ == "__main__":
    data = load_data()
    for colony in sorted(data["colony"].unique()):
        analyse_colony(data, colony)
