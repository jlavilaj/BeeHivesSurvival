"""
assay2_lab_paper_transfer_kaplan_meier_cox.py
-----------------------------------------------
Assay 2: effect of transferring larvae to new cells lined with absorbent
lab paper (from D9 onwards), evaluated separately within each humidity
regime block (Block 1 = 95/80/60, Block 2 = 70/70/60).

Reproduces:
- The Kaplan-Meier survival curves per block, comparing "transferred" vs
  "not transferred" brood (Figures 5A and 5B in the manuscript).
- The Cox proportional-hazards model (hazard ratio for the transfer
  protocol) fitted independently within each humidity block.

Input data: data/processed/assay2_lab_paper_transfer.csv
(one row per larva; day_3 ... day_22 = 1 while alive, blank after death)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from survival_utils import add_time_and_event_columns

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "assay2_lab_paper_transfer.csv"

DAY_COLUMNS = ["day_3", "day_4", "day_5", "day_6", "day_7", "day_8", "day_15", "day_22"]
DAY_TO_TIME = {"day_3": 3, "day_4": 4, "day_5": 5, "day_6": 6, "day_7": 7, "day_8": 8, "day_15": 15, "day_22": 22}

# Figure/block labels used in the manuscript.
BLOCK_TITLES = {
    "BLOQUE 1": "A. Kaplan-Meier Humidity 95/80/60",
    "BLOQUE 2": "B. Kaplan-Meier Humidity 70/70/60",
}
TRANSFER_LABELS = {
    1: "No Transferred to new cells",
    2: "Transferred to new cells with lab paper",
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = add_time_and_event_columns(df, DAY_COLUMNS, DAY_TO_TIME)
    return df


def analyse_block(df, block_name):
    print(f"\n{'=' * 40}\nBlock: {block_name}\n{'=' * 40}")
    block_data = df[df["block"] == block_name]

    # --- Kaplan-Meier curves: transferred vs not transferred ---
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for transferred_code, group in block_data.groupby("transferred_code"):
        label = TRANSFER_LABELS[transferred_code]
        kmf.fit(group["time"], group["event"], label=label)
        kmf.plot_survival_function(ci_show=True)

    plt.title(f" {BLOCK_TITLES[block_name]}")
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Cox model: hazard ratio for the transfer protocol ---
    if block_data["transferred_code"].nunique() < 2:
        print(f"Cannot fit a Cox model for {block_name}: only one group present.")
        return
    if block_data["event"].sum() == 0:
        print(f"Cannot fit a Cox model for {block_name}: no events observed.")
        return

    cph = CoxPHFitter()
    cox_data = block_data[["time", "event", "transferred_code"]]
    cph.fit(cox_data, duration_col="time", event_col="event")
    print("\nCox model results for the transfer protocol (hazard ratio):")
    cph.print_summary()


if __name__ == "__main__":
    data = load_data()
    for block in sorted(data["block"].unique()):
        analyse_block(data, block)
