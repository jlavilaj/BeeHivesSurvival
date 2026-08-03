"""
assay1_humidity_kaplan_meier_cox.py
------------------------------------
Assay 1: effect of the two humidity regimes (95/80/60 vs 70/70/60) on
in vitro brood survival, evaluated under the base OECD/EFSA laboratory
protocol (no larvae transfer to new cells).

Reproduces:
- The pooled Kaplan-Meier survival curves per humidity regime
  ("Kaplan-Meier per humidity regime" figure in the manuscript).
- The Cox proportional-hazards model comparing the two regimes
  (hazard ratio reported in Section 3.1 of the manuscript).
- The Schoenfeld-residual proportional-hazards assumption check.

Input data: data/processed/assay1_humidity_regimes.csv
(one row per larva; day_3 ... day_22 = 1 while alive, blank after death)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

from survival_utils import add_time_and_event_columns

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "assay1_humidity_regimes.csv"

DAY_COLUMNS = ["day_3", "day_4", "day_5", "day_6", "day_7", "day_8", "day_15", "day_22"]
DAY_TO_TIME = {"day_3": 3, "day_4": 4, "day_5": 5, "day_6": 6, "day_7": 7, "day_8": 8, "day_15": 15, "day_22": 22}

HUMIDITY_LABELS = {1: "95/80/60", 2: "70/70/60"}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = add_time_and_event_columns(df, DAY_COLUMNS, DAY_TO_TIME)
    return df


def plot_kaplan_meier(df):
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for regime_code, group in df.groupby("humidity_regime_code"):
        label = HUMIDITY_LABELS[regime_code]
        kmf.fit(durations=group["time"], event_observed=group["event"], label=f"Humidity {label}")
        kmf.plot_survival_function(ci_show=True)

    plt.title("Kaplan-Meier per humidity regime")
    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fit_cox_model(df):
    cph = CoxPHFitter()
    df_cox = df[["time", "event", "humidity_regime_code"]]
    cph.fit(df_cox, duration_col="time", event_col="event")
    cph.print_summary()

    # Schoenfeld-residual test of the proportional-hazards assumption.
    results = proportional_hazard_test(cph, df_cox, time_transform="rank")
    print("\nPROPORTIONAL HAZARDS ASSUMPTION TEST")
    print(results.summary)
    cph.check_assumptions(df_cox, show_plots=True, p_value_threshold=0.05)

    return cph


if __name__ == "__main__":
    data = load_data()
    plot_kaplan_meier(data)
    fit_cox_model(data)
