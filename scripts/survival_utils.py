"""
survival_utils.py
------------------
Shared helper functions used by all the survival-analysis scripts in this
repository (Assays 1, 2 and 3 of the honey bee brood humidity study).

The raw data are recorded as a binary "still alive" matrix: one column per
observation day (day_3, day_4, ... day_22), with a value of 1 if the larva/
pupa was alive on that day and a blank (NaN) once it has died (the column
is simply left empty for every day after death, i.e. the record is
right-censored from that point on if the individual survives to the last
observation day, D22).

`row_to_time_and_event` converts one row of that matrix into the two
values required by a standard survival analysis:

- ``time``  : the last day on which the individual was confirmed alive.
- ``event`` : 1 if death was observed within the study period (day < 22),
              0 if the individual was still alive at day 22 (right-censored).
"""

import pandas as pd


def row_to_time_and_event(row, day_columns, day_to_time, time_if_dead_before_first_check=None):
    """Convert one row of the alive/dead matrix into (time, event).

    Parameters
    ----------
    row : pandas.Series
        A single row containing one column per observation day.
    day_columns : list[str]
        Ordered list of column names to scan, e.g. ["day_3", "day_4", ...].
    day_to_time : dict[str, int]
        Maps each column name in `day_columns` to the actual day number it
        represents (e.g. {"day_3": 3, "day_15": 15, "day_22": 22}).
    time_if_dead_before_first_check : int, optional
        Time to assign when the larva/pupa was already dead at the very
        first check (i.e. every day column is empty). Defaults to the day
        number of the first entry in `day_columns` minus 1.

    Returns
    -------
    (time, event) : tuple[int, int]
        `event = 1` means death was observed; `event = 0` means the
        individual was still alive (censored) at the last observation day.
    """
    last_day_alive = None
    for day_col in day_columns:
        value = row[day_col]
        # A row only "survives" a day if the cell is exactly 1 (not NaN,
        # not 0, not any other stray value that sometimes appears in the
        # raw sheets).
        if pd.isna(value) or not (isinstance(value, (int, float)) and value == 1):
            break
        last_day_alive = day_to_time[day_col]

    if last_day_alive is None:
        if time_if_dead_before_first_check is None:
            time_if_dead_before_first_check = day_to_time[day_columns[0]] - 1
        return time_if_dead_before_first_check, 1

    last_possible_day = max(day_to_time.values())
    event = 0 if last_day_alive == last_possible_day else 1
    return last_day_alive, event


def add_time_and_event_columns(df, day_columns, day_to_time, **kwargs):
    """Apply `row_to_time_and_event` to a whole dataframe and attach the
    resulting `time` and `event` columns."""
    df = df.copy()
    df[["time", "event"]] = df.apply(
        lambda row: row_to_time_and_event(row, day_columns, day_to_time, **kwargs),
        axis=1,
        result_type="expand",
    )
    return df
