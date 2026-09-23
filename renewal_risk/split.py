"""Load out-of-time windows and prove they cannot leak into each other."""

from datetime import date, timedelta

import pandas as pd

from renewal_risk.config import (
    AS_OF_LEAD_DAYS,
    CALIB,
    DATA_END,
    FINAL_CALIB,
    FINAL_HORIZON_TRAIN,
    FINAL_TRAIN,
    HORIZON_CYCLES,
    HORIZON_TRAIN,
    LABEL_WINDOW_DAYS,
    TEST,
    TRAIN,
)
from renewal_risk.features import CATEGORICAL, FEATURES, NUMERIC

WINDOWS = {
    "train": TRAIN,
    "calib": CALIB,
    "test": TEST,
    "horizon_train": HORIZON_TRAIN,
    "final_train": FINAL_TRAIN,
    "final_calib": FINAL_CALIB,
    "final_horizon_train": FINAL_HORIZON_TRAIN,
}


def window_problems(data_end: str = DATA_END) -> list[str]:
    """Return every violated rule; an empty list means the windows are safe."""
    d = {
        k: (date.fromisoformat(a), date.fromisoformat(b))
        for k, (a, b) in WINDOWS.items()
    }
    first_test_as_of = d["test"][0] - timedelta(days=AS_OF_LEAD_DAYS)
    problems = []
    for name in ["train", "calib"]:
        resolved = d[name][1] + timedelta(days=LABEL_WINDOW_DAYS)
        if resolved > first_test_as_of:
            problems.append(
                f"{name} labels resolve {resolved}, after first test as-of {first_test_as_of}"
            )
    horizon_resolved = d["horizon_train"][1] + timedelta(
        days=LABEL_WINDOW_DAYS * HORIZON_CYCLES
    )
    if horizon_resolved > first_test_as_of:
        problems.append(
            f"horizon_train outcomes resolve {horizon_resolved}, after first test as-of"
        )
    if d["train"][1] >= d["calib"][0]:
        problems.append("train and calib windows overlap")
    horizon_end = d["test"][1] + timedelta(days=LABEL_WINDOW_DAYS * HORIZON_CYCLES)
    if horizon_end > date.fromisoformat(data_end):
        problems.append(
            f"{HORIZON_CYCLES}-cycle outcomes for test end {horizon_end}, after data end"
        )
    return problems


# `features` has 36 columns; these are the only ones the models and reports read.
# Pulling the rest inflates every frame below for nothing (final_train is 13.8M rows).
EXTRA_COLUMNS = ["msno", "churn", "cycle", "scheduled_exp", "as_of", "first_txn_day"]
LOAD_COLUMNS = FEATURES + EXTRA_COLUMNS


def checked_window(con, window: str) -> tuple[str, str]:
    """The window's bounds, after re-proving against the data that it cannot leak."""
    actual_end = con.sql("SELECT max(transaction_date) FROM txn").fetchone()[0]
    problems = window_problems(str(actual_end))
    assert not problems, problems
    return WINDOWS[window]


def load(con, window: str) -> pd.DataFrame:
    """Monthly decisions with a closed outcome window whose scheduled expiry is in the window."""
    start, end = checked_window(con, window)
    df = con.execute(
        f"""SELECT {", ".join(LOAD_COLUMNS)} FROM features
           WHERE is_monthly AND window_closed AND scheduled_exp BETWEEN ? AND ?""",
        [start, end],
    ).df()
    return prepare(df, copy=False)          # the frame is ours alone; no need to duplicate it


def _levels(s: pd.Series) -> pd.Series:
    """The same strings as astype("string"), stored once per level, not once per row.

    These seven columns were 60% of a loaded frame at 362 bytes a row; as levels
    they are 7. On final_train that is the difference between 8.2 GB and 2.9 GB.
    """
    c = s.astype("category")
    c = c.cat.rename_categories(c.cat.categories.astype(str))
    if c.isna().any():
        if "missing" not in c.cat.categories:
            c = c.cat.add_categories("missing")
        c = c.fillna("missing")
    return c


def prepare(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Categoricals as strings with an explicit 'missing' level; numerics as float.

    `copy=False` converts in place. Pass it only for a frame nobody else holds:
    copying a multi-GB frame just to retype 20 columns doubles the peak.
    """
    if copy:
        df = df.copy()
    for c in CATEGORICAL:
        df[c] = _levels(df[c])
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df
