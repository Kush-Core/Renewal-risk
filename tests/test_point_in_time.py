"""Features must not change when data after the as-of date is removed."""
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from renewal_risk.config import SQL_DIR
from renewal_risk.features import FEATURES

RAW = Path("data/synthetic")
CUTOFF = "2016-06-30"


def build(txn_filter: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE txn AS SELECT DISTINCT * FROM read_csv(
            ['{RAW / "transactions.csv"}', '{RAW / "transactions_v2.csv"}'],
            header = true, dateformat = '%Y%m%d',
            types = {{'transaction_date': 'DATE', 'membership_expire_date': 'DATE'}})
        WHERE {txn_filter}""")
    con.execute(f"""
        CREATE TABLE members AS
        SELECT msno, city, registered_via, registration_init_time,
               CASE WHEN bd BETWEEN 10 AND 80 THEN bd END AS age, NULLIF(gender, '') AS gender
        FROM read_csv('{RAW / "members_v3.csv"}', header = true, dateformat = '%Y%m%d',
                      types = {{'registration_init_time': 'DATE'}})""")
    for name in ["02_decisions.sql", "03_features.sql"]:
        con.execute((SQL_DIR / name).read_text())
    return con


@pytest.fixture(scope="module")
def full_and_truncated():
    if not (RAW / "transactions.csv").exists():
        pytest.skip("run scripts/make_synthetic.py first")
    cols = ", ".join(["msno", "spell_no", "n_cancels_in_spell"] + FEATURES)
    full = build("true").sql(
        f"SELECT {cols} FROM features WHERE as_of <= DATE '{CUTOFF}'").df()
    cut = build(f"transaction_date <= DATE '{CUTOFF}'").sql(
        f"SELECT {cols} FROM features WHERE as_of <= DATE '{CUTOFF}'").df()
    return full.merge(cut, on=["msno", "spell_no"], suffixes=("_full", "_cut"), validate="1:1")


def differs(df: pd.DataFrame, col: str) -> pd.Series:
    a, b = df[f"{col}_full"], df[f"{col}_cut"]
    return ~((a == b) | (a.isna() & b.isna()))


def test_same_rows_survive_truncation(full_and_truncated):
    assert len(full_and_truncated) > 0


@pytest.mark.parametrize("col", FEATURES)
def test_feature_is_point_in_time(full_and_truncated, col):
    assert differs(full_and_truncated, col).sum() == 0


def test_harness_catches_a_known_leak(full_and_truncated):
    # n_cancels_in_spell counts cancels after the as-of date: it must be flagged.
    assert differs(full_and_truncated, "n_cancels_in_spell").sum() > 0