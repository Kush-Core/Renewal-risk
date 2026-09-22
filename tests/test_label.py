"""Module 2: the label logic, checked against the organisers' worked examples."""
from datetime import date

import duckdb

from renewal_risk.config import SQL_DIR

TXN_DDL = """CREATE TABLE txn (msno VARCHAR, payment_method_id INT, payment_plan_days INT,
    plan_list_price INT, actual_amount_paid INT, is_auto_renew INT, transaction_date DATE,
    membership_expire_date DATE, is_cancel INT)"""
FILLER = ("zz", date(2017, 9, 1), date(2017, 10, 1), 0)  # moves data_end past every window


def build(rows):
    con = duckdb.connect()
    con.execute(TXN_DDL)
    con.executemany("INSERT INTO txn VALUES (?, 41, 30, 149, 149, 1, ?, ?, ?)", rows)
    con.execute((SQL_DIR / "02_decisions.sql").read_text())
    return con


def spells(con, msno):
    return con.execute(
        """SELECT spell_no, effective_exp, churn, relationship_no, cycle, n_cancels_in_spell
           FROM decisions WHERE msno = ? ORDER BY spell_no""", [msno]).fetchall()


def test_late_renewal_is_churn():                      # organisers' example 1
    con = build([("a", date(2017, 1, 1), date(2017, 2, 28), 0),
                 ("a", date(2017, 2, 25), date(2017, 3, 15), 0),
                 ("a", date(2017, 4, 30), date(2017, 5, 20), 0), FILLER])
    assert spells(con, "a")[1][:3] == (2, date(2017, 3, 15), 1)


def test_cancel_for_plan_change_is_not_churn():        # organisers' example 2
    con = build([("b", date(2017, 1, 1), date(2017, 2, 28), 0),
                 ("b", date(2017, 2, 25), date(2017, 4, 3), 0),
                 ("b", date(2017, 3, 15), date(2017, 3, 16), 1),
                 ("b", date(2017, 4, 1), date(2017, 6, 30), 0), FILLER])
    assert spells(con, "b")[1][:3] == (2, date(2017, 3, 16), 0)


def test_open_window_is_censored_even_if_renewal_is_visible():
    con = build([("c", date(2017, 3, 1), date(2017, 3, 20), 0),
                 ("c", date(2017, 3, 19), date(2017, 4, 18), 0)])
    assert [s[2] for s in spells(con, "c")] == [None, None]


def test_same_day_cancel_belongs_to_previous_spell():
    con = build([("e", date(2016, 1, 1), date(2016, 1, 31), 0),
                 ("e", date(2016, 1, 20), date(2016, 1, 21), 1),
                 ("e", date(2016, 1, 20), date(2016, 3, 20), 0), FILLER])
    first, second = spells(con, "e")
    assert first[1] == date(2016, 1, 21) and first[5] == 1 and first[2] == 0
    assert second[5] == 0


def test_lapse_starts_a_new_relationship():
    con = build([("f", date(2016, 1, 1), date(2016, 1, 31), 0),
                 ("f", date(2016, 3, 15), date(2016, 4, 14), 0),   # 44-day gap
                 ("f", date(2016, 4, 14), date(2016, 5, 14), 0), FILLER])
    rows = spells(con, "f")
    assert [r[2] for r in rows] == [1, 0, 1]          # last spell: no renewal before data_end
    assert [(r[3], r[4]) for r in rows] == [(1, 1), (2, 1), (2, 2)]