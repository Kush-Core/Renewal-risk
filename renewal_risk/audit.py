"""Module 1: profile the raw data before deciding any cleaning rule."""
from renewal_risk.config import REPORTS_DIR
from renewal_risk.db import connect

CHECKS = {
    "rows and date range per file": """
        SELECT regexp_extract(filename, '[^/\\\\]+$') AS file, count(*) AS rows,
               min(transaction_date) AS first_txn, max(transaction_date) AS last_txn
        FROM txn_raw GROUP BY ALL ORDER BY file""",
    "users": """
        SELECT count(DISTINCT t.msno) AS users_in_txn,
               count(DISTINCT m.msno) AS of_which_in_members
        FROM txn_raw t LEFT JOIN members_raw m USING (msno)""",
    "exact duplicate rows (ignoring source file)": """
        SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT * EXCLUDE (filename) FROM txn_raw))
               AS duplicate_rows
        FROM txn_raw""",
    "flags": """
        SELECT round(avg(is_cancel), 4) AS cancel_share,
               round(avg(is_auto_renew), 4) AS auto_renew_share,
               sum((is_cancel = 0 AND membership_expire_date < transaction_date)::INT) AS paid_expiry_before_txn,
               sum((is_cancel = 0 AND payment_plan_days = 0)::INT) AS paid_zero_plan_days,
               sum((is_cancel = 0 AND actual_amount_paid = 0)::INT) AS paid_zero_amount
        FROM txn_raw""",
    "plan length, paid rows (top 10)": """
        SELECT payment_plan_days, count(*) AS rows,
               round(count(*) / sum(count(*)) OVER (), 4) AS share
        FROM txn_raw WHERE is_cancel = 0
        GROUP BY ALL ORDER BY rows DESC LIMIT 10""",
    "users mixing monthly and other plan lengths": """
        SELECT round(avg(mixed::INT), 4) AS share_of_users
        FROM (SELECT msno, count(DISTINCT payment_plan_days IN (30, 31)) = 2 AS mixed
              FROM txn_raw WHERE is_cancel = 0 GROUP BY msno)""",
    "members: age (bd) outside 10-80, blank gender": """
        SELECT sum((bd NOT BETWEEN 10 AND 80)::INT) AS bd_out_of_range,
               sum((gender IS NULL OR gender = '')::INT) AS gender_missing,
               count(*) - count(DISTINCT msno) AS duplicate_msno,
               count(*) AS members
        FROM members_raw""",
}


def main() -> None:
    con = connect(read_only=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Data audit", ""]
    for title, sql in CHECKS.items():
        df = con.sql(sql).df()
        print(f"\n== {title}\n{df.to_string(index=False)}")
        lines += [f"## {title}", "", df.to_markdown(index=False), ""]
    (REPORTS_DIR / "audit.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()