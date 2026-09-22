"""Module 2: build spells and decisions, then summarise them."""
from renewal_risk.db import connect, run_sql_file

SUMMARY = """
SELECT count(*)                                         AS decisions,
       round(avg(is_monthly::INT), 4)                   AS monthly_share,
       round(avg((churn IS NULL)::INT), 4)              AS censored_share,
       round(avg(churn) FILTER (WHERE is_monthly), 4)   AS monthly_churn_rate,
       round(avg((n_cancels_in_spell > 0)::INT), 4)     AS spells_with_cancel
FROM decisions"""

BY_CYCLE = """
SELECT CASE WHEN cycle = 1 THEN '1' WHEN cycle <= 6 THEN '2-6' ELSE '7+' END AS cycle_bucket,
       count(*) AS decisions, round(avg(churn), 4) AS churn_rate
FROM decisions WHERE is_monthly AND churn IS NOT NULL
GROUP BY ALL ORDER BY cycle_bucket"""


def main() -> None:
    con = connect()
    run_sql_file(con, "02_decisions.sql")
    print(con.sql(SUMMARY).df().to_string(index=False))
    print(con.sql(BY_CYCLE).df().to_string(index=False))


if __name__ == "__main__":
    main()