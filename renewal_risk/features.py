"""Build point-in-time features. The lists below are the model's only inputs."""
from renewal_risk.db import connect, run_sql_file

NUMERIC = ["amount_paid", "list_price", "discount", "price_change", "gap_days",
           "paid_to_date", "cancels_to_date", "cancels_in_spell_to_asof",
           "tenure_days", "days_to_expiry", "age", "registration_days", "relationship_no"]
CATEGORICAL = ["cycle_capped", "is_auto_renew", "left_truncated", "payment_method_id",
               "city", "registered_via", "gender"]
FEATURES = NUMERIC + CATEGORICAL


def main() -> None:
    con = connect()
    run_sql_file(con, "03_features.sql")
    null_rates = con.sql(
        "SELECT " + ", ".join(f"round(avg(({c} IS NULL)::INT), 4) AS {c}" for c in FEATURES)
        + " FROM features WHERE is_monthly").df().T.rename(columns={0: "null_rate"})
    print(con.sql("SELECT count(*) AS rows FROM features").fetchone())
    print(null_rates[null_rates["null_rate"] > 0].to_string())


if __name__ == "__main__":
    main()