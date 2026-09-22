"""Module 1: load raw CSVs into DuckDB with explicit types."""
from renewal_risk.config import RAW_DIR
from renewal_risk.db import connect

TXN_COLUMNS = {
    "msno": "VARCHAR", "payment_method_id": "INTEGER", "payment_plan_days": "INTEGER",
    "plan_list_price": "INTEGER", "actual_amount_paid": "INTEGER", "is_auto_renew": "INTEGER",
    "transaction_date": "DATE", "membership_expire_date": "DATE", "is_cancel": "INTEGER",
}
MEMBER_COLUMNS = {
    "msno": "VARCHAR", "city": "INTEGER", "bd": "INTEGER", "gender": "VARCHAR",
    "registered_via": "INTEGER", "registration_init_time": "DATE",
}


def read_csv_sql(files: list[str], columns: dict[str, str], with_filename: bool = False) -> str:
    cols = "{" + ", ".join(f"'{k}': '{v}'" for k, v in columns.items()) + "}"
    extra = ", filename = true" if with_filename else ""
    return (f"read_csv({files!r}, header = true, types = {cols}, "
            f"dateformat = '%Y%m%d'{extra})")


def main() -> None:
    con = connect()
    txn_files = [str(RAW_DIR / "transactions.csv"), str(RAW_DIR / "transactions_v2.csv")]
    con.execute("CREATE OR REPLACE TABLE txn_raw AS SELECT * FROM "
                + read_csv_sql(txn_files, TXN_COLUMNS, with_filename=True))
    con.execute("CREATE OR REPLACE TABLE members_raw AS SELECT * FROM "
                + read_csv_sql([str(RAW_DIR / "members_v3.csv")], MEMBER_COLUMNS))
    con.execute("CREATE OR REPLACE TABLE labels_official AS SELECT * FROM "
                + read_csv_sql([str(RAW_DIR / "train_v2.csv")], {"msno": "VARCHAR", "is_churn": "INTEGER"}))
    for t in ["txn_raw", "members_raw", "labels_official"]:
        print(t, con.sql(f"SELECT count(*) FROM {t}").fetchone()[0])


if __name__ == "__main__":
    main()