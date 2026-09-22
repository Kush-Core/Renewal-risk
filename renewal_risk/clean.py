"""Module 1: apply the cleaning rules and print the counts DECISIONS.md needs."""

from renewal_risk.db import connect, run_sql_file


def main() -> None:
    con = connect()
    run_sql_file(con, "01_clean.sql")
    for raw, clean in [("txn_raw", "txn"), ("members_raw", "members")]:
        before = con.sql(f"SELECT count(*) FROM {raw}").fetchone()[0]
        after = con.sql(f"SELECT count(*) FROM {clean}").fetchone()[0]
        print(f"{raw} -> {clean}: {before} -> {after} rows ({before - after} removed)")


if __name__ == "__main__":
    main()
