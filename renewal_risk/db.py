"""DuckDB connection and SQL-file runner."""
import duckdb

from renewal_risk.config import DB_PATH, MEMORY_LIMIT, SQL_DIR, TEMP_DIR, THREADS


def connect(read_only: bool = False, memory_limit: str | None = None) -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    # Bound DuckDB so it cannot crowd out the pandas frames and design matrices
    # this process builds next; over the limit it spills to disk, which is much
    # cheaper than letting the OS swap the whole process.
    con.execute(f"SET memory_limit = '{memory_limit or MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{TEMP_DIR}'")
    # Explicit because the reports depend on it: this is what carries the
    # deterministic order of the features table through into every load().
    con.execute("SET preserve_insertion_order = true")
    if THREADS:
        con.execute(f"SET threads = {int(THREADS)}")
    return con


def run_sql_file(conn: duckdb.DuckDBPyConnection, name: str) -> None:
    """Run every statement in sql/<name>, in order."""
    conn.execute((SQL_DIR / name).read_text())
