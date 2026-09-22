"""DuckDB connection and SQL-file runner."""
import duckdb

from renewal_risk.config import DB_PATH, SQL_DIR


def connect(read_only:bool = False)->duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH), read_only=read_only)

def run_sql_file(conn:duckdb.DuckDBPyConnection, name:str)->None:
    """Run every statement in sql/<name>, in order."""
    conn.execute((SQL_DIR / name).read_text())