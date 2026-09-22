"""Paths and dates in one place. Override paths with environment variables."""
import os
from pathlib import Path

RAW_DIR = Path(os.environ.get("RAW_DIR", "data/raw"))
DB_PATH = Path(os.environ.get("DB_PATH", "data/renewal.duckdb"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "reports"))
SQL_DIR = Path(__file__).resolve().parent.parent / "sql"