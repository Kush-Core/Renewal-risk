"""Paths and dates in one place. Override paths with environment variables."""
import os
from pathlib import Path

RAW_DIR = Path(os.environ.get("RAW_DIR", "data/raw"))
DB_PATH = Path(os.environ.get("DB_PATH", "data/renewal.duckdb"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "reports"))
SQL_DIR = Path(__file__).resolve().parent.parent / "sql"

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "data/models"))

# Module 5-8: out-of-time windows on scheduled_exp (inclusive). Labels resolve 30 days
# after expiry, so each later window starts after the earlier windows' labels are known.
TRAIN = ("2015-07-01", "2016-02-29")
CALIB = ("2016-03-01", "2016-04-30")
TEST = ("2016-07-01", "2016-09-30")
DATA_END = "2017-03-31"          # last transaction date; asserted against the data
AS_OF_LEAD_DAYS = 7              # must match greatest(start_date, scheduled_exp - 7) in SQL
LABEL_WINDOW_DAYS = 30
HORIZON_CYCLES = 6
# Module 8: landmarks for the long-horizon model; their 6-cycle outcomes resolve before testing.
HORIZON_TRAIN = ("2015-07-01", "2015-12-15")