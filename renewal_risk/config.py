"""Paths and dates in one place. Override paths with environment variables."""
import os
from pathlib import Path

RAW_DIR = Path(os.environ.get("RAW_DIR", "data/raw"))
DB_PATH = Path(os.environ.get("DB_PATH", "data/renewal.duckdb"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "reports"))
SQL_DIR = Path(__file__).resolve().parent.parent / "sql"

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "data/models"))

# out-of-time windows on scheduled_exp (inclusive). Labels resolve 30 days
# after expiry, so each later window starts after the earlier windows' labels are known.
TRAIN = ("2015-07-01", "2016-02-29")
CALIB = ("2016-03-01", "2016-04-30")
TEST = ("2016-07-01", "2016-09-30")
DATA_END = "2017-03-31"          # last transaction date; asserted against the data
AS_OF_LEAD_DAYS = 7              # must match greatest(start_date, scheduled_exp - 7) in SQL
LABEL_WINDOW_DAYS = 30
HORIZON_CYCLES = 6
# Landmarks for the long-horizon model; their 6-cycle outcomes resolve before testing.
HORIZON_TRAIN = ("2015-07-01", "2015-12-15")

# Refit on everything whose outcome is known, then score the open book.
FINAL_TRAIN = ("2015-07-01", "2016-12-31")
FINAL_CALIB = ("2017-01-01", "2017-02-28")
FINAL_HORIZON_TRAIN = ("2015-07-01", "2016-09-30")   # 6-cycle outcomes mostly observable
SCORE_CLIP = (0.001, 0.999)                           # never publish a certainty

# --- DuckDB limits -----------------------------------------------------------
# DuckDB's default memory limit is 80% of RAM. This pipeline holds large pandas
# frames and dense design matrices in the same process at the same time, so that
# default leaves nothing for Python and the run ends up in swap. Cap DuckDB and
# let it spill to temp_directory instead: spilling is far cheaper than swapping.
def _physical_memory_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return 8 * 2**30


MEMORY_FRACTION = float(os.environ.get("DUCKDB_MEMORY_FRACTION", "0.40"))
MEMORY_LIMIT = os.environ.get(
    "DUCKDB_MEMORY_LIMIT",
    f"{max(1, int(MEMORY_FRACTION * _physical_memory_bytes() / 2**30))}GB",
)
THREADS = os.environ.get("DUCKDB_THREADS")           # None = DuckDB's default (all cores)
TEMP_DIR = Path(os.environ.get("DUCKDB_TEMP_DIR", "data/duckdb_tmp"))
