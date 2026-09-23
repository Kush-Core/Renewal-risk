"""Refit on all resolved outcomes, score the open book, write a warehouse table."""
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from renewal_risk import explain, horizon
from renewal_risk.config import FINAL_CALIB, FINAL_TRAIN, HORIZON_CYCLES, SCORE_CLIP
from renewal_risk.db import connect
from renewal_risk.features import CATEGORICAL, FEATURES, NUMERIC
from renewal_risk.model import logistic
from renewal_risk.split import LOAD_COLUMNS, load, prepare

MODEL_VERSION = f"logistic-{FINAL_TRAIN[1]}_iso-{FINAL_CALIB[1]}_landmark-h{HORIZON_CYCLES}"

DDL = """
CREATE TABLE IF NOT EXISTS churn_scores (
    msno VARCHAR NOT NULL,
    as_of DATE NOT NULL,
    scheduled_exp DATE NOT NULL,
    p_churn_next_renewal DOUBLE NOT NULL,
    survival_3_cycles DOUBLE NOT NULL,
    survival_6_cycles DOUBLE NOT NULL,
    top_reasons VARCHAR,
    model_version VARCHAR NOT NULL,
    scored_at TIMESTAMP NOT NULL,
    PRIMARY KEY (msno, as_of, model_version)
)"""

# The open book: each customer's latest monthly decision whose outcome is not yet known.
OPEN_BOOK = f"""
SELECT {", ".join(LOAD_COLUMNS)} FROM features
WHERE is_monthly AND NOT window_closed
QUALIFY row_number() OVER (PARTITION BY msno ORDER BY spell_no DESC) = 1
ORDER BY msno"""


def main() -> None:
    con = connect()
    train, calib = load(con, "final_train"), load(con, "final_calib")
    near = logistic().fit(train[FEATURES], train["churn"])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        near.predict_proba(calib[FEATURES])[:, 1], calib["churn"])

    lm = horizon.landmarks(con, "final_horizon_train")
    rows = horizon.follow_up(con, lm).merge(lm.drop(columns=["churn"]), on="msno")
    rows["offset"] = rows["offset"].astype(str)
    long = logistic(NUMERIC, CATEGORICAL + ["offset"]).fit(rows[FEATURES + ["offset"]], rows["churn"])

    book = prepare(con.sql(OPEN_BOOK).df())
    p = np.clip(iso.predict(near.predict_proba(book[FEATURES])[:, 1]), *SCORE_CLIP)
    # Survival starts from the published next-renewal score, so the two can never disagree;
    # later cycles use the landmark model (the combination back-tested in horizon.py).
    surv, s3 = 1 - p, None
    for j in range(1, HORIZON_CYCLES):
        surv = surv * (1 - long.predict_proba(horizon.with_offset(book, j)[FEATURES + ["offset"]])[:, 1])
        if j == 2:
            s3 = surv.copy()
    parts, _ = explain.contributions(near, book[FEATURES], train[FEATURES])
    reasons = explain.top_reasons(parts)

    scores = pd.DataFrame({
        "msno": book["msno"], "as_of": book["as_of"], "scheduled_exp": book["scheduled_exp"],
        "p_churn_next_renewal": p, "survival_3_cycles": s3, "survival_6_cycles": surv,
        "top_reasons": reasons, "model_version": MODEL_VERSION,
        "scored_at": datetime.now(UTC).replace(tzinfo=None)})

    # Contract checks: fail loudly before anything reaches the warehouse.
    assert scores["msno"].is_unique, "one row per customer"
    assert scores[["p_churn_next_renewal", "survival_3_cycles", "survival_6_cycles"]].notna().all().all()
    assert scores["p_churn_next_renewal"].between(*SCORE_CLIP).all()
    assert (scores["survival_6_cycles"] <= scores["survival_3_cycles"] + 1e-12).all()

    con.execute(DDL)
    con.register("new_scores", scores)
    con.execute("BEGIN TRANSACTION")
    con.execute("DELETE FROM churn_scores WHERE model_version = ?", [MODEL_VERSION])
    con.execute("INSERT INTO churn_scores SELECT * FROM new_scores")
    con.execute("COMMIT")

    print(f"model_version: {MODEL_VERSION}")
    print(con.execute("""
        SELECT count(*) AS rows, min(as_of) AS first_as_of, max(as_of) AS last_as_of,
               round(avg(p_churn_next_renewal), 4) AS mean_p,
               round(sum(p_churn_next_renewal)) AS expected_churners,
               round(avg(survival_6_cycles), 4) AS mean_survival_6
        FROM churn_scores WHERE model_version = ?""", [MODEL_VERSION]).df().to_string(index=False))


if __name__ == "__main__":
    main()