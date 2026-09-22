"""A long-horizon survival curve, back-tested over 6 renewal cycles."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from renewal_risk import baseline
from renewal_risk.config import HORIZON_CYCLES, MODELS_DIR, REPORTS_DIR
from renewal_risk.db import connect
from renewal_risk.features import CATEGORICAL, FEATURES, NUMERIC
from renewal_risk.model import logistic
from renewal_risk.split import load

# Follow-up decisions of the same relationship, up to HORIZON_CYCLES cycles after the landmark.
FOLLOW_UP = f"""
SELECT i.msno, (d.cycle - i.cycle)::INT AS offset, d.churn
FROM idx i
JOIN decisions d
  ON d.msno = i.msno AND d.relationship_no = i.relationship_no
 AND d.cycle BETWEEN i.cycle AND i.cycle + {HORIZON_CYCLES - 1}
WHERE d.churn IS NOT NULL"""


def landmarks(con, window: str) -> pd.DataFrame:
    """One landmark per customer: their first decision in the window."""
    df = load(con, window)
    return df.sort_values("scheduled_exp").groupby("msno", as_index=False).first()


def follow_up(con, index: pd.DataFrame) -> pd.DataFrame:
    con.register("idx", index[["msno", "relationship_no", "cycle"]])
    return con.sql(FOLLOW_UP).df()


def with_offset(index: pd.DataFrame, j: int) -> pd.DataFrame:
    return index.assign(offset=str(j))


def roll_forward(index: pd.DataFrame, j: int) -> pd.DataFrame:
    """The naive alternative: advance the clock j cycles, freeze everything else."""
    r = index.copy()
    r["cycle"] = r["cycle"] + j
    r["cycle_capped"] = np.minimum(r["cycle"], 12).astype(int).astype(str)
    r[["tenure_days", "registration_days"]] += 30 * j
    r["paid_to_date"] += j
    return r


def main() -> None:
    con = connect(read_only=True)

    # 1. Landmark model: covariates frozen at the landmark, time = cycles since landmark.
    lm = landmarks(con, "horizon_train")
    rows = follow_up(con, lm).merge(lm.drop(columns=["churn"]), on="msno")
    rows["offset"] = rows["offset"].astype(str)
    model = logistic(NUMERIC, CATEGORICAL + ["offset"]).fit(rows[FEATURES + ["offset"]], rows["churn"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "horizon_logistic.joblib")
    print(f"landmark model: {len(lm)} landmarks, {len(rows)} person-period rows")

    # 2. Observed survival of the test cohort: Kaplan-Meier, because slow survivors are censored.
    index = landmarks(con, "test")
    raw = follow_up(con, index)
    churned_at = raw[raw["churn"] == 1].groupby("msno")["offset"].min()
    last_closed = raw.groupby("msno")["offset"].max()
    churned = index["msno"].map(churned_at).notna()
    duration = np.where(churned, index["msno"].map(churned_at) + 1, index["msno"].map(last_closed) + 1)
    km = KaplanMeierFitter().fit(duration, churned.astype(int))

    # 3. Predicted curves for the same customers.
    near = joblib.load(MODELS_DIR / "logistic.joblib")
    near_iso = joblib.load(MODELS_DIR / "logistic_isotonic.joblib")
    rates = baseline.fit_rates(load(con, "train"))
    s = {"landmark_model": np.ones(len(index)), "published_hybrid": np.ones(len(index)),
         "near_term_model_rolled": np.ones(len(index)), "baseline": np.ones(len(index))}
    p_next = near_iso.predict(near.predict_proba(index[FEATURES])[:, 1])
    out = []
    for h in range(1, HORIZON_CYCLES + 1):
        j = h - 1
        hazard = model.predict_proba(with_offset(index, j)[FEATURES + ["offset"]])[:, 1]
        s["landmark_model"] *= 1 - hazard
        # What Module 10 publishes: the near-term score for the next renewal, landmark after.
        s["published_hybrid"] *= 1 - (p_next if j == 0 else hazard)
        rolled = roll_forward(index, j)
        s["near_term_model_rolled"] *= 1 - near_iso.predict(near.predict_proba(rolled[FEATURES])[:, 1])
        s["baseline"] *= 1 - baseline.predict(rates, rolled)
        out.append({"h": h, "at_risk": int((duration >= h).sum()),
                    "observed_km": float(km.predict(h)), **{k: v.mean() for k, v in s.items()}})
    t = pd.DataFrame(out)
    for k in s:
        t[f"{k}_error_pp"] = 100 * (t[k] - t["observed_km"])
    print(f"test landmarks: {len(index)}")
    print(t.round(4).to_string(index=False))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(REPORTS_DIR / "horizon.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for col in ["observed_km", *s]:
        ax.plot(t["h"], t[col], marker="o", label=col)
    ax.set(xlabel="renewal cycles after the landmark", ylabel="share still subscribed",
           title="Multi-cycle survival, test cohort")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "horizon.png", dpi=150)


if __name__ == "__main__":
    main()