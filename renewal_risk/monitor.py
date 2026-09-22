"""The numbers that decide when to retrain."""
import joblib
import numpy as np
import pandas as pd

from renewal_risk.config import MODELS_DIR, REPORTS_DIR
from renewal_risk.db import connect
from renewal_risk.features import CATEGORICAL, FEATURES
from renewal_risk.split import load

OE_BAND = (0.9, 1.1)      # observed/expected tolerance per month (a choice; state it)
PSI_LIMIT = 0.25          # common rule of thumb: above 0.25 = a large population shift
WATCH = ["cancels_in_spell_to_asof", "tenure_days", "relationship_no", "is_auto_renew",
         "payment_method_id", "amount_paid"]


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population stability index; categorical inputs are compared level by level."""
    if reference.name in CATEGORICAL:
        levels = sorted(set(reference) | set(current))
        e = reference.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
        a = current.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    else:
        ref, cur = reference.dropna(), current.dropna()
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        e = np.histogram(np.clip(ref, edges[0], edges[-1]), edges)[0] / len(ref)
        a = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / len(cur)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main() -> None:
    con = connect(read_only=True)
    train, test = load(con, "train"), load(con, "test")
    model = joblib.load(MODELS_DIR / "logistic.joblib")
    iso = joblib.load(MODELS_DIR / "logistic_isotonic.joblib")
    test = test.assign(p=iso.predict(model.predict_proba(test[FEATURES])[:, 1]),
                       month=pd.to_datetime(test["scheduled_exp"]).dt.to_period("M").astype(str))

    monthly = test.groupby("month").agg(decisions=("churn", "size"), observed=("churn", "mean"),
                                        expected=("p", "mean"))
    monthly["obs_over_exp"] = monthly["observed"] / monthly["expected"]
    monthly["outside_band"] = ~monthly["obs_over_exp"].between(*OE_BAND)
    drift = pd.DataFrame({"feature": WATCH,
                          "psi_train_vs_test": [psi(train[c].rename(c), test[c].rename(c)) for c in WATCH]})
    drift["over_limit"] = drift["psi_train_vs_test"] > PSI_LIMIT

    # Retrain on evidence the scores are wrong; drift alone only triggers a look.
    retrain = (monthly["outside_band"] & monthly["outside_band"].shift(fill_value=False)).any()
    investigate = drift["over_limit"].any()
    print(monthly.round(4).to_string())
    print(drift.round(4).to_string(index=False))
    print(f"retrain (obs/exp outside {OE_BAND} two months running): {retrain}")
    print(f"investigate (PSI above {PSI_LIMIT}): {investigate}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    monthly.round(4).to_csv(REPORTS_DIR / "monitor_monthly.csv")
    drift.round(4).to_csv(REPORTS_DIR / "monitor_drift.csv", index=False)


if __name__ == "__main__":
    main()