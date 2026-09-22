"""Global drivers, per-account reason codes, and a check that they add up."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from renewal_risk.config import MODELS_DIR, REPORTS_DIR
from renewal_risk.db import connect
from renewal_risk.features import CATEGORICAL, FEATURES
from renewal_risk.split import load

PLAIN = {
    "cancels_in_spell_to_asof": "cancelled during the current period",
    "is_auto_renew": "auto-renew setting",
    "cycle_capped": "number of consecutive renewals",
    "discount": "discount on the current plan",
    "amount_paid": "price paid", "list_price": "list price",
    "gap_days": "gap before this renewal", "price_change": "price change at this renewal",
    "relationship_no": "lapsed and returned before", "tenure_days": "time as a customer",
    "paid_to_date": "payments to date", "cancels_to_date": "past cancellations",
    "payment_method_id": "payment method", "days_to_expiry": "days to expiry",
    "age": "age", "registration_days": "time since registration", "city": "city",
    "registered_via": "sign-up channel", "gender": "gender", "left_truncated": "joined before the data starts",
    "tenure": "time as a customer (renewals, payments, days)",
}


# Features that move together are permuted together; shuffling one alone breaks the
# combination the model relies on and exaggerates its importance.
GROUPS = {"tenure": ["cycle_capped", "paid_to_date", "tenure_days", "registration_days"]}


def grouped_permutation_importance(model, X: pd.DataFrame, y: pd.Series,
                                   n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    grouped = {c for cols in GROUPS.values() for c in cols}
    groups = {**GROUPS, **{c: [c] for c in FEATURES if c not in grouped}}
    rng = np.random.default_rng(seed)
    base = brier_score_loss(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in groups.items():
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            scores.append(brier_score_loss(y, model.predict_proba(Xp)[:, 1]) - base)
        rows.append({"feature_or_group": name, "brier_increase": np.mean(scores),
                     "std": np.std(scores), "meaning": PLAIN[name]})
    return pd.DataFrame(rows).sort_values("brier_increase", ascending=False)


def source_feature(transformed_name: str) -> str:
    """Map 'num__gap_days', 'num__missingindicator_gap_days', 'cat__city_13' to the input column."""
    name = transformed_name.split("__", 1)[1].removeprefix("missingindicator_")
    for c in sorted(CATEGORICAL, key=len, reverse=True):
        if name == c or name.startswith(c + "_"):
            return c
    return name


def contributions(model, X: pd.DataFrame, X_ref: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Contribution of each feature (or group) to the log-odds, relative to the reference mean."""
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    Z, Z_ref = pre.transform(X), pre.transform(X_ref)
    coef = clf.coef_[0]
    parts = (Z - Z_ref.mean(axis=0)) * coef
    base = clf.intercept_[0] + Z_ref.mean(axis=0) @ coef
    group_of = {c: g for g, cols in GROUPS.items() for c in cols}
    names = [group_of.get(f, f) for f in map(source_feature, pre.get_feature_names_out())]
    by_feature = pd.DataFrame(parts, columns=names).T.groupby(level=0).sum().T
    return by_feature, np.full(len(X), base)


def main() -> None:
    con = connect(read_only=True)
    train, test = load(con, "train"), load(con, "test")
    model = joblib.load(MODELS_DIR / "logistic.joblib")
    iso = joblib.load(MODELS_DIR / "logistic_isotonic.joblib")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Global: which inputs matter, measured by how much Brier worsens when each is shuffled.
    sample = test.sample(n=min(10_000, len(test)), random_state=0)
    drivers = grouped_permutation_importance(model, sample[FEATURES], sample["churn"])
    drivers.to_csv(REPORTS_DIR / "drivers.csv", index=False)
    print(drivers.head(8).round(6).to_string(index=False))

    # 2. Local: top three reasons per account, and proof that the parts sum to the score.
    parts, base = contributions(model, test[FEATURES], train[FEATURES])
    logit = np.log(model.predict_proba(test[FEATURES])[:, 1] / model.predict_proba(test[FEATURES])[:, 0])
    assert np.allclose(base + parts.sum(axis=1), logit), "contributions do not add up to the score"
    top = parts.apply(lambda r: "; ".join(PLAIN[f] for f in r.nlargest(3).index if r[f] > 0), axis=1)
    out = test[["msno", "as_of"]].assign(
        p_churn=iso.predict(model.predict_proba(test[FEATURES])[:, 1]), reasons=top.to_numpy())
    out.sort_values("p_churn", ascending=False).head(10).to_csv(REPORTS_DIR / "reason_codes_top10.csv", index=False)
    print("\nhighest-risk test accounts:")
    print(out.sort_values("p_churn", ascending=False).head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()