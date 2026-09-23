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


CHUNK = 500_000          # rows transformed at once; bounds the dense matrix below


def _dense(Z) -> np.ndarray:
    return Z.toarray() if hasattr(Z, "toarray") else Z


def _group_matrix(pre) -> tuple[list[str], np.ndarray]:
    """Indicator matrix mapping each transformed column to its feature or group."""
    group_of = {c: g for g, cols in GROUPS.items() for c in cols}
    names = [group_of.get(f, f) for f in map(source_feature, pre.get_feature_names_out())]
    groups = sorted(set(names))                       # groupby(level=0) sorts the names
    G = np.zeros((len(names), len(groups)))
    G[np.arange(len(names)), [groups.index(n) for n in names]] = 1.0
    return groups, G


def _transformed_mean(pre, X: pd.DataFrame, chunk: int) -> np.ndarray:
    """Column means of the transformed reference, without holding it all at once."""
    total = np.zeros(len(pre.get_feature_names_out()))
    for lo in range(0, len(X), chunk):
        total += _dense(pre.transform(X.iloc[lo:lo + chunk])).sum(axis=0)
    return total / len(X)


def contributions(model, X: pd.DataFrame, X_ref: pd.DataFrame,
                  chunk: int = CHUNK) -> tuple[pd.DataFrame, np.ndarray]:
    """Contribution of each feature (or group) to the log-odds, relative to the reference mean.

    Folding the coefficients into the grouping matrix turns the whole thing into one
    matrix product per chunk. The frames here reach millions of rows, where building
    an (N x 88) frame and transposing it to group the columns does not fit in memory.
    """
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    coef = clf.coef_[0]
    groups, G = _group_matrix(pre)
    ref_mean = _transformed_mean(pre, X_ref, chunk)
    W = G * coef[:, None]
    offset = (ref_mean * coef) @ G
    out = np.empty((len(X), len(groups)))
    for lo in range(0, len(X), chunk):
        out[lo:lo + chunk] = _dense(pre.transform(X.iloc[lo:lo + chunk])) @ W - offset
    base = clf.intercept_[0] + ref_mean @ coef
    return pd.DataFrame(out, columns=groups), np.full(len(X), base)


def top_reasons(parts: pd.DataFrame, k: int = 3) -> np.ndarray:
    """The k largest positive contributions per row, in words.

    Same answer as nlargest(k) applied row by row, without a Python call per row:
    the test window is 2.5M rows and the open book scored next to it 1.0M.
    """
    v = parts.to_numpy()
    words = np.array([PLAIN[c] for c in parts.columns], dtype=object)
    order = np.argsort(-v, axis=1, kind="stable")[:, :k]
    picked, keep = words[order], np.take_along_axis(v, order, axis=1) > 0
    out = np.where(keep[:, 0], picked[:, 0], "").astype(object)
    for j in range(1, order.shape[1]):
        sep = np.where(out != "", "; ", "").astype(object)
        out = np.where(keep[:, j], out + sep + picked[:, j], out)
    return out


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
    X = test[FEATURES]                      # materialise the column subset once
    p_raw = model.predict_proba(X)[:, 1]    # and score it once, not three times
    parts, base = contributions(model, X, train[FEATURES])
    logit = np.log(p_raw / (1 - p_raw))
    assert np.allclose(base + parts.sum(axis=1), logit), "contributions do not add up to the score"
    out = test[["msno", "as_of"]].assign(
        p_churn=iso.predict(p_raw), reasons=top_reasons(parts))
    out.sort_values("p_churn", ascending=False).head(10).to_csv(REPORTS_DIR / "reason_codes_top10.csv", index=False)
    print("\nhighest-risk test accounts:")
    print(out.sort_values("p_churn", ascending=False).head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()