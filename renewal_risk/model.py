"""Discrete-time hazard models on renewal decisions."""
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from renewal_risk.config import MODELS_DIR
from renewal_risk.db import connect
from renewal_risk.features import CATEGORICAL, FEATURES, NUMERIC
from renewal_risk.split import load


def logistic(numeric=NUMERIC, categorical=CATEGORICAL) -> Pipeline:
    # Sparse design matrix: the one-hot block lands at exactly the 0.3 density that
    # ColumnTransformer would densify by default, and dense is 640 bytes a row.
    # final_train is 13.8M rows, so dense means 8.8 GB and an hour of swapping;
    # sparse is 4 GB and stays in memory. Same model either way.
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                              sparse_output=True), categorical),
    ], sparse_threshold=1.0)
    # No class_weight: re-weighting classes distorts the probabilities finance will sum.
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=1.0, max_iter=2000))])


def boosting() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    is_cat = [True] * len(CATEGORICAL) + [False] * len(NUMERIC)
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                         min_samples_leaf=100, l2_regularization=1.0,
                                         categorical_features=is_cat, random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    con = connect(read_only=True)
    train = load(con, "train")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"train decisions: {len(train)}, churn rate: {train['churn'].mean():.4f}")
    for name, make in [("logistic", logistic), ("boosting", boosting)]:
        model = make().fit(train[FEATURES], train["churn"])
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"saved {name}; mean train prediction "
              f"{model.predict_proba(train[FEATURES])[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()