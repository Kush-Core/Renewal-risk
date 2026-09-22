"""A rule-based forecast (segment churn rates) and its back-test."""
import numpy as np
import pandas as pd

from renewal_risk.config import REPORTS_DIR
from renewal_risk.db import connect
from renewal_risk.split import load


def segment(df: pd.DataFrame) -> pd.Series:
    bucket = np.select([df["cycle"] == 1, df["cycle"] <= 6], ["1", "2-6"], "7+")
    return "auto=" + df["is_auto_renew"].astype(str) + "|cycle=" + bucket


def fit_rates(train: pd.DataFrame) -> pd.Series:
    return train.assign(seg=segment(train)).groupby("seg")["churn"].mean()


def predict(rates: pd.Series, df: pd.DataFrame) -> np.ndarray:
    return segment(df).map(rates).fillna(rates.mean()).to_numpy()


def backtest(df: pd.DataFrame, pred: np.ndarray, by: pd.Series) -> pd.DataFrame:
    t = (pd.DataFrame({"group": by.to_numpy(), "y": df["churn"].to_numpy(), "p": pred})
         .groupby("group")
         .agg(decisions=("y", "size"), observed=("y", "mean"), predicted=("p", "mean")))
    t["obs_over_exp"] = t["observed"] / t["predicted"]
    t["gap_pp"] = 100 * (t["predicted"] - t["observed"])
    # z-score of the gap under the forecast's own binomial noise
    t["z"] = (t["observed"] - t["predicted"]) / np.sqrt(t["predicted"] * (1 - t["predicted"]) / t["decisions"])
    material = (abs(t["obs_over_exp"] - 1) > 0.10) & (abs(t["z"]) >= 2)
    t["verdict"] = np.where(~material, "within tolerance",
                   np.where(t["obs_over_exp"] > 1, "optimistic (under-predicts churn)",
                            "conservative (over-predicts churn)"))
    return t.round(4)


def main() -> None:
    con = connect(read_only=True)
    train, test = load(con, "train"), load(con, "test")
    rates = fit_rates(train)
    pred = predict(rates, test)
    cohort = pd.to_datetime(test["first_txn_day"]).dt.to_period("Q").astype(str)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("segment churn rates learned on train:\n", rates.round(4).to_string())
    for name, by in [("segment", segment(test)), ("cohort", cohort)]:
        t = backtest(test, pred, by)
        t.to_csv(REPORTS_DIR / f"backtest_{name}.csv")
        print(f"\nback-test by {name}:\n{t.to_string()}")
    overall = test["churn"].mean() / pred.mean()
    print(f"\noverall observed/expected on test: {overall:.4f}")


if __name__ == "__main__":
    main()