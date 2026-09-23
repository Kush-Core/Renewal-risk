"""Accuracy against outcomes (calibration) first, ranking second."""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from renewal_risk import baseline
from renewal_risk.config import MODELS_DIR, REPORTS_DIR
from renewal_risk.db import connect
from renewal_risk.features import FEATURES
from renewal_risk.split import load


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Slope of a logistic fit of outcomes on logit(p); 1.0 means well spread."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(logit, y).coef_[0, 0])


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"observed_rate": y.mean(), "mean_predicted": p.mean(), "obs_over_exp": y.mean() / p.mean(),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
            "auc": roc_auc_score(y, p), "calibration_slope": calibration_slope(y, p)}


def brier_gain_ci(df: pd.DataFrame, p_model: np.ndarray, p_base: np.ndarray,
                  reps: int = 1000, seed: int = 0, cells: int = 16_000_000
                  ) -> tuple[float, float, float]:
    """Brier(baseline) - Brier(model), with a 95% CI from resampling customers, not rows.

    The resample is drawn in batches of rows. One (reps, customers) array would be
    8 GB on the test window; numpy fills a 2-D draw in C order, so batching consumes
    the same stream in the same order and returns the same numbers.
    """
    y = df["churn"].to_numpy()
    # groupby on a million 44-character hashes is slow; factorize(sort=True) gives
    # the same per-customer order as groupby("msno") and bincount does the sums.
    codes, users = pd.factorize(df["msno"].to_numpy(), sort=True)
    d = (p_base - y) ** 2 - (p_model - y) ** 2
    sum_d = np.bincount(codes, weights=d, minlength=len(users))
    n_per = np.bincount(codes, minlength=len(users)).astype(float)
    rng = np.random.default_rng(seed)
    batch = max(1, min(reps, cells // max(1, len(users))))
    boot = np.empty(reps)
    for lo in range(0, reps, batch):
        k = min(batch, reps - lo)
        idx = rng.integers(0, len(users), size=(k, len(users)))
        boot[lo:lo + k] = sum_d[idx].sum(axis=1) / n_per[idx].sum(axis=1)
    return sum_d.sum() / n_per.sum(), *np.percentile(boot, [2.5, 97.5])


def predictions(con) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    train, calib, test = load(con, "train"), load(con, "calib"), load(con, "test")
    rates = baseline.fit_rates(train)
    preds = {"baseline": baseline.predict(rates, test)}
    for name in ["logistic", "boosting"]:
        model = joblib.load(MODELS_DIR / f"{name}.joblib")
        raw_test = model.predict_proba(test[FEATURES])[:, 1]
        raw_calib = model.predict_proba(calib[FEATURES])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_calib, calib["churn"])
        joblib.dump(iso, MODELS_DIR / f"{name}_isotonic.joblib")
        preds[name] = raw_test
        preds[f"{name}+isotonic"] = iso.predict(raw_test)
    return test, preds


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test, preds = predictions(connect(read_only=True))
    y = test["churn"].to_numpy()
    table = pd.DataFrame({k: metrics(y, p) for k, p in preds.items()}).T
    table["brier_gain_vs_baseline"] = np.nan
    table["gain_ci_low"] = np.nan
    table["gain_ci_high"] = np.nan
    for k, p in preds.items():
        if k != "baseline":
            table.loc[k, ["brier_gain_vs_baseline", "gain_ci_low", "gain_ci_high"]] = \
                brier_gain_ci(test, p, preds["baseline"])
    print(f"test decisions: {len(test)}, customers: {test['msno'].nunique()}")
    print(table.round(5).to_string())
    table.round(5).to_csv(REPORTS_DIR / "metrics_test.csv")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    curves = {k: calibration_curve(y, p, n_bins=10, strategy="quantile") for k, p in preds.items()}
    for k, (frac, mean_p) in curves.items():
        ax.plot(mean_p, frac, marker="o", ms=3, label=k)
    lim = 1.05 * max(max(f.max(), m.max()) for f, m in curves.values())
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="mean predicted churn (decile)",
           ylabel="observed churn", title="Calibration on the test window")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration_test.png", dpi=150)

    production = "logistic+isotonic"   # fixed in advance, never chosen from this table
    seg = baseline.backtest(test, preds[production], baseline.segment(test))
    seg.to_csv(REPORTS_DIR / "backtest_segment_model.csv")
    print(f"\n{production} by segment:\n{seg.to_string()}")


if __name__ == "__main__":
    main()