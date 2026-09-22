"""Kaplan-Meier survival over renewal cycles, by segment."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from renewal_risk.config import REPORTS_DIR
from renewal_risk.db import connect

# One row per customer relationship whose start we actually observe.
RELATIONSHIPS = """
SELECT msno, relationship_no,
       coalesce(min(cycle) FILTER (WHERE churn = 1),
                max(cycle) FILTER (WHERE churn IS NOT NULL)) AS duration,
       (min(cycle) FILTER (WHERE churn = 1) IS NOT NULL)::INT AS event,
       arg_min(is_auto_renew, cycle)                         AS auto_renew_at_start,
       arg_min((discount > 0)::INT, cycle)                   AS discounted_at_start
FROM features
GROUP BY msno, relationship_no
HAVING arg_min(is_monthly, cycle) AND NOT arg_min(left_truncated, cycle)
   AND duration IS NOT NULL"""

HORIZONS = [1, 3, 6, 12]


def survival_table(df, segment: str):
    rows, fig, ax = [], *plt.subplots(figsize=(7, 4))
    for value, g in df.groupby(segment):
        kmf = KaplanMeierFitter(label=f"{segment}={value}").fit(g["duration"], g["event"])
        kmf.plot_survival_function(ax=ax)
        rows.append({segment: value, "relationships": len(g),
                     **{f"S({h})": round(float(kmf.predict(h)), 4) for h in HORIZONS},
                     "median_cycles": float(kmf.median_survival_time_)})
    ax.set_xlabel("renewal decisions survived (cycles)")
    ax.set_ylabel("share still subscribed")
    ax.set_xlim(0, 24)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"km_{segment}.png", dpi=150)
    a, b = (g for _, g in df.groupby(segment))
    p = logrank_test(a["duration"], b["duration"], a["event"], b["event"]).p_value
    return rows, p


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = connect(read_only=True).sql(RELATIONSHIPS).df()
    print(f"relationships: {len(df)}, churned: {df['event'].sum()}")
    overall = KaplanMeierFitter().fit(df["duration"], df["event"])
    first_cycle_churn = ((df["duration"] == 1) & (df["event"] == 1)).mean()
    print(f"S(1) = {float(overall.predict(1)):.4f}; 1 - cycle-1 churn share = {1 - first_cycle_churn:.4f}")
    for segment in ["auto_renew_at_start", "discounted_at_start"]:
        rows, p = survival_table(df, segment)
        for r in rows:
            print(r)
        print(f"log-rank p-value ({segment}): {p:.3g}")


if __name__ == "__main__":
    main()