"""Module 2: compare our labels with the official train_v2 labels.

Official scope (competition data description): a user is labelled for a month if the
expiry on their last transaction up to the month's last day falls inside that month.
"""
from renewal_risk.config import REPORTS_DIR
from renewal_risk.db import connect

MONTH_START, MONTH_END = "2017-02-01", "2017-02-28"

COMPARE = f"""
WITH ordered AS (
    SELECT msno, membership_expire_date,
           row_number() OVER (PARTITION BY msno
               ORDER BY transaction_date, is_cancel DESC, membership_expire_date) AS row_no
    FROM txn WHERE transaction_date <= DATE '{MONTH_END}'
),
official_scope AS (
    SELECT msno, arg_max(membership_expire_date, row_no) AS current_exp
    FROM ordered GROUP BY msno
),
ours AS (
    SELECT msno, effective_exp, churn
    FROM decisions WHERE start_date <= DATE '{MONTH_END}'
    QUALIFY row_number() OVER (PARTITION BY msno ORDER BY spell_no DESC) = 1
)
SELECT o.msno, o.is_churn AS official, u.churn AS ours, s.current_exp, u.effective_exp,
       s.current_exp BETWEEN DATE '{MONTH_START}' AND DATE '{MONTH_END}' AS in_month
FROM labels_official o
LEFT JOIN official_scope s USING (msno)
LEFT JOIN ours u USING (msno)"""


def main() -> None:
    con = connect(read_only=True)
    df = con.sql(COMPARE).df()
    matched = df[df["ours"].notna()]
    agree = (matched["ours"] == matched["official"]).mean()
    lines = [
        f"official labels: {len(df)}",
        f"share whose current expiry falls in {MONTH_START[:7]}: {df['in_month'].mean():.4f}",
        f"share with a non-censored label of ours: {len(matched) / len(df):.4f}",
        f"agreement on those: {agree:.4f}",
        (f"effective_exp equals official current expiry: "
         f"{(matched['effective_exp'] == matched['current_exp']).mean():.4f}"),
        "",
        "confusion (rows = official, columns = ours):",
        matched.groupby(["official", "ours"]).size().unstack(fill_value=0).to_string(),
    ]
    text = "\n".join(lines)
    print(text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "label_validation.txt").write_text(text)
    df[(df["ours"].notna()) & (df["ours"] != df["official"])].head(50).to_csv(
        REPORTS_DIR / "label_disagreements.csv", index=False)


if __name__ == "__main__":
    main()