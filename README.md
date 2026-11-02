<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/header-dark.svg">
  <img width="100%" alt="Renewal Risk — Churn risk per renewal decision from a 23M-row subscription payment log" src="assets/header-light.svg">
</picture>

# Renewal risk

Churn risk for a subscription business, scored one renewal at a time.

The input is a raw payment log. The output is a warehouse table, `churn_scores`, holding one
row per open subscription: the probability that the next renewal does not happen, survival
over the next six renewal cycles, and the three reasons that pushed the score up the most.

Built on the [KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)
data (WSDM Cup 2018) — 23M transactions and 6.8M members covering 2015-01-01 to 2017-03-31.

---

## The idea

A subscription does not "churn" on a date of its own choosing. It churns at a **renewal
decision**: the plan expires, and either money arrives or it does not. So the unit of
analysis here is one renewal decision, not one customer-month and not one customer.

From the transaction log the pipeline reconstructs, per customer:

- **spells** — a paid row opens a spell; cancel rows attach to the spell they follow and can
  move its expiry earlier. On the same date, cancels sort before payments (cancel the old
  plan, then buy the new one).
- **relationships** — consecutive spells, broken by a lapse of more than 30 days. A customer
  who leaves and comes back starts relationship 2 at cycle 1, so tenure does not lie.
- **decisions** — one row per spell, with the label resolved 30 days after the effective
  expiry: `churn = 1` if no paid transaction lands in that window, `0` if one does, `NULL`
  (censored) if the window has not closed before the data ends.

Every feature is computed **as of 7 days before the scheduled expiry** — the moment you would
actually score the account and still have time to act. That cut-off is enforced in SQL and
tested: [test_point_in_time.py](tests/test_point_in_time.py) rebuilds the feature table from a
truncated copy of the log and asserts that all 20 features come out unchanged.
It also asserts that a column known to leak (`n_cancels_in_spell`, which counts cancels after
the as-of date) *is* flagged, so the harness cannot pass by doing nothing.

22,077,552 decisions come out of 22.9M transactions. 90.9% are monthly plans; 10.7% are
censored. Churn on a monthly decision is 5.98% overall, and strongly front-loaded:

| renewal cycle | decisions | churn rate |
|---|---:|---:|
| 1 | 2,004,423 | 23.68% |
| 2–6 | 6,045,072 | 4.88% |
| 7+ | 9,895,231 | 3.06% |

### Is the label right?

The competition ships official labels for one month (2017-02). Ours are built from a different
definition — a renewal panel rather than a monthly snapshot — so they are a check, not a target.
On the 1.90% of officially labelled users whose decision our panel has already resolved, the two
agree **88.32%** of the time, and our `effective_exp` matches the official "current expiry"
for **96.84%** of users. The disagreements are written to
[reports/label_disagreements.csv](reports/label_disagreements.csv);
[test_label.py](tests/test_label.py) pins the organisers' two worked examples plus the
same-day-cancel, lapse and open-window cases.

---

## Quick start

```bash
uv sync                                   # Python 3.13; dependencies in pyproject.toml
# put transactions.csv, transactions_v2.csv, members_v3.csv, train_v2.csv in data/raw/
./scripts/run_all.sh
```

That rebuilds everything from the CSVs, then runs pytest and ruff. Budget ~16 minutes and
~9 GB of disk for the DuckDB file (`data/` is gitignored; code and reports are what get committed).

Without the real data, generate a 20k-user stand-in with the same schema and a known churn
process, then point the pipeline at it:

```bash
python scripts/make_synthetic.py          # writes data/synthetic/ (20k users, 204k rows)
RAW_DIR=data/synthetic DB_PATH=/tmp/syn.duckdb REPORTS_DIR=/tmp/syn_reports \
  MODELS_DIR=/tmp/syn_models SKIP_CHECKS=1 ./scripts/run_all.sh
```

Override all four paths. `RAW_DIR` alone would send a synthetic run straight through the real
DuckDB file, the committed reports and the saved models. All fourteen steps take ~17 s on the
synthetic set, against ~16 minutes on the real data.

Never report numbers from it — it exists so the code can be exercised end to end in seconds,
and so `test_point_in_time.py` has something to truncate.

### Re-running parts of it

```bash
FROM=model ./scripts/run_all.sh           # skip the SQL, reuse the tables already built
ONLY="evaluate horizon" ./scripts/run_all.sh
SKIP_CHECKS=1 ./scripts/run_all.sh        # no pytest/ruff
```

Paths come from environment variables (`RAW_DIR`, `DB_PATH`, `REPORTS_DIR`, `MODELS_DIR`);
everything else lives in [config.py](renewal_risk/config.py).

### A note on memory

This was developed on an 8 GB machine against a 23M-row log, and that constraint shaped the
code: DuckDB is capped (`SQL_FRACTION`/`ML_FRACTION` in the run script) so it spills to disk
instead of pushing the Python process into swap, the logistic design matrix is kept sparse,
categoricals are stored as levels rather than strings, and the landmark collapse happens in
SQL rather than pandas. The comments at those points explain the sizes involved. On a large
machine you can raise both fractions and ignore all of it.

---

## Pipeline

Fourteen steps, each a module you can run on its own (`python -m renewal_risk.<step>`).
Times are from a full run on an 8 GB M1 MacBook Pro.

| step | what it does | time |
|---|---|---:|
| `load` | raw CSVs into DuckDB with explicit types | 11 s |
| `audit` | profile the raw data before any rule is chosen → [reports/audit.md](reports/audit.md) | 12 s |
| `clean` | four cleaning rules ([01_clean.sql](sql/01_clean.sql)); removes 9,801 of 23M rows | 8 s |
| `decisions` | spells, relationships, cycles, label ([02_decisions.sql](sql/02_decisions.sql)) | 26 s |
| `validate_label` | our label vs. the official one | 5 s |
| `features` | point-in-time features ([03_features.sql](sql/03_features.sql)) | 57 s |
| `km` | Kaplan-Meier survival by segment | 6 s |
| `baseline` | segment churn rates + back-test — the thing to beat | 8 s |
| `model` | logistic and gradient-boosted hazard models | 169 s |
| `evaluate` | calibration first, ranking second; isotonic on the calib window | 69 s |
| `horizon` | landmark model, back-tested over 6 cycles against observed KM | 75 s |
| `explain` | grouped permutation importance + additive reason codes | 23 s |
| `score` | refit on everything resolved, score the open book, write `churn_scores` | 452 s |
| `monitor` | observed/expected per month and PSI drift; retrain triggers | 13 s |

---

## How it is evaluated

Out-of-time windows, with an embargo wide enough that a label cannot be known before the
next window starts. The rules are executable, not prose:
[`window_problems()`](renewal_risk/split.py) returns every violation and is asserted on every
load, so a careless edit to a date in `config.py` fails the run rather than quietly leaking.

| window | scheduled expiry in | used for |
|---|---|---|
| `train` | 2015-07-01 → 2016-02-29 | fitting (5,272,484 decisions, 8.06% churn) |
| `calib` | 2016-03-01 → 2016-04-30 | isotonic calibration only (1,593,633 decisions, 10.86% churn) |
| `test` | 2016-07-01 → 2016-09-30 | everything reported below (2,551,971 decisions, 1,000,890 customers, 3.755% churn) |
| `horizon_train` | 2015-07-01 → 2015-12-15 | landmark model; its 6-cycle outcomes resolve before the test window opens |

Calibration is the headline metric, not AUC. A ranking that is right about order but wrong
about level cannot be summed into an expected-churner count, and that count is what the
business actually spends money against.

---

## Results

Every figure below is from one full run (2026-09-23) and matches the files in
[reports/](reports/). A rebuild moves some of them in the last digit — see *Not
bit-reproducible* under Known limits.

### Next renewal, test window

| model | Brier | log loss | AUC | calib. slope | obs/exp | Brier gain vs. baseline (95% CI) |
|---|---:|---:|---:|---:|---:|---|
| baseline (segment rates) | 0.0396 | 0.1524 | 0.768 | 0.807 | 0.525 | — |
| logistic | 0.0309 | 0.1136 | 0.903 | 0.807 | 0.726 | 0.0087 (0.0085–0.0088) |
| **logistic + isotonic** (shipped) | 0.0329 | 0.1360 | 0.898 | 1.158 | 0.441 | 0.0067 (0.0066–0.0068) |
| boosting | 0.0274 | 0.1030 | 0.919 | 1.034 | 0.696 | 0.0122 (0.0121–0.0123) |
| boosting + isotonic | 0.0353 | 0.1335 | 0.907 | 1.086 | 0.410 | 0.0043 (0.0042–0.0045) |

The confidence interval resamples **customers**, not rows: the same account appears at several
renewals, and treating those as independent would make every interval look tighter than it is.

Two things in that table are worth saying out loud rather than burying:

**The shipped model is not the most accurate one.** Boosting wins on Brier by 0.0055 and on
AUC by 0.02. The pipeline ships logistic+isotonic because the reason codes are additive
log-odds contributions that are *asserted* to sum to the published score
([explain.py](renewal_risk/explain.py)) — an identity a linear model gives for free, where a
tree ensemble would need SHAP and far more compute over a million-row book. The choice was
fixed before this table was computed rather than picked from it, and the cost is 0.0055 Brier.

**Everything over-predicts churn on the test window.** Churn fell from 8.06% in the training
window to 3.76% in the test window, and no model here is told that. Isotonic fixes the
*slope* (0.81 → 1.16, i.e. the spread of the probabilities is right) but makes the *level*
worse (obs/exp 0.73 → 0.44), because it was fitted on a March–April 2016 window whose own
churn rate was 10.86% — higher than the training window, let alone the test window. Every
segment reads "conservative" in the back-test:

| segment | decisions | observed | predicted | obs/exp |
|---|---:|---:|---:|---:|
| auto=0, cycle 1 | 61,590 | 0.3685 | 0.4227 | 0.87 |
| auto=0, cycle 2–6 | 116,427 | 0.1859 | 0.2582 | 0.72 |
| auto=0, cycle 7+ | 103,637 | 0.0737 | 0.1401 | 0.53 |
| auto=1, cycle 1 | 203,413 | 0.0729 | 0.1923 | 0.38 |
| auto=1, cycle 2–6 | 439,043 | 0.0184 | 0.1054 | 0.17 |
| auto=1, cycle 7+ | 1,627,861 | 0.0129 | 0.0375 | 0.34 |

The monitor catches exactly this, which is the point of having one: observed/expected is
0.480, 0.423, 0.425 for July, August and September 2016 — outside the (0.9, 1.1) band three
months running, so `retrain = True`. PSI flags `tenure_days` (1.07) and `amount_paid` (0.29)
above the 0.25 limit, so `investigate = True`. Ranking is unaffected and stays strong
(AUC 0.90); it is the level that has moved, and a refit on recent data is the fix.

### Six cycles out

Predicting six renewals ahead by rolling the next-renewal model forward — advancing the clock
and freezing everything else — degrades badly, because it never lets the population thin out.
A landmark model (covariates frozen at the landmark, time measured in cycles since) holds up
against the observed Kaplan-Meier curve of the same test cohort:

| cycles ahead | observed (KM) | landmark model | shipped hybrid | near-term rolled | baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.942 | −1.5 pp | −6.0 pp | −6.0 pp | −6.8 pp |
| 2 | 0.914 | −1.8 pp | −6.7 pp | −9.8 pp | −7.7 pp |
| 3 | 0.889 | −1.6 pp | −6.5 pp | −12.2 pp | −8.5 pp |
| 4 | 0.868 | −1.0 pp | −5.9 pp | −14.1 pp | −9.4 pp |
| 5 | 0.850 | −0.4 pp | −5.1 pp | −15.2 pp | −10.4 pp |
| 6 | 0.823 | +1.0 pp | −3.7 pp | −14.6 pp | −10.1 pp |

The landmark model is within 1.9 pp everywhere. The *shipped* curve is the hybrid: it starts
from the published next-renewal probability and uses landmark hazards after that, so the two
published numbers can never contradict each other. That consistency costs 3.7–6.7 pp of six-cycle
accuracy, inherited from the level shift described above.

Kaplan-Meier on 1,486,401 relationships gives the shape of the problem before any model:
S(1) = 0.696, and the two splits that matter separate immediately —

| segment | S(1) | S(6) | S(12) | median |
|---|---:|---:|---:|---:|
| auto-renew off at start | 0.631 | 0.274 | 0.172 | 2 cycles |
| auto-renew on at start | 0.724 | 0.614 | 0.539 | 16 cycles |
| no discount at start | 0.809 | 0.613 | 0.516 | 13 cycles |
| discounted at start | 0.209 | 0.094 | 0.074 | 1 cycle |

Log-rank p ≈ 0 for both. Discounted sign-ups are a churn machine: most do not survive the
first renewal.

### What drives it

Permutation importance on the Brier score, with correlated features permuted **together** —
shuffling `tenure_days` alone while `cycle_capped` and `paid_to_date` still carry the same
information understates it badly.

| feature / group | Brier increase | meaning |
|---|---:|---|
| tenure (cycle, payments, days) | 0.00874 | time as a customer |
| `cancels_in_spell_to_asof` | 0.00670 | cancelled during the current period |
| `is_auto_renew` | 0.00176 | auto-renew setting |
| `discount` | 0.00121 | discount on the current plan |
| `list_price` | 0.00102 | list price |
| `price_change` | 0.00092 | price change at this renewal |

Per account, the top three positive log-odds contributions are turned into plain English
("cancelled during the current period; time as a customer; discount on the current plan") and
shipped with the score. [reports/reason_codes_top10.csv](reports/reason_codes_top10.csv) has
the highest-risk accounts in the test window.

---

## What ships

`score` refits on every outcome that has resolved (train through 2016-12-31, isotonic on
2017-01→02, landmark model through 2016-09-30), scores the open book — each customer's latest
monthly decision whose window has not closed — and writes one idempotent, versioned table:

```sql
CREATE TABLE churn_scores (
    msno                 VARCHAR   NOT NULL,
    as_of                DATE      NOT NULL,
    scheduled_exp        DATE      NOT NULL,
    p_churn_next_renewal DOUBLE    NOT NULL,   -- clipped to [0.001, 0.999]
    survival_3_cycles    DOUBLE    NOT NULL,
    survival_6_cycles    DOUBLE    NOT NULL,
    top_reasons          VARCHAR,
    model_version        VARCHAR   NOT NULL,
    scored_at            TIMESTAMP NOT NULL,
    PRIMARY KEY (msno, as_of, model_version)
)
```

A re-run deletes and reinserts its own `model_version` inside one transaction, so the step is
safe to repeat and two versions can sit side by side. Four contract assertions run *before*
anything is written: one row per customer, no nulls, probabilities inside the clip, and
six-cycle survival never above three-cycle.

Last run: **1,039,338 open subscriptions**, mean p = 0.0382, **39,661 expected churners**,
mean six-cycle survival 0.923, `model_version = logistic-2016-12-31_iso-2017-02-28_landmark-h6`.
Given the calibration gap above, treat that count as an upper bound until a refit lands.

---

## Known limits

- **Level shift, unhandled.** Churn falls from 8.06% to 3.76% between the training and test
  windows; the scores inherit it, the monitor flags it, nothing auto-corrects it. A recency-weighted refit
  or a calibration window adjacent to the scoring window is the obvious next move.
- **Demographics are mostly missing.** 57% of monthly decisions have no usable age and 57% no
  gender. Daily listening logs are not used at all: the model sees payments, not engagement.
- **`max(as_of)` in the scored book is 2028-11-29.** 76,267 source transactions carry expiry
  dates past 2017 (the furthest is 2036-10-15). They are real rows, not a pipeline bug, but
  they should be quarantined rather than scored.
- **Not bit-reproducible.** Two runs over the identical CSVs disagree slightly: 2,004,423 vs.
  2,004,422 cycle-1 decisions, 1,486,401 vs. 1,486,400 relationships, test AUC 0.9028 vs.
  0.9016. 20,054 transactions (10,019 pairs) tie on the full sort key of the spell window in
  [02_decisions.sql](sql/02_decisions.sql) — same customer, date, `is_cancel` and expiry — and
  every one of those pairs differs on plan, price or payment method, so `row_number()` splits
  them in whatever order the scan happens to produce. Adding a deterministic tiebreak to that
  window would close it.
- **Accuracy left on the table.** Boosting is better on every ranking and accuracy metric; the
  additive-reason-code requirement is what keeps it out of production.
- Censoring is handled for the survival curves (Kaplan-Meier) but decisions with an open
  window are simply excluded from training, which is the conservative choice, not the
  efficient one.

---

## Layout

```
renewal_risk/      one module per pipeline step; config.py holds every path and date
sql/               01_clean · 02_decisions · 03_features — the panel and the label live here
tests/             label logic, point-in-time leakage, window embargo (31 tests)
scripts/           run_all.sh (the whole pipeline) · make_synthetic.py (stand-in data)
reports/           every table and figure the run produces; committed so results are reviewable
data/              raw CSVs, the DuckDB file, saved models — gitignored
```

Tests run in ~1.5 s and need no data except the synthetic set (the point-in-time test skips
without it). Lint is `ruff check renewal_risk tests scripts`; both run at the end of
`run_all.sh`.
