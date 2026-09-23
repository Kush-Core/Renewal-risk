-- Cumulative history per user per transaction day.
CREATE OR REPLACE TABLE user_days AS
SELECT msno, txn_day,
       (sum(n_paid)   OVER w)::INT AS paid_to_date,
       (sum(n_cancel) OVER w)::INT AS cancels_to_date,
       min(txn_day)  OVER w AS first_txn_day
FROM (
    SELECT msno, transaction_date AS txn_day,
           sum(1 - is_cancel) AS n_paid, sum(is_cancel) AS n_cancel
    FROM txn GROUP BY ALL
)
WINDOW w AS (PARTITION BY msno ORDER BY txn_day ROWS UNBOUNDED PRECEDING);

CREATE OR REPLACE TABLE features AS
WITH d AS (
    SELECT *,
           amount_paid - lag(amount_paid) OVER (PARTITION BY msno ORDER BY spell_no) AS price_change,
           least(cycle, 12) AS cycle_capped
    FROM decisions
),
at_asof AS (            -- latest history row dated on or before the as-of date
    SELECT d.*, u.paid_to_date, u.cancels_to_date, u.first_txn_day
    FROM d ASOF LEFT JOIN user_days u
      ON d.msno = u.msno AND d.as_of >= u.txn_day
),
before_start AS (       -- latest history row dated strictly before the spell started
    SELECT a.*, coalesce(u.cancels_to_date, 0) AS cancels_before_start
    FROM at_asof a ASOF LEFT JOIN user_days u
      ON a.msno = u.msno AND a.start_date > u.txn_day
)
SELECT b.*,
       b.list_price - b.amount_paid               AS discount,
       b.cancels_to_date - b.cancels_before_start AS cancels_in_spell_to_asof,
       b.as_of - b.first_txn_day                  AS tenure_days,
       b.scheduled_exp - b.as_of                  AS days_to_expiry,
       m.city, m.registered_via, m.age, m.gender,
       CASE WHEN m.registration_init_time <= b.as_of
            THEN b.as_of - m.registration_init_time END AS registration_days
FROM before_start b
LEFT JOIN members m USING (msno)
-- The ASOF joins above do not preserve row order, so without this the table is
-- laid out differently on every build and anything that picks rows by position
-- (sampling, train/validation splits) silently changes. (msno, spell_no) is
-- unique, so this is a total order; leading with scheduled_exp also lets the
-- window filters in split.load prune row groups.
ORDER BY b.scheduled_exp, b.msno, b.spell_no;