-- Module 2: one row per renewal decision, with the 30-day churn label.
-- A spell starts at a paid row; cancel rows join the spell they follow.
-- On the same date, cancels sort before payments (cancel old plan, then buy new).

CREATE OR REPLACE TABLE spells AS
WITH ordered AS (
    SELECT *,
           row_number() OVER w AS row_no,
           (sum(1 - is_cancel) OVER (w ROWS UNBOUNDED PRECEDING))::INT AS spell_no
    FROM txn
    WINDOW w AS (PARTITION BY msno
                 ORDER BY transaction_date, is_cancel DESC, membership_expire_date)
)
SELECT msno,
       spell_no,
       arg_min(transaction_date, row_no)       AS start_date,     -- the paid row opens the spell
       arg_min(membership_expire_date, row_no) AS scheduled_exp,  -- known at purchase
       arg_max(membership_expire_date, row_no) AS effective_exp,  -- after any cancels
       arg_min(payment_plan_days, row_no)      AS plan_days,
       arg_min(plan_list_price, row_no)        AS list_price,
       arg_min(actual_amount_paid, row_no)     AS amount_paid,
       arg_min(is_auto_renew, row_no)          AS is_auto_renew,
       arg_min(payment_method_id, row_no)      AS payment_method_id,
       sum(is_cancel)::INT                     AS n_cancels_in_spell
FROM ordered
WHERE spell_no > 0              -- cancels before a user's first payment have no spell
GROUP BY msno, spell_no;

CREATE OR REPLACE TABLE decisions AS
WITH linked AS (
    SELECT *,
           lead(start_date)   OVER (PARTITION BY msno ORDER BY spell_no) AS next_start_date,
           lag(effective_exp) OVER (PARTITION BY msno ORDER BY spell_no) AS prev_effective_exp
    FROM spells
),
flagged AS (
    SELECT *,
           -- a new relationship starts at the first spell or after a lapse of more than 30 days
           (prev_effective_exp IS NULL
            OR start_date - prev_effective_exp > 30)::INT AS starts_relationship
    FROM linked
),
numbered AS (
    SELECT *,
           (sum(starts_relationship) OVER (PARTITION BY msno ORDER BY spell_no
                                           ROWS UNBOUNDED PRECEDING))::INT AS relationship_no
    FROM flagged
),
bounds AS (
    SELECT min(transaction_date) AS data_start, max(transaction_date) AS data_end FROM txn
)
SELECT n.msno, n.spell_no, n.relationship_no,
       row_number() OVER (PARTITION BY n.msno, n.relationship_no ORDER BY n.spell_no) AS cycle,
       n.start_date, n.scheduled_exp, n.effective_exp,
       greatest(n.start_date, n.scheduled_exp - 7) AS as_of,        -- score 7 days before expiry
       n.plan_days, n.list_price, n.amount_paid, n.is_auto_renew, n.payment_method_id,
       n.n_cancels_in_spell, n.prev_effective_exp,
       n.start_date - n.prev_effective_exp AS gap_days,
       n.plan_days IN (30, 31) AS is_monthly,
       -- the first relationship may have begun before the data did
       (n.relationship_no = 1 AND n.start_date < b.data_start + 31) AS left_truncated,
       n.next_start_date,
       n.effective_exp + 30 <= b.data_end AS window_closed,
       CASE WHEN n.effective_exp + 30 > b.data_end THEN NULL          -- censored
            WHEN n.next_start_date <= n.effective_exp + 30 THEN 0     -- renewed in time
            ELSE 1 END AS churn
FROM numbered n CROSS JOIN bounds b;