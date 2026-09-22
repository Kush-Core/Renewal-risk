-- Module 1 cleaning rules. Record each rule and its row count in DECISIONS.md.

-- Rule 1: drop exact duplicate rows (the same row can appear twice, or in both files).
-- Rule 2: drop paid rows whose membership expires before the transaction date.
CREATE OR REPLACE TABLE txn AS
SELECT DISTINCT * EXCLUDE (filename)
FROM txn_raw
WHERE NOT (is_cancel = 0 AND membership_expire_date < transaction_date);

-- Rule 3: implausible ages become NULL; blank gender becomes NULL.
-- Rule 4: one row per member (keep the earliest registration if msno repeats).
CREATE OR REPLACE TABLE members AS
SELECT msno, city, registered_via, registration_init_time,
       CASE WHEN bd BETWEEN 10 AND 80 THEN bd END AS age,
       NULLIF(gender, '') AS gender
FROM members_raw
QUALIFY row_number() OVER (PARTITION BY msno ORDER BY registration_init_time) = 1;