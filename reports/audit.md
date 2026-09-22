# Data audit

## rows and date range per file

| file                |     rows | first_txn           | last_txn            |
|:--------------------|---------:|:--------------------|:--------------------|
| transactions.csv    | 21547746 | 2015-01-01 00:00:00 | 2017-02-28 00:00:00 |
| transactions_v2.csv |  1431009 | 2015-01-01 00:00:00 | 2017-03-31 00:00:00 |

## users

|   users_in_txn |   of_which_in_members |
|---------------:|----------------------:|
|        2426143 |               1988333 |

## exact duplicate rows (ignoring source file)

|   duplicate_rows |
|-----------------:|
|             3339 |

## flags

|   cancel_share |   auto_renew_share |   paid_expiry_before_txn |   paid_zero_plan_days |   paid_zero_amount |
|---------------:|-------------------:|-------------------------:|----------------------:|-------------------:|
|         0.0388 |             0.8478 |                     6462 |                854136 |             971561 |

## plan length, paid rows (top 10)

|   payment_plan_days |             rows |   share |
|--------------------:|-----------------:|--------:|
|                  30 |      1.93336e+07 |  0.8753 |
|                   0 | 854136           |  0.0387 |
|                  31 | 733485           |  0.0332 |
|                   7 | 589807           |  0.0267 |
|                 410 | 162236           |  0.0073 |
|                 195 | 138801           |  0.0063 |
|                 180 |  76172           |  0.0034 |
|                  10 |  38632           |  0.0017 |
|                  90 |  31440           |  0.0014 |
|                 100 |  28252           |  0.0013 |

## users mixing monthly and other plan lengths

|   share_of_users |
|-----------------:|
|           0.2768 |

## members: age (bd) outside 10-80, blank gender

|   bd_out_of_range |   gender_missing |   duplicate_msno |     members |
|------------------:|-----------------:|-----------------:|------------:|
|       4.55056e+06 |       4.4295e+06 |                0 | 6.76947e+06 |
