"""Write a small synthetic dataset with the KKBox schema to data/synthetic/.

Purpose: run every module end to end in minutes before touching the real data.
The churn process is known, so the pipeline has signal to find. It is NOT a
substitute for the real data: never report numbers from it.
"""
import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(7)
OUT = Path("data/synthetic")
OUT.mkdir(parents=True, exist_ok=True)
START, V1_END, V2_END = date(2015, 1, 1), date(2017, 2, 28), date(2017, 3, 31)
N_USERS = 20_000


def ymd(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


rows, members = [], []
for i in range(N_USERS):
    msno = f"u{i:06d}"
    reg = START + timedelta(days=random.randint(-400, 700))
    members.append([msno, random.randint(1, 22),
                    random.choice([0, 0, random.randint(15, 60), -7000]),
                    random.choice(["", "male", "female"]),
                    random.choice([3, 4, 7, 9]), ymd(reg)])
    t = max(reg, START) + timedelta(days=random.randint(0, 120))
    auto = random.random() < 0.75
    discount = random.random() < 0.15
    cycle, lapsed = 0, 0
    while t <= V2_END:
        plan = 30 if random.random() < 0.93 else 180
        list_price = 149 if plan == 30 else 894
        paid = 99 if (discount and plan == 30) else list_price
        exp = t + timedelta(days=plan)
        rows.append([msno, 41 if auto else 38, plan, list_price, paid,
                     int(auto), ymd(t), ymd(exp), 0])
        if random.random() < 0.005:                      # exact duplicate row
            rows.append(list(rows[-1]))
        z = -3.2 + 1.6 * (not auto) + 0.9 * discount - 0.04 * min(cycle, 24) + 0.8 * lapsed
        churn = random.random() < sigmoid(z)
        if random.random() < (0.45 if churn else 0.03):  # active cancel moves expiry back
            tc = t + timedelta(days=random.randint(1, max(1, plan - 2)))
            exp = tc + timedelta(days=1)
            rows.append([msno, 41 if auto else 38, plan, list_price, 0,
                         int(auto), ymd(tc), ymd(exp), 1])
            if not churn:                                # plan change: rebuy soon after
                t = tc + timedelta(days=random.randint(1, 15))
                auto = random.random() < 0.6
                cycle += 1
                continue
        if churn:
            if random.random() < 0.3:                    # returns after a lapse
                t = exp + timedelta(days=random.randint(31, 200))
                cycle, lapsed = 0, 1
                continue
            break
        t = exp - timedelta(days=random.randint(0, 2)) if auto else exp + timedelta(days=random.randint(-3, 25))
        cycle += 1

header = ["msno", "payment_method_id", "payment_plan_days", "plan_list_price",
          "actual_amount_paid", "is_auto_renew", "transaction_date",
          "membership_expire_date", "is_cancel"]
v1 = [r for r in rows if r[6] <= ymd(V1_END)]
v2 = [r for r in rows if ymd(V1_END) < r[6] <= ymd(V2_END)]
for name, data in [("transactions.csv", v1), ("transactions_v2.csv", v2)]:
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)
with open(OUT / "members_v3.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["msno", "city", "bd", "gender", "registered_via", "registration_init_time"])
    w.writerows(members)

# Reference labels built the way the official description defines them:
# current expiry = expiry on the last transaction dated <= 2017-02-28;
# in scope if it falls in Feb 2017; churn if no paid transaction by expiry + 30 days.
by_user = {}
for r in rows:
    by_user.setdefault(r[0], []).append(r)
labels = []
for msno, rs in by_user.items():
    rs.sort(key=lambda r: (r[6], -r[8], r[7]))
    upto = [r for r in rs if r[6] <= ymd(V1_END)]
    if not upto:
        continue
    cur = upto[-1][7]
    if not (20170201 <= cur <= 20170228):
        continue
    cur_d = date(cur // 10000, cur // 100 % 100, cur % 100)
    renewed = any(r[8] == 0 and ymd(V1_END) < r[6] <= ymd(cur_d + timedelta(days=30)) for r in rs)
    labels.append([msno, int(not renewed)])
with open(OUT / "train_v2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["msno", "is_churn"])
    w.writerows(labels)
print(f"transactions={len(v1)} transactions_v2={len(v2)} members={len(members)} labels={len(labels)}")