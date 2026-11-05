"""Create a small Amazon-like synthetic dataset (reviews + item metadata) for tests."""

import json
import os
import sys

import numpy as np
import pandas as pd

POS = ["Great product, works perfectly.", "I love it, excellent quality.", "Very happy, would buy again.",
       "Fantastic value and fast to set up.", "Exactly what I needed, highly recommended."]
NEG = ["Terrible, broke after two days.", "Very poor quality, do not buy.", "Disappointed, it does not work.",
       "Cheap material and awful finish.", "Returned it, complete waste of money."]
MID = ["It is okay for the price.", "Average, nothing special.", "Does the job but could be better."]


def main(out_dir, n_users=300, n_items=120, seed=0):
    rng = np.random.default_rng(seed)
    cats = ["Beauty", "Electronics", "Books", "Sports"]
    item_cat = {f"I{j:04d}": cats[j % len(cats)] for j in range(n_items)}
    pop = rng.zipf(1.6, n_items).astype(float)
    pop /= pop.sum()
    rows = []
    t0 = int(pd.Timestamp("2019-01-01").timestamp())
    t1 = int(pd.Timestamp("2022-12-31").timestamp())
    for u in range(n_users):
        n = int(np.clip(rng.lognormal(2.0, 0.7), 3, 60))
        fav = cats[u % len(cats)]
        w = np.array([pop[j] * (4.0 if item_cat[f"I{j:04d}"] == fav else 1.0) for j in range(n_items)])
        items = rng.choice(n_items, size=min(n, n_items), replace=False, p=w / w.sum())
        bias = rng.normal(0.5, 0.8)
        for j in items:
            r = int(np.clip(np.rint(3.5 + bias + rng.normal(0, 1)), 1, 5))
            text = rng.choice(POS if r >= 4 else NEG if r <= 2 else MID)
            rows.append({"user_id": f"U{u:04d}", "parent_asin": f"I{j:04d}", "rating": float(r),
                         "text": f"{text} ({item_cat[f'I{j:04d}']})", "title": "t",
                         "timestamp": int(rng.integers(t0, t1)) * 1000})   # milliseconds, as Amazon 2023
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "Synthetic.csv"), sep="\t", index=False)
    with open(os.path.join(out_dir, "meta_Synthetic.jsonl"), "w") as fh:
        for i, c in item_cat.items():
            fh.write(json.dumps({"parent_asin": i, "main_category": c, "title": f"{c} item {i}"}) + "\n")
    print(f"{len(df)} interactions written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/input/synthetic")
