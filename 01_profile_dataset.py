"""
Step 1 - Profile the H&M dataset (numbers for the 'Dataset Description' slide).

Run:  python 01_profile_dataset.py
Set the folder holding the 3 CSVs with the HM_DATA_DIR environment variable
(default: a folder called 'data' next to this script).
"""
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("HM_DATA_DIR", "data"))
CHUNK = 2_000_000

# ---------- customers & articles (small files) ----------
customers = pd.read_csv(DATA_DIR / "customers.csv")
articles = pd.read_csv(DATA_DIR / "articles.csv")

for name, df in [("customers.csv", customers), ("articles.csv", articles)]:
    print(f"\n=== {name}: {len(df):,} rows x {df.shape[1]} columns ===")
    miss = df.isna().sum()
    print("Columns with missing values:")
    print(miss[miss > 0].to_string() if (miss > 0).any() else "  none")
    print("Duplicate rows:", df.duplicated().sum())

# ---------- transactions (large file, read in chunks) ----------
rows = 0
min_date, max_date = None, None
missing = 0
repeated_rows = 0          # identical rows inside a chunk = same item bought more than once
price_min, price_max = float("inf"), float("-inf")
cust_seen, art_seen = set(), set()
channels = set()

for chunk in pd.read_csv(DATA_DIR / "transactions_train.csv", chunksize=CHUNK,
                         dtype={"article_id": "int64", "sales_channel_id": "int8"},
                         parse_dates=["t_dat"]):
    rows += len(chunk)
    cmin, cmax = chunk["t_dat"].min(), chunk["t_dat"].max()
    min_date = cmin if min_date is None else min(min_date, cmin)
    max_date = cmax if max_date is None else max(max_date, cmax)
    missing += int(chunk.isna().sum().sum())
    repeated_rows += int(chunk.duplicated().sum())
    price_min = min(price_min, chunk["price"].min())
    price_max = max(price_max, chunk["price"].max())
    cust_seen.update(chunk["customer_id"].unique())
    art_seen.update(chunk["article_id"].unique())
    channels.update(int(x) for x in chunk["sales_channel_id"].unique())
    print(f"  processed {rows:,} transaction rows...", end="\r")

print(f"\n\n=== transactions_train.csv: {rows:,} rows x 5 columns ===")
print("Time period      :", min_date.date(), "to", max_date.date())
print("Missing values   :", missing)
print("Repeated rows    :", repeated_rows, "(same customer/article/date/price/channel - treated as quantity > 1)")
print("Price range      :", price_min, "to", price_max, "(prices are anonymised/scaled, not real currency)")
print("Distinct customers who bought:", len(cust_seen))
print("Distinct articles sold       :", len(art_seen))
print("Sales channel ids:", sorted(channels))
