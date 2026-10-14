"""
Step 3 - ETL: H&M CSV files -> Oracle star schema.

Oracle Database:
    Host: localhost
    Port: 1521
    Service: XEPDB1
    User: HM_DW
"""

import os
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("HM_DATA_DIR", "data"))

SAMPLE_CUSTOMERS = int(
    os.environ.get("HM_SAMPLE_CUSTOMERS", "50000")
)

CHUNK = 2_000_000


# Oracle connection
# Using URL.create avoids problems with special characters in password.
DB_URL = URL.create(
    drivername="oracle+oracledb",
    username=os.environ["HM_DB_USER"],
    password=os.environ["HM_DB_PASSWORD"],
    host=os.environ["HM_DB_HOST"],
    port=int(os.environ["HM_DB_PORT"]),
    query={"service_name": os.environ["HM_DB_SERVICE"]},
)


# ------------------------------------------------------------------
# EXTRACT
# ------------------------------------------------------------------

print("Extract: reading customers and articles...")

customers = pd.read_csv(
    DATA_DIR / "customers.csv"
)

articles = pd.read_csv(
    DATA_DIR / "articles.csv"
)

# Select a reproducible customer sample to keep the warehouse manageable.
sample_ids = customers["customer_id"].drop_duplicates().sample(
    n=min(
        SAMPLE_CUSTOMERS,
        customers["customer_id"].nunique()
    ),
    random_state=42,
)

sample_set = set(sample_ids)

print(
    f"Extract: reading transactions in chunks "
    f"(keeping {len(sample_set):,} sampled customers)..."
)

parts = []

for chunk in pd.read_csv(
    DATA_DIR / "transactions_train.csv",
    chunksize=CHUNK,
    dtype={
        "article_id": "int64",
        "sales_channel_id": "int8",
    },
    parse_dates=["t_dat"],
):
    parts.append(
        chunk[
            chunk["customer_id"].isin(sample_set)
        ]
    )

tx = pd.concat(
    parts,
    ignore_index=True
)

print(
    f"  {len(tx):,} transaction rows extracted"
)


# ------------------------------------------------------------------
# TRANSFORM
# ------------------------------------------------------------------

print("Transform: building dimensions...")


# ------------------------------------------------------------------
# DIM_CUSTOMER
# ------------------------------------------------------------------

c = (
    customers[
        customers["customer_id"].isin(sample_set)
    ]
    .drop_duplicates("customer_id")
    .copy()
)

# Fill missing age using median.
c["age"] = (
    c["age"]
    .fillna(c["age"].median())
    .round()
    .astype(int)
)

# Create age groups.
c["age_group"] = pd.cut(
    c["age"],
    bins=[0, 25, 35, 45, 55, 200],
    labels=[
        "16-25",
        "26-35",
        "36-45",
        "46-55",
        "56+",
    ],
).astype(str)

# Handle missing categorical values.
c["club_member_status"] = (
    c["club_member_status"]
    .fillna("UNKNOWN")
)

c["fashion_news_frequency"] = (
    c["fashion_news_frequency"]
    .replace({"None": "NONE"})
    .fillna("NONE")
)

# Convert Active column to numeric flag.
c["is_active"] = (
    c["Active"]
    .fillna(0)
    .astype(int)
)

dim_customer = c[
    [
        "customer_id",
        "age",
        "age_group",
        "club_member_status",
        "fashion_news_frequency",
        "is_active",
    ]
].reset_index(drop=True)

# Surrogate key.
dim_customer.insert(
    0,
    "customer_key",
    dim_customer.index + 1
)


# ------------------------------------------------------------------
# DIM_PRODUCT
# ------------------------------------------------------------------

prod_cols = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "department_name",
    "index_group_name",
    "colour_group_name",
]

dim_product = (
    articles[prod_cols]
    .drop_duplicates("article_id")
    .reset_index(drop=True)
)

# Surrogate key.
dim_product.insert(
    0,
    "product_key",
    dim_product.index + 1
)


# ------------------------------------------------------------------
# DIM_TIME
# ------------------------------------------------------------------

dates = pd.Series(
    tx["t_dat"]
    .drop_duplicates()
    .sort_values()
    .values,
    name="full_date",
)

dim_time = pd.DataFrame({
    "full_date": dates
})

dim_time["time_key"] = (
    dim_time["full_date"]
    .dt.strftime("%Y%m%d")
    .astype(int)
)

dim_time["day"] = (
    dim_time["full_date"].dt.day
)

dim_time["month"] = (
    dim_time["full_date"].dt.month
)

dim_time["month_name"] = (
    dim_time["full_date"].dt.month_name()
)

dim_time["quarter"] = (
    dim_time["full_date"].dt.quarter
)

dim_time["year"] = (
    dim_time["full_date"].dt.year
)

dim_time["day_name"] = (
    dim_time["full_date"].dt.day_name()
)

dim_time["full_date"] = (
    dim_time["full_date"].dt.date
)

dim_time = dim_time[
    [
        "time_key",
        "full_date",
        "day",
        "month",
        "month_name",
        "quarter",
        "year",
        "day_name",
    ]
]


# ------------------------------------------------------------------
# DIM_CHANNEL
# ------------------------------------------------------------------

dim_channel = pd.DataFrame({
    "channel_key": [1, 2],
    "channel_id": [1, 2],
    "channel_name": [
        "Channel 1",
        "Channel 2",
    ],
})


# ------------------------------------------------------------------
# FACT_SALES
# ------------------------------------------------------------------

print("Transform: building the fact table...")

grain = [
    "t_dat",
    "customer_id",
    "article_id",
    "price",
    "sales_channel_id",
]

# Repeated transaction rows become quantity.
f = (
    tx.groupby(
        grain,
        as_index=False
    )
    .size()
    .rename(columns={"size": "quantity"})
)

# Calculate total sales.
f["sales_amount"] = (
    f["price"] * f["quantity"]
).round(6)

# Foreign/surrogate keys.
f["time_key"] = (
    f["t_dat"]
    .dt.strftime("%Y%m%d")
    .astype(int)
)

customer_key_map = dict(
    zip(
        dim_customer["customer_id"],
        dim_customer["customer_key"],
    )
)

product_key_map = dict(
    zip(
        dim_product["article_id"],
        dim_product["product_key"],
    )
)

f["customer_key"] = (
    f["customer_id"]
    .map(customer_key_map)
)

f["product_key"] = (
    f["article_id"]
    .map(product_key_map)
)

f["channel_key"] = (
    f["sales_channel_id"]
    .astype(int)
)

# Remove rows with unmatched dimension keys.
before = len(f)

f = f.dropna(
    subset=[
        "customer_key",
        "product_key",
    ]
)

print(
    f"  dropped {before - len(f):,} rows "
    f"with no matching customer/article"
)

f = f.rename(
    columns={
        "price": "unit_price"
    }
)

f = f[
    [
        "time_key",
        "customer_key",
        "product_key",
        "channel_key",
        "unit_price",
        "quantity",
        "sales_amount",
    ]
].astype({
    "time_key": int,
    "customer_key": int,
    "product_key": int,
    "channel_key": int,
    "quantity": int,
})

f["unit_price"] = (
    f["unit_price"].round(6)
)

f["sales_amount"] = (
    f["sales_amount"].round(6)
)

# Surrogate sales key.
f.insert(
    0,
    "sales_key",
    range(1, len(f) + 1)
)


# ------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------

print("Load: connecting to Oracle...")

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
)

print("Load: inserting dimensions...")

# Tables were already created using 02_create_warehouse.sql.
# Therefore, ETL only inserts the data.

dim_time.to_sql(
    "dim_time",
    engine,
    if_exists="append",
    index=False,
    chunksize=5000,
)

dim_customer.to_sql(
    "dim_customer",
    engine,
    if_exists="append",
    index=False,
    chunksize=5000,
)

dim_product.to_sql(
    "dim_product",
    engine,
    if_exists="append",
    index=False,
    chunksize=5000,
)

dim_channel.to_sql(
    "dim_channel",
    engine,
    if_exists="append",
    index=False,
    chunksize=2,
)

print("Load: inserting fact_sales...")

f.to_sql(
    "fact_sales",
    engine,
    if_exists="append",
    index=False,
    chunksize=5000,
)

# ------------------------------------------------------------------
# VERIFY
# ------------------------------------------------------------------

print("\nRow counts in the warehouse:")

with engine.connect() as conn:
    for table_name in [
        "dim_time",
        "dim_customer",
        "dim_product",
        "dim_channel",
        "fact_sales",
    ]:
        count = conn.execute(
            text(
                f"SELECT COUNT(*) FROM {table_name}"
            )
        ).scalar()

        print(
            f"  {table_name:14s} {count:,}"
        )

print("Done.")