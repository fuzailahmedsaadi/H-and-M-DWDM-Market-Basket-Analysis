"""Review 2 - Step 1: Prepare Market Basket Analysis data from Oracle.

Basket definition: one customer + transaction date + sales channel.
The H&M transaction dataset has no explicit order/basket ID, so this is a
proxy basket definition and should be described as such in the report.

Run from the project root:
    python 05_prepare_baskets.py

Outputs:
    outputs/basket_items.csv
    outputs/basket_summary.csv
"""

import os
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_URL = URL.create(
    drivername="oracle+oracledb",
    username=os.environ["HM_DB_USER"],
    password=os.environ["HM_DB_PASSWORD"],
    host=os.environ["HM_DB_HOST"],
    port=int(os.environ["HM_DB_PORT"]),
    query={"service_name": os.environ["HM_DB_SERVICE"]},
)

ITEMS_SQL = text("""
    SELECT
        f.customer_key          AS customer_key,
        t.time_key              AS time_key,
        t.full_date             AS full_date,
        f.channel_key           AS channel_key,
        p.product_key           AS product_key,
        p.article_id            AS article_id,
        p.prod_name             AS prod_name,
        p.product_type_name     AS product_type_name,
        p.product_group_name    AS product_group_name,
        SUM(f.quantity)         AS quantity,
        SUM(f.sales_amount)     AS item_sales
    FROM fact_sales f
    JOIN dim_time t
        ON f.time_key = t.time_key
    JOIN dim_product p
        ON f.product_key = p.product_key
    GROUP BY
        f.customer_key,
        t.time_key,
        t.full_date,
        f.channel_key,
        p.product_key,
        p.article_id,
        p.prod_name,
        p.product_type_name,
        p.product_group_name
    ORDER BY
        f.customer_key,
        t.time_key,
        f.channel_key,
        p.product_key
""")


def main() -> None:
    print("Review 2 - Step 1: Preparing transaction baskets from Oracle...")
    print("Connecting to Oracle HM_DW / XEPDB1...")

    engine = create_engine(DB_URL, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            db_user = conn.execute(text("SELECT USER FROM DUAL")).scalar_one()
            fact_rows = conn.execute(
                text("SELECT COUNT(*) FROM fact_sales")
            ).scalar_one()

        if str(db_user).upper() != "HM_DW":
            raise RuntimeError(
                f"Connected as {db_user}, expected HM_DW. Check HM_DB_USER."
            )
        if int(fact_rows) == 0:
            raise RuntimeError(
                "FACT_SALES is empty. Run and verify the Review 1 ETL first."
            )

        print(f"Connected as: {db_user}")
        print(f"FACT_SALES rows available: {int(fact_rows):,}")
        print("Reading and aggregating products by customer/date/channel...")

        items = pd.read_sql_query(ITEMS_SQL, engine)
        if items.empty:
            raise RuntimeError("The basket-item query returned no rows.")

        # Oracle may return column names in uppercase; normalize for Python.
        items.columns = [str(col).lower() for col in items.columns]

        # Keep the original H&M 10-character article identifier format in CSV.
        items["article_id"] = (
            pd.to_numeric(items["article_id"], errors="raise")
            .astype("int64")
            .astype(str)
            .str.zfill(10)
        )
        items["customer_key"] = pd.to_numeric(items["customer_key"]).astype("int64")
        items["time_key"] = pd.to_numeric(items["time_key"]).astype("int64")
        items["channel_key"] = pd.to_numeric(items["channel_key"]).astype("int64")
        items["product_key"] = pd.to_numeric(items["product_key"]).astype("int64")
        items["quantity"] = pd.to_numeric(items["quantity"]).astype("int64")
        items["item_sales"] = pd.to_numeric(items["item_sales"]).astype(float)
        items["full_date"] = pd.to_datetime(items["full_date"]).dt.strftime("%Y-%m-%d")

        # Basket key is customer + date + channel. Product rows are already
        # aggregated to one row per product within each basket.
        items["basket_id"] = (
            items["customer_key"].astype(str)
            + "_"
            + items["time_key"].astype(str)
            + "_"
            + items["channel_key"].astype(str)
        )

        item_columns = [
            "basket_id",
            "customer_key",
            "full_date",
            "time_key",
            "channel_key",
            "product_key",
            "article_id",
            "prod_name",
            "product_type_name",
            "product_group_name",
            "quantity",
            "item_sales",
        ]
        items = items[item_columns].sort_values(
            ["customer_key", "time_key", "channel_key", "product_key"]
        )

        summary = (
            items.groupby("basket_id", as_index=False)
            .agg(
                customer_key=("customer_key", "first"),
                full_date=("full_date", "first"),
                time_key=("time_key", "first"),
                channel_key=("channel_key", "first"),
                unique_items=("product_key", "nunique"),
                total_quantity=("quantity", "sum"),
                basket_value=("item_sales", "sum"),
            )
            .sort_values(["customer_key", "time_key", "channel_key"])
        )

        items_path = OUTPUT_DIR / "basket_items.csv"
        summary_path = OUTPUT_DIR / "basket_summary.csv"
        items.to_csv(items_path, index=False)
        summary.to_csv(summary_path, index=False, float_format="%.6f")

        multi_item_baskets = int((summary["unique_items"] >= 2).sum())
        single_item_baskets = int((summary["unique_items"] == 1).sum())

        print("\nBasket preparation completed successfully.")
        print(f"Basket-product rows: {len(items):,}")
        print(f"Unique baskets: {len(summary):,}")
        print(f"Baskets with 2+ different products: {multi_item_baskets:,}")
        print(f"Baskets with 1 product: {single_item_baskets:,}")
        print(f"\nSaved: {items_path}")
        print(f"Saved: {summary_path}")
        print("\nNote: basket = customer + date + channel (proxy; no order ID exists).")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
