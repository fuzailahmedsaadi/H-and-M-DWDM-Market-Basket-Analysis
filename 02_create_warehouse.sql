-- ============================================================
-- H&M DWDM Project - Oracle Star Schema
-- Oracle Database 21c XE
-- Schema/User: HM_DW
-- ============================================================

-- ------------------------------------------------------------
-- 1. TIME DIMENSION
-- ------------------------------------------------------------
CREATE TABLE dim_time (
    time_key      NUMBER(8) PRIMARY KEY,
    full_date     DATE NOT NULL,
    day           NUMBER(2),
    month         NUMBER(2),
    month_name    VARCHAR2(12),
    quarter       NUMBER(1),
    year          NUMBER(4),
    day_name      VARCHAR2(12)
);

-- ------------------------------------------------------------
-- 2. CUSTOMER DIMENSION
-- ------------------------------------------------------------
CREATE TABLE dim_customer (
    customer_key            NUMBER(10) PRIMARY KEY,
    customer_id             VARCHAR2(64),
    age                     NUMBER(3),
    age_group               VARCHAR2(10),
    club_member_status      VARCHAR2(20),
    fashion_news_frequency  VARCHAR2(20),
    is_active               NUMBER(1)
);

-- ------------------------------------------------------------
-- 3. PRODUCT DIMENSION
-- ------------------------------------------------------------
CREATE TABLE dim_product (
    product_key         NUMBER(10) PRIMARY KEY,
    article_id          NUMBER(19),
    prod_name           VARCHAR2(255),
    product_type_name   VARCHAR2(100),
    product_group_name  VARCHAR2(100),
    department_name     VARCHAR2(100),
    index_group_name    VARCHAR2(100),
    colour_group_name   VARCHAR2(100)
);

-- ------------------------------------------------------------
-- 4. CHANNEL DIMENSION
-- ------------------------------------------------------------
CREATE TABLE dim_channel (
    channel_key   NUMBER(2) PRIMARY KEY,
    channel_id    NUMBER(2),
    channel_name  VARCHAR2(30)
);

-- ------------------------------------------------------------
-- 5. SALES FACT TABLE
-- ------------------------------------------------------------
CREATE TABLE fact_sales (
    sales_key      NUMBER(12) PRIMARY KEY,
    time_key       NUMBER(8) NOT NULL,
    customer_key   NUMBER(10) NOT NULL,
    product_key    NUMBER(10) NOT NULL,
    channel_key    NUMBER(2) NOT NULL,
    unit_price     NUMBER(14,6),
    quantity       NUMBER(10),
    sales_amount   NUMBER(14,6),

    CONSTRAINT fk_sales_time
        FOREIGN KEY (time_key)
        REFERENCES dim_time(time_key),

    CONSTRAINT fk_sales_customer
        FOREIGN KEY (customer_key)
        REFERENCES dim_customer(customer_key),

    CONSTRAINT fk_sales_product
        FOREIGN KEY (product_key)
        REFERENCES dim_product(product_key),

    CONSTRAINT fk_sales_channel
        FOREIGN KEY (channel_key)
        REFERENCES dim_channel(channel_key)
);