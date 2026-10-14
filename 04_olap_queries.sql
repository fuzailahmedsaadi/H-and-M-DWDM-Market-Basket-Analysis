-- Step 4 - Sample OLAP queries (run each one and screenshot the result for slide 6)

-- 1. ROLL-UP: product type -> product group -> index group (here: product group level)
SELECT p.index_group_name, p.product_group_name,
       SUM(f.sales_amount) AS total_sales, SUM(f.quantity) AS units_sold
FROM fact_sales f
JOIN dim_product p ON f.product_key = p.product_key
GROUP BY p.index_group_name, p.product_group_name
ORDER BY p.index_group_name, total_sales DESC;

-- 2. DRILL-DOWN: Year -> Quarter -> Month
SELECT t.year, t.quarter, t.month, SUM(f.sales_amount) AS total_sales
FROM fact_sales f
JOIN dim_time t ON f.time_key = t.time_key
GROUP BY t.year, t.quarter, t.month
ORDER BY t.year, t.quarter, t.month;

-- 3. SLICE: fix one dimension (Year = 2019)
SELECT p.product_group_name, SUM(f.sales_amount) AS total_sales
FROM fact_sales f
JOIN dim_time t ON f.time_key = t.time_key
JOIN dim_product p ON f.product_key = p.product_key
WHERE t.year = 2019
GROUP BY p.product_group_name
ORDER BY total_sales DESC;

-- 4. DICE: sub-cube on 3 dimensions (age group, product group, year)
SELECT c.age_group, p.product_group_name, SUM(f.sales_amount) AS total_sales
FROM fact_sales f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_product p ON f.product_key = p.product_key
JOIN dim_time t ON f.time_key = t.time_key
WHERE c.age_group IN ('16-25', '26-35')
  AND p.product_group_name IN ('Garment Upper body', 'Accessories')
  AND t.year IN (2019, 2020)
GROUP BY c.age_group, p.product_group_name
ORDER BY c.age_group, total_sales DESC;

-- 5. PIVOT: sales channels as columns
SELECT p.product_group_name,
       SUM(CASE WHEN f.channel_key = 1 THEN f.sales_amount END) AS channel_1,
       SUM(CASE WHEN f.channel_key = 2 THEN f.sales_amount END) AS channel_2
FROM fact_sales f
JOIN dim_product p ON f.product_key = p.product_key
GROUP BY p.product_group_name
ORDER BY p.product_group_name;
