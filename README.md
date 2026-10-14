# H&M Fashion Retail Data Warehouse and Market Basket Analysis

## Project Overview
This project uses the H&M Personalized Fashion dataset to analyze retail sales, customer purchasing behaviour, and product-category relationships using Oracle Database and Python.

## Technologies Used
- Python 3.11
- Pandas and NumPy
- Oracle Database 21c XE
- SQLAlchemy and python-oracledb
- Scikit-learn
- MLxtend

## Project Implementation

### Phase 1: Data Warehouse and OLAP
- Dataset profiling
- Star-schema data warehouse creation
- ETL using Python and Oracle Database
- OLAP operations: roll-up, drill-down, slice, dice, and pivot

### Phase 2: Market Basket Analysis
1. Transaction basket preparation
2. High-value and low-value basket classification
3. K-Means clustering
4. Association rule mining using FP-Growth
5. Next-item category prediction

## Results
- Sales records processed: 1,035,601
- Unique baskets: 332,271
- Classification accuracy: 82.06%
- Clusters generated: 4
- Frequent itemsets identified: 62
- Next-item Top-1 hit rate: 52.41%
- Next-item Top-3 hit rate: 83.37%

## Important Notes
- Baskets are approximated using customer, date, and sales channel because the dataset has no order ID.
- Classification predicts high-value versus low-value baskets, not actual profit margins, because cost and margin information is unavailable.
- Large datasets, virtual environments, trained models, and large generated CSV files are excluded from Git.

## Project Structure
- `01_profile_dataset.py` — dataset profiling
- `02_create_warehouse.sql` — warehouse schema
- `03_etl.py` — data extraction and loading
- `04_olap_queries.sql` — OLAP queries
- `05_prepare_baskets.py` — basket preparation
- `06_classify_baskets.py` — basket classification
- `07_cluster_baskets.py` — K-Means clustering
- `08_association_rules.py` — association rule mining
- `09_predict_next_item.py` — next-item prediction
- `outputs/` — selected analysis results
- `screenshots/` — project execution evidence

## Setup
1. Install Python and Oracle Database.
2. Obtain the H&M dataset from Kaggle and place the CSV files in `data/`.
3. Create and activate a Python virtual environment.
4. Install the required Python packages.
5. Configure database credentials in a local `.env` file.
6. Create the warehouse tables and run the scripts in the appropriate order.

Never upload `.env` or database credentials to GitHub. Do not rerun ETL on populated tables unless duplicate insertion is prevented.
