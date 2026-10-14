"""Review 2 - Step 4: Cluster H&M transaction baskets.

Reads outputs/basket_summary.csv and groups similar baskets with K-Means.
The basket definition is the proxy created in Step 1: customer + date + channel.

Outputs:
  outputs/basket_clusters.csv
  outputs/cluster_summary.csv
  outputs/clustering_metrics.txt
  models/basket_cluster_model.joblib
  models/basket_cluster_scaler.joblib
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
INPUT_FILE = OUTPUT_DIR / "basket_summary.csv"
N_CLUSTERS = 4
RANDOM_STATE = 42
SILHOUETTE_SAMPLE_SIZE = 3000


def main() -> None:
    print("Review 2 - Step 4: Clustering transaction baskets")
    print(f"Reading: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Run 05_prepare_baskets.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, parse_dates=["full_date"])
    required = {
        "basket_id", "customer_key", "full_date", "channel_key",
        "unique_items", "total_quantity", "basket_value"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"basket_summary.csv is missing columns: {sorted(missing)}")

    # Prepare numeric features. We deliberately exclude basket_id and customer_key:
    # they are identifiers, not meaningful measures of similarity.
    df["is_channel_2"] = (df["channel_key"] == 2).astype(int)
    df["month"] = df["full_date"].dt.month
    df["day_of_week"] = df["full_date"].dt.dayofweek
    df["day_of_month"] = df["full_date"].dt.day

    feature_columns = [
        "unique_items",
        "total_quantity",
        "basket_value",
        "is_channel_2",
        "month",
        "day_of_week",
        "day_of_month",
    ]

    for column in feature_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    usable = df.dropna(subset=feature_columns).copy()
    if len(usable) < N_CLUSTERS:
        raise ValueError(
            f"Need at least {N_CLUSTERS} valid baskets; found {len(usable)}."
        )

    print(f"Usable baskets: {len(usable):,}")
    print(f"Features: {', '.join(feature_columns)}")
    print(f"Fitting K-Means with {N_CLUSTERS} clusters...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(usable[feature_columns])

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )
    labels = model.fit_predict(X_scaled)
    usable["cluster_id"] = labels

    # A cluster profile makes IDs interpretable using the original units.
    usable["is_channel_2"] = usable["is_channel_2"].astype(int)
    summary = (
        usable.groupby("cluster_id", as_index=False)
        .agg(
            basket_count=("basket_id", "count"),
            mean_basket_value=("basket_value", "mean"),
            median_basket_value=("basket_value", "median"),
            mean_unique_items=("unique_items", "mean"),
            mean_total_quantity=("total_quantity", "mean"),
            channel_2_percent=("is_channel_2", lambda s: 100 * s.mean()),
        )
        .sort_values("cluster_id")
    )

    # Silhouette score is estimated on a reproducible sample to avoid an
    # expensive all-pairs calculation over 300k+ baskets.
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(usable))
    silhouette = silhouette_score(
        X_scaled,
        labels,
        sample_size=sample_size,
        random_state=RANDOM_STATE,
    )

    clustered_file = OUTPUT_DIR / "basket_clusters.csv"
    summary_file = OUTPUT_DIR / "cluster_summary.csv"
    metrics_file = OUTPUT_DIR / "clustering_metrics.txt"
    model_file = MODEL_DIR / "basket_cluster_model.joblib"
    scaler_file = MODEL_DIR / "basket_cluster_scaler.joblib"

    # Preserve original basket summary columns plus the assigned cluster ID.
    original_columns = list(pd.read_csv(INPUT_FILE, nrows=0).columns)
    output_df = usable[original_columns + ["cluster_id"]]
    output_df.to_csv(clustered_file, index=False)
    summary.to_csv(summary_file, index=False, float_format="%.6f")
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    with metrics_file.open("w", encoding="utf-8") as handle:
        handle.write("H&M Basket Clustering Results\n")
        handle.write("=============================\n")
        handle.write(f"Input baskets: {len(df):,}\n")
        handle.write(f"Baskets used: {len(usable):,}\n")
        handle.write(f"Algorithm: K-Means\n")
        handle.write(f"Number of clusters: {N_CLUSTERS}\n")
        handle.write(f"Random state: {RANDOM_STATE}\n")
        handle.write(
            f"Silhouette score (sample of {sample_size:,} baskets): "
            f"{silhouette:.4f}\n\n"
        )
        handle.write("Features:\n")
        for feature in feature_columns:
            handle.write(f"- {feature}\n")
        handle.write("\nCluster profile:\n")
        handle.write(summary.to_string(index=False))
        handle.write("\n")

    print("\nClustering completed.")
    print(f"Silhouette score (sample of {sample_size:,}): {silhouette:.4f}")
    print("Cluster summary:")
    print(summary.to_string(index=False))
    print("\nSaved outputs:")
    print(f"  {clustered_file}")
    print(f"  {summary_file}")
    print(f"  {metrics_file}")
    print("Saved models:")
    print(f"  {model_file}")
    print(f"  {scaler_file}")
    print("Done.")


if __name__ == "__main__":
    main()
