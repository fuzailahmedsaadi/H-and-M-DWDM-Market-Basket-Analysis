"""Review 2 - Step 3: classify H&M baskets as low-value or high-value.

Input:
    outputs/basket_summary.csv

Outputs:
    outputs/basket_value_labels.csv
    outputs/classification_test_predictions.csv
    outputs/classification_metrics.txt
    outputs/classification_feature_importance.csv
    models/basket_value_classifier.joblib

Important:
    H&M provides anonymised/scaled prices, not product costs or margins.
    This script therefore classifies basket value, NOT profit margin.
    Basket value is used only to create the target label and is NOT a model
    input feature (avoids direct target leakage).
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "outputs" / "basket_summary.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Features available from basket-level information; basket_value is deliberately excluded.
FEATURES = [
    "unique_items",
    "total_quantity",
    "channel_key",
    "month",
    "quarter",
    "day_of_week",
    "day_of_month",
]


def main() -> None:
    print("Review 2 - Step 3: High-value vs low-value basket classification")
    print(f"Reading: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Run 05_prepare_baskets.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    required = {
        "basket_id", "full_date", "channel_key", "unique_items",
        "total_quantity", "basket_value",
    }
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["full_date"] = pd.to_datetime(df["full_date"], errors="coerce")
    for col in ["channel_key", "unique_items", "total_quantity", "basket_value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["basket_id", "full_date", "channel_key", "unique_items", "total_quantity", "basket_value"]
    ).copy()
    if len(df) < 20:
        raise ValueError(f"Not enough usable baskets for classification: {len(df)}")

    # Date-derived features.
    df["month"] = df["full_date"].dt.month
    df["quarter"] = df["full_date"].dt.quarter
    df["day_of_week"] = df["full_date"].dt.dayofweek
    df["day_of_month"] = df["full_date"].dt.day

    # Define a reproducible project label from the median basket value.
    threshold = float(df["basket_value"].median())
    df["target"] = (df["basket_value"] > threshold).astype("int8")
    df["basket_class"] = df["target"].map({0: "Low-value", 1: "High-value"})

    if df["target"].nunique() != 2 or df["target"].value_counts().min() < 2:
        raise ValueError("Could not form two adequately represented classes from basket_value.")

    labels_path = OUTPUT_DIR / "basket_value_labels.csv"
    df[["basket_id", "basket_value", "basket_class"]].to_csv(labels_path, index=False)

    X = df[FEATURES].astype("float32")
    y = df["target"]
    row_indices = df.index
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Usable baskets: {len(df):,}")
    print(f"Median basket-value threshold: {threshold:.8f} (scaled/anonymised units)")
    print("Class counts:")
    print(df["basket_class"].value_counts().to_string())
    print(f"Training rows: {len(X_train):,}; test rows: {len(X_test):,}")
    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=5,
        max_samples=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Low-value", "High-value"],
        digits=4,
        zero_division=0,
    )

    test_predictions = pd.DataFrame({
        "basket_id": df.loc[idx_test, "basket_id"].to_numpy(),
        "basket_value": df.loc[idx_test, "basket_value"].to_numpy(),
        "actual_class": y_test.map({0: "Low-value", 1: "High-value"}).to_numpy(),
        "predicted_class": pd.Series(predictions, index=idx_test).map(
            {0: "Low-value", 1: "High-value"}
        ).to_numpy(),
    })
    pred_path = OUTPUT_DIR / "classification_test_predictions.csv"
    test_predictions.to_csv(pred_path, index=False)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_path = OUTPUT_DIR / "classification_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    model_path = MODEL_DIR / "basket_value_classifier.joblib"
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold}, model_path)

    metrics_path = OUTPUT_DIR / "classification_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("H&M Basket Value Classification - Results\n")
        f.write("==========================================\n")
        f.write("Target: High-value vs Low-value basket, threshold = median basket_value\n")
        f.write("Note: H&M prices are anonymised/scaled; no cost/margin data is available.\n")
        f.write(f"Usable baskets: {len(df):,}\n")
        f.write(f"Median threshold: {threshold:.8f}\n")
        f.write(f"Train rows: {len(X_train):,}\nTest rows: {len(X_test):,}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Confusion matrix (rows=actual [Low, High], columns=predicted [Low, High]):\n")
        f.write(f"{matrix}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nFeature importance (highest first):\n")
        f.write(importance.to_string(index=False))
        f.write("\n")

    print("\nClassification completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion matrix (actual rows: Low, High; predicted columns: Low, High):")
    print(matrix)
    print("\nClassification report:")
    print(report)
    print("\nTop features:")
    print(importance.head(7).to_string(index=False))
    print("\nSaved outputs:")
    print(f"  {labels_path}")
    print(f"  {pred_path}")
    print(f"  {metrics_path}")
    print(f"  {importance_path}")
    print(f"  {model_path}")
    print("Done.")


if __name__ == "__main__":
    main()
