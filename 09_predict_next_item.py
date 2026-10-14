"""Market Basket Analysis: time-based next-product-group prediction.

Input:
    outputs/basket_items.csv

Outputs:
    outputs/next_item_predictions.csv
    outputs/next_item_transition_counts.csv
    outputs/next_item_prediction_metrics.txt

Important methodology notes:
- The source dataset has no order ID or exact purchase timestamp.
- For sequential prediction, a purchase event is approximated as one customer-day:
  product groups bought by the same customer on the same date are combined,
  regardless of channel. This avoids inventing an order between same-day channels.
- Product groups are predicted, not exact article IDs.
- A chronological date cutoff is used: transitions ending on/before the cutoff
  train the model; transitions ending after it are held out for testing.
"""

from collections import Counter
from pathlib import Path
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
INPUT_FILE = OUTPUT_DIR / "basket_items.csv"
TRAIN_DATE_FRACTION = 0.80
TOP_K = 3


def clean_group_name(series: pd.Series) -> pd.Series:
    return (
        series.fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )


def main() -> int:
    print("Review 2 - Step 6: Next Likely Item Prediction")
    print("Reading customer/date/product-group data...")

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Make sure outputs/basket_items.csv exists in the project folder.")
        return 1

    columns = ["customer_key", "full_date", "product_group_name"]
    try:
        items = pd.read_csv(INPUT_FILE, usecols=columns, low_memory=False)
    except (ValueError, OSError) as exc:
        print(f"ERROR: Could not read required columns from {INPUT_FILE}: {exc}")
        return 1

    if items.empty:
        print("ERROR: basket_items.csv contains no rows.")
        return 1

    items = items.dropna(subset=["customer_key", "full_date"]).copy()
    items["full_date"] = pd.to_datetime(items["full_date"], errors="coerce")
    items = items.dropna(subset=["full_date"])
    items["product_group_name"] = clean_group_name(items["product_group_name"])
    items["customer_key"] = items["customer_key"].astype(str)

    # One category should occur only once within a customer's daily event.
    # Daily events union the channels, because their order within the same day
    # cannot be established from the source data.
    items = items.drop_duplicates(
        subset=["customer_key", "full_date", "product_group_name"]
    )
    daily = (
        items.groupby(["customer_key", "full_date"], sort=False)["product_group_name"]
        .agg(lambda values: tuple(sorted(set(values))))
        .rename("product_groups")
        .reset_index()
        .sort_values(["customer_key", "full_date"])
        .reset_index(drop=True)
    )

    if daily["full_date"].nunique() < 2:
        print("ERROR: At least two distinct purchase dates are required.")
        return 1

    unique_dates = sorted(daily["full_date"].drop_duplicates().tolist())
    cutoff_index = min(
        len(unique_dates) - 1,
        max(0, int((len(unique_dates) - 1) * TRAIN_DATE_FRACTION)),
    )
    cutoff_date = unique_dates[cutoff_index]

    pair_counts: Counter[tuple[str, str]] = Counter()
    target_group_counts: Counter[str] = Counter()
    train_transitions = 0
    test_transitions: list[tuple[str, str, pd.Timestamp, pd.Timestamp, tuple[str, ...], tuple[str, ...]]] = []

    print(f"Customer-day events: {len(daily):,}")
    print(f"Distinct product groups: {items['product_group_name'].nunique():,}")
    print(f"Chronological training cutoff: {cutoff_date.date()}")
    print("Learning category transitions between consecutive purchase days...")

    for customer_key, customer_rows in daily.groupby("customer_key", sort=False):
        records = list(
            customer_rows[["full_date", "product_groups"]].itertuples(
                index=False, name=None
            )
        )
        for (previous_date, previous_groups), (next_date, next_groups) in zip(
            records, records[1:]
        ):
            if not previous_groups or not next_groups:
                continue

            if next_date <= cutoff_date:
                train_transitions += 1
                for target_group in next_groups:
                    target_group_counts[target_group] += 1
                for source_group in previous_groups:
                    for target_group in next_groups:
                        pair_counts[(source_group, target_group)] += 1
            else:
                test_transitions.append(
                    (
                        str(customer_key),
                        "",  # basket IDs are not required for daily sequence predictions
                        previous_date,
                        next_date,
                        tuple(previous_groups),
                        tuple(next_groups),
                    )
                )

    if train_transitions == 0:
        print("ERROR: No training transitions were found before the cutoff.")
        return 1
    if not test_transitions:
        print("ERROR: No test transitions were found after the cutoff.")
        return 1

    # Convert category-pair counts into a lookup table for transparent reporting.
    transition_rows = [
        {"previous_product_group": source, "next_product_group": target, "transition_count": count}
        for (source, target), count in pair_counts.items()
    ]
    transition_table = pd.DataFrame(transition_rows).sort_values(
        ["transition_count", "previous_product_group", "next_product_group"],
        ascending=[False, True, True],
    )

    # Most common next groups in training serve as a safe fallback for an
    # unseen source group. For known groups, score destinations by the total
    # learned transition counts from all groups in the previous daily event.
    global_fallback = [group for group, _ in target_group_counts.most_common(TOP_K)]
    source_lookup: dict[str, Counter[str]] = {}
    for (source, target), count in pair_counts.items():
        source_lookup.setdefault(source, Counter())[target] += count

    predictions = []
    hit_at_1 = 0
    hit_at_k = 0
    recall_at_k_sum = 0.0

    for customer_key, _, previous_date, next_date, previous_groups, actual_groups in test_transitions:
        scores: Counter[str] = Counter()
        for source_group in previous_groups:
            scores.update(source_lookup.get(source_group, Counter()))

        if scores:
            ranked_groups = [
                group for group, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            ]
        else:
            ranked_groups = list(global_fallback)

        # Ensure top-K is populated when possible using the global fallback.
        for group in global_fallback:
            if group not in ranked_groups:
                ranked_groups.append(group)
            if len(ranked_groups) >= TOP_K:
                break

        recommended = ranked_groups[:TOP_K]
        actual_set = set(actual_groups)
        matched = actual_set.intersection(recommended)
        first_hit = bool(recommended and recommended[0] in actual_set)
        any_hit = bool(matched)
        recall = len(matched) / len(actual_set) if actual_set else 0.0
        hit_at_1 += int(first_hit)
        hit_at_k += int(any_hit)
        recall_at_k_sum += recall

        predictions.append(
            {
                "customer_key": customer_key,
                "previous_date": previous_date.date().isoformat(),
                "target_date": next_date.date().isoformat(),
                "previous_product_groups": " | ".join(previous_groups),
                "actual_next_product_groups": " | ".join(actual_groups),
                "predicted_top_1": recommended[0] if recommended else "",
                "predicted_top_3": " | ".join(recommended),
                "hit_at_1": int(first_hit),
                "hit_at_3": int(any_hit),
                "recall_at_3": recall,
            }
        )

    test_count = len(predictions)
    hit1_rate = hit_at_1 / test_count
    hitk_rate = hit_at_k / test_count
    mean_recall = recall_at_k_sum / test_count

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = OUTPUT_DIR / "next_item_predictions.csv"
    transitions_path = OUTPUT_DIR / "next_item_transition_counts.csv"
    metrics_path = OUTPUT_DIR / "next_item_prediction_metrics.txt"

    pd.DataFrame(predictions).to_csv(predictions_path, index=False, float_format="%.6f")
    transition_table.to_csv(transitions_path, index=False)

    metrics_lines = [
        "MARKET BASKET ANALYSIS - NEXT LIKELY PRODUCT-GROUP PREDICTION",
        "",
        f"Customer-day events: {len(daily):,}",
        f"Product groups: {items['product_group_name'].nunique():,}",
        f"Training transitions: {train_transitions:,}",
        f"Test transitions: {test_count:,}",
        f"Chronological cutoff date: {cutoff_date.date().isoformat()}",
        f"Top-1 accuracy (hit rate): {hit1_rate:.4f} ({hit1_rate:.2%})",
        f"Top-{TOP_K} hit rate: {hitk_rate:.4f} ({hitk_rate:.2%})",
        f"Mean recall at {TOP_K}: {mean_recall:.4f} ({mean_recall:.2%})",
        "",
        "Method: for each customer, learn product-group transitions between consecutive",
        "purchase days in the earlier time period; recommend the most frequent next groups",
        "for product groups seen on the previous purchase day.",
        "",
        "Important limitations:",
        "- The dataset has no explicit order ID or exact time of purchase.",
        "- A customer-day is used as the purchase-event proxy; channels are combined per day.",
        "- Predictions are at product-group level, not exact article level.",
        "- These are offline historical evaluation metrics, not guaranteed future outcomes.",
        "",
        "Top 10 learned category transitions:",
    ]
    for row in transition_table.head(10).itertuples(index=False):
        metrics_lines.append(
            f"{row.previous_product_group} -> {row.next_product_group} | count={row.transition_count:,}"
        )
    metrics_path.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    print("\nNext-item prediction completed successfully.")
    print(f"Training transitions: {train_transitions:,}")
    print(f"Test transitions: {test_count:,}")
    print(f"Top-1 accuracy (hit rate): {hit1_rate:.2%}")
    print(f"Top-{TOP_K} hit rate: {hitk_rate:.2%}")
    print(f"Mean recall at {TOP_K}: {mean_recall:.2%}")
    print("\nTop 10 learned transitions:")
    print(transition_table.head(10).to_string(index=False))
    print("\nSaved:")
    print(f"{predictions_path}")
    print(f"{transitions_path}")
    print(f"{metrics_path}")
    print("Reminder: a customer-day is a proxy purchase event, not a confirmed order.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
