"""Market Basket Analysis: frequent itemsets and association rules.

Input:  outputs/basket_items.csv
Outputs: outputs/frequent_itemsets.csv, outputs/association_rules.csv,
         outputs/association_rule_metrics.txt

Important: A basket is defined as customer + date + channel because the
source dataset has no order ID. Results therefore describe proxy baskets.
"""

from pathlib import Path
import sys

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
INPUT_FILE = OUTPUT_DIR / "basket_items.csv"

# Starting thresholds; we can adjust them later if the resulting rules are
# too few or too many.
MIN_SUPPORT = 0.005
MIN_CONFIDENCE = 0.10
MAX_ITEMSET_LENGTH = 3


def itemset_to_text(itemset) -> str:
    """Convert mlxtend frozensets into readable, stable CSV text."""
    return " | ".join(sorted(str(item) for item in itemset))


def main() -> int:
    print("Review 2 - Step 5: Association Rule Mining")
    print("Reading basket IDs and product groups...")

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Make sure outputs/basket_items.csv exists in the project folder.")
        return 1

    # Read only the two needed columns to reduce memory usage.
    items = pd.read_csv(
        INPUT_FILE,
        usecols=["basket_id", "product_group_name"],
        low_memory=False,
    )

    items = items.dropna(subset=["basket_id"]).copy()
    items["product_group_name"] = (
        items["product_group_name"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    # A category should appear only once per basket, even if several articles
    # from that category were bought in the same proxy basket.
    items = items.drop_duplicates(subset=["basket_id", "product_group_name"])

    # Build one list of unique product groups per basket.
    basket_groups = items.groupby("basket_id", sort=False)["product_group_name"].agg(list)
    basket_groups = basket_groups[basket_groups.map(len) > 0]

    print(f"Unique baskets: {len(basket_groups):,}")
    print(f"Unique product groups: {items['product_group_name'].nunique():,}")
    print(f"Minimum support: {MIN_SUPPORT:.3f} ({MIN_SUPPORT:.1%} of baskets)")
    print("Building basket-by-category matrix...")

    transactions = basket_groups.tolist()
    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions, sparse=True)
    basket_matrix = pd.DataFrame.sparse.from_spmatrix(
        encoded,
        columns=encoder.columns_,
        index=basket_groups.index,
    )

    print(f"Matrix shape: {basket_matrix.shape[0]:,} baskets x {basket_matrix.shape[1]:,} product groups")
    print("Mining frequent itemsets with FP-Growth...")

    frequent_itemsets = fpgrowth(
        basket_matrix,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_ITEMSET_LENGTH,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    itemsets_output = OUTPUT_DIR / "frequent_itemsets.csv"
    rules_output = OUTPUT_DIR / "association_rules.csv"
    metrics_output = OUTPUT_DIR / "association_rule_metrics.txt"

    if frequent_itemsets.empty:
        frequent_itemsets.to_csv(itemsets_output, index=False)
        pd.DataFrame().to_csv(rules_output, index=False)
        message = (
            "No frequent itemsets found with the current support threshold "
            f"({MIN_SUPPORT}). Try lowering MIN_SUPPORT in 08_association_rules.py.\n"
        )
        metrics_output.write_text(message, encoding="utf-8")
        print(message)
        return 0

    frequent_itemsets["itemsets_text"] = frequent_itemsets["itemsets"].map(itemset_to_text)
    frequent_itemsets["itemset_length"] = frequent_itemsets["itemsets"].map(len)
    frequent_itemsets.drop(columns=["itemsets"]).rename(
        columns={"itemsets_text": "itemsets"}
    ).sort_values(["itemset_length", "support"], ascending=[True, False]).to_csv(
        itemsets_output, index=False
    )

    print(f"Frequent itemsets found: {len(frequent_itemsets):,}")
    print("Generating association rules...")

    if not (frequent_itemsets["itemsets"].map(len) >= 2).any():
        rules = pd.DataFrame()
    else:
        rules = association_rules(
            frequent_itemsets.drop(columns=["itemsets_text", "itemset_length"]),
            metric="confidence",
            min_threshold=MIN_CONFIDENCE,
        )

    if not rules.empty:
        # Keep rules with lift > 1: the consequent is more likely with the
        # antecedent than it is across all baskets.
        rules = rules[rules["lift"] > 1.0].copy()
        rules["antecedents"] = rules["antecedents"].map(itemset_to_text)
        rules["consequents"] = rules["consequents"].map(itemset_to_text)
        rules = rules.sort_values(
            ["lift", "confidence", "support"], ascending=[False, False, False]
        )

    rules.to_csv(rules_output, index=False)

    lines = [
        "MARKET BASKET ANALYSIS - ASSOCIATION RULE MINING",
        "",
        f"Proxy baskets analysed: {len(basket_groups):,}",
        f"Product groups: {basket_matrix.shape[1]:,}",
        f"Minimum support: {MIN_SUPPORT:.4f}",
        f"Minimum confidence: {MIN_CONFIDENCE:.4f}",
        f"Maximum itemset length: {MAX_ITEMSET_LENGTH}",
        f"Frequent itemsets found: {len(frequent_itemsets):,}",
        f"Association rules retained (lift > 1): {len(rules):,}",
        "",
        "Interpretation: support is the fraction of baskets containing both sides;",
        "confidence is the fraction of antecedent baskets that also contain the",
        "consequent; lift > 1 means the consequent is more common with the",
        "antecedent than overall.",
        "",
        "Caveat: baskets are customer + date + channel proxies, not confirmed order IDs.",
    ]

    if not rules.empty:
        lines.extend(["", "Top 10 rules by lift:"])
        for _, rule in rules.head(10).iterrows():
            lines.append(
                f"{rule['antecedents']} -> {rule['consequents']} | "
                f"support={rule['support']:.4f}, "
                f"confidence={rule['confidence']:.4f}, lift={rule['lift']:.4f}"
            )

    metrics_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nTop association rules (sorted by lift):")
    if rules.empty:
        print("No rules passed the current thresholds. We can tune them after checking this output.")
    else:
        print(
            rules[["antecedents", "consequents", "support", "confidence", "lift"]]
            .head(10)
            .to_string(index=False)
        )

    print("\nAssociation rule mining completed.")
    print(f"Saved: {itemsets_output}")
    print(f"Saved: {rules_output}")
    print(f"Saved: {metrics_output}")
    print("Reminder: each basket is customer + date + channel (proxy; no order ID exists).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
