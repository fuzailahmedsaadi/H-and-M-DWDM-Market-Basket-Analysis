"""H&M DWDM Review 3 dashboard — light retail analytics UI.

Run from the project root:
    python -m streamlit run dashboard.py

Reads existing CSV/TXT outputs only. It does not connect to Oracle, rerun ETL,
retrain models, or modify project outputs.
"""
from pathlib import Path
import re

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

st.set_page_config(
    page_title="H&M Retail Intelligence | Review 3",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A light theme is configured in .streamlit/config.toml as well. These CSS
# rules refine spacing, contrast, metric cards, tables, and navigation styling.
st.markdown(
    """
    <style>
    :root {
      --hm-red: #c8102e;
      --hm-red-dark: #a80d25;
      --ink: #19232f;
      --muted: #657181;
      --line: #e4e8ee;
      --canvas: #f5f7fa;
      --paper: #ffffff;
    }
    html, body, [data-testid="stAppViewContainer"] {
      background: var(--canvas) !important;
      color: var(--ink) !important;
    }
    [data-testid="stHeader"] { background: rgba(245,247,250,0.92) !important; }
    .main .block-container {
      max-width: 1480px;
      padding-top: 1.65rem;
      padding-bottom: 3rem;
      padding-left: 2.2rem;
      padding-right: 2.2rem;
    }
    [data-testid="stSidebar"] {
      background: #ffffff !important;
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
      border-radius: 9px;
      padding-top: .15rem;
      padding-bottom: .15rem;
    }
    h1, h2, h3 { color: var(--ink) !important; letter-spacing: -0.025em; }
    h1 { font-weight: 760 !important; }
    h2, h3 { font-weight: 700 !important; }
    p, li, label { color: var(--ink); }
    [data-testid="stMetric"] {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 1rem 1.1rem;
      box-shadow: 0 2px 9px rgba(20,35,55,.035);
      min-height: 112px;
    }
    [data-testid="stMetricLabel"] p {
      color: var(--muted) !important;
      font-size: .84rem !important;
      font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
      color: var(--ink) !important;
      font-weight: 760 !important;
      letter-spacing: -0.03em;
    }
    [data-testid="stMetricDelta"] { font-size: .78rem !important; }
    [data-testid="stDataFrame"] {
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: white;
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    div[data-testid="stExpander"] {
      background: white;
      border: 1px solid var(--line);
      border-radius: 12px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
      border-color: var(--line) !important;
      border-radius: 14px !important;
    }
    .hero {
      background: linear-gradient(112deg, #ffffff 0%, #fff8f9 100%);
      border: 1px solid #f0dce1;
      border-left: 5px solid var(--hm-red);
      border-radius: 16px;
      padding: 1.35rem 1.6rem 1.25rem 1.6rem;
      margin: .1rem 0 1.5rem 0;
    }
    .hero-eyebrow {
      color: var(--hm-red) !important;
      font-size: .72rem;
      font-weight: 800;
      letter-spacing: .14em;
      text-transform: uppercase;
      margin-bottom: .45rem;
    }
    .hero-title {
      color: var(--ink) !important;
      font-size: clamp(1.55rem, 2.3vw, 2.25rem);
      font-weight: 800;
      line-height: 1.13;
      letter-spacing: -.04em;
      margin: 0 0 .45rem 0;
    }
    .hero-subtitle {
      color: var(--muted) !important;
      font-size: .98rem;
      line-height: 1.5;
      margin: 0;
    }
    .sidebar-brand {
      background: #fff7f8;
      border: 1px solid #f2dfe3;
      border-radius: 12px;
      padding: 1rem .9rem;
      margin: .15rem 0 1.15rem 0;
    }
    .sidebar-brand .mark {
      font-size: 1.35rem;
      font-weight: 900;
      letter-spacing: -.06em;
      color: var(--hm-red);
      margin-bottom: .15rem;
    }
    .sidebar-brand .sub {
      color: var(--muted);
      font-size: .72rem;
      line-height: 1.35;
      text-transform: uppercase;
      letter-spacing: .08em;
      font-weight: 700;
    }
    .section-kicker {
      color: var(--hm-red) !important;
      font-size: .74rem;
      text-transform: uppercase;
      font-weight: 800;
      letter-spacing: .12em;
      margin-bottom: .15rem;
    }
    .note {
      background: #fff9ec;
      border: 1px solid #f4e6c8;
      border-radius: 11px;
      padding: .85rem 1rem;
      color: #6e5628 !important;
      font-size: .9rem;
      line-height: 1.5;
    }
    .success-note {
      background: #f0faf5;
      border: 1px solid #d2eddf;
      border-radius: 11px;
      padding: .85rem 1rem;
      color: #235d42 !important;
      font-size: .9rem;
      line-height: 1.5;
    }
    .footer {
      border-top: 1px solid var(--line);
      margin-top: 2.4rem;
      padding-top: .85rem;
      color: #7b8795;
      font-size: .78rem;
    }
    code { color: #9b1930 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def report_text(filename: str) -> str:
    path = OUT / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8-sig", errors="replace")


def metric_from_report(filename: str, pattern: str, default=None, cast=float):
    match = re.search(pattern, report_text(filename), flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return default
    try:
        return cast(match.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return default


@st.cache_data(show_spinner="Loading basket summary...")
def load_basket_summary() -> pd.DataFrame:
    path = OUT / "basket_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "full_date" in df.columns:
        df["full_date"] = pd.to_datetime(df["full_date"], errors="coerce")
    for column in ["unique_items", "total_quantity", "basket_value", "channel_key"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    path = OUT / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def show_missing(filename: str) -> None:
    st.warning(f"Could not find `outputs/{filename}`. Check that the file exists in your project folder.")


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"<div class='hero'><div class='hero-eyebrow'>{kicker}</div>"
        f"<div class='hero-title'>{title}</div><p class='hero-subtitle'>{description}</p></div>",
        unsafe_allow_html=True,
    )


summary = load_basket_summary()
classification_report = report_text("classification_metrics.txt")
clustering_report = report_text("clustering_metrics.txt")
association_report = report_text("association_rule_metrics.txt")
prediction_report = report_text("next_item_prediction_metrics.txt")

# Navigation labels are intentionally short for a cleaner sidebar.
pages = [
    "Overview",
    "Basket preparation",
    "Classification",
    "Clustering",
    "Association rules",
    "Next-item prediction",
    "Methodology & limitations",
]

with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'><div class='mark'>H&amp;M</div>"
        "<div class='sub'>Retail intelligence<br>DWDM · Review 3</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Navigate")
    page = st.radio("Dashboard sections", pages, label_visibility="collapsed")
    st.divider()
    st.caption("Local mode · Reads saved files from `outputs/`.")
    st.caption("No ETL, database writes, or model training are triggered.")

# Results are loaded from the saved project output reports. Recorded run counts
# are fallbacks if a summary file is absent, not newly calculated results.
basket_count = len(summary) if not summary.empty else 332_271
sales_rows = 1_035_601
basket_product_rows = 1_027_402
multi_count = int((summary["unique_items"] >= 2).sum()) if not summary.empty and "unique_items" in summary.columns else 221_894
single_count = int((summary["unique_items"] == 1).sum()) if not summary.empty and "unique_items" in summary.columns else 110_377
accuracy = metric_from_report("classification_metrics.txt", r"Accuracy:\s*([0-9.]+)", 0.8206)
silhouette = metric_from_report("clustering_metrics.txt", r"Silhouette score[^:]*:\s*([0-9.]+)", 0.2207)
frequent_count = metric_from_report("association_rule_metrics.txt", r"Frequent itemsets found:\s*([0-9,]+)", 62, int)
rule_count = metric_from_report("association_rule_metrics.txt", r"Association rules retained[^:]*:\s*([0-9,]+)", 71, int)
top1 = metric_from_report("next_item_prediction_metrics.txt", r"Top-1 accuracy \(hit rate\):\s*[0-9.]+\s*\(([0-9.]+)%\)", 52.41)
top3 = metric_from_report("next_item_prediction_metrics.txt", r"Top-3 hit rate:\s*[0-9.]+\s*\(([0-9.]+)%\)", 83.37)
recall3 = metric_from_report("next_item_prediction_metrics.txt", r"Mean recall at 3:\s*[0-9.]+\s*\(([0-9.]+)%\)", 72.02)

if page == "Overview":
    page_header(
        "REVIEW 3 · PROJECT OVERVIEW",
        "Project overview",
        "See the main results from all six analysis steps in one place, from preparing baskets to predicting likely next product categories.",
    )
    st.markdown("<div class='section-kicker'>Performance snapshot</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Sales rows", f"{sales_rows:,}")
    cols[1].metric("Proxy baskets", f"{basket_count:,}")
    cols[2].metric("Classifier accuracy", f"{accuracy * 100:.2f}%")
    cols[3].metric("Top-3 hit rate", f"{top3:.2f}%")
    cols = st.columns(4)
    cols[0].metric("Multi-product baskets", f"{multi_count:,}")
    cols[1].metric("Clusters", "4")
    cols[2].metric("Frequent itemsets", f"{frequent_count:,}")
    cols[3].metric("Retained rules", f"{rule_count:,}")

    st.markdown("### Analysis journey")
    journey = st.columns(3)
    journey[0].markdown("**01 · Prepare**\n\nCreate customer-date-channel proxy baskets and summarize their contents.")
    journey[1].markdown("**02 · Understand**\n\nClassify basket value and discover basket profiles with machine learning.")
    journey[2].markdown("**03 · Recommend**\n\nFind product-group associations and predict likely next categories.")
    left, right = st.columns([1.05, 1.3], gap="large")
    with left:
        st.markdown("### Basket composition")
        composition = pd.DataFrame({"Basket type": ["1 product", "2+ products"], "Baskets": [single_count, multi_count]})
        st.bar_chart(composition.set_index("Basket type"), color="#c8102e")
    with right:
        st.markdown("### At a glance")
        st.markdown(
            "<div class='success-note'><b>Implemented:</b> basket preparation, value classification, four-cluster segmentation, FP-Growth association rules, and time-based next-category prediction.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='note' style='margin-top: .7rem'><b>Methodology note:</b> the dataset has no explicit order ID. Basket analyses use customer + date + channel as a proxy; the next-item task uses customer-day events.</div>",
            unsafe_allow_html=True,
        )

elif page == "Basket preparation":
    page_header("STEPS 1–2", "Basket preparation and validation", "See how sales records were grouped into baskets, and review basket counts, product counts, quantities, and basket values.")
    cols = st.columns(3)
    cols[0].metric("Sales rows read", f"{sales_rows:,}")
    cols[1].metric("Basket-product rows", f"{basket_product_rows:,}")
    cols[2].metric("Unique baskets", f"{basket_count:,}")
    cols = st.columns(2)
    cols[0].metric("Baskets with 2+ products", f"{multi_count:,}")
    cols[1].metric("Baskets with 1 product", f"{single_count:,}")
    st.markdown("### What the preparation means")
    st.write("Each basket is keyed by customer, date/time key, and sales channel. It is a proxy basket because the source data has no explicit order ID. Separate purchases by the same customer on the same date and channel may therefore be combined.")
    if not summary.empty:
        st.markdown("### Basket summary preview")
        st.dataframe(summary.head(15), use_container_width=True, hide_index=True)
        if "unique_items" in summary.columns:
            st.markdown("### Number of unique products in a basket")
            dist = summary["unique_items"].fillna(0).astype(int).value_counts().sort_index().head(20)
            st.bar_chart(dist, color="#c8102e")
    else:
        show_missing("basket_summary.csv")

elif page == "Classification":
    page_header("STEP 3 · CLASSIFICATION", "High-value and low-value baskets", "Review how the Random Forest classifies baskets, how accurately it performed, and which features influenced its predictions most.")
    cols = st.columns(3)
    cols[0].metric("Test accuracy", f"{accuracy * 100:.2f}%")
    cols[1].metric("Training baskets", "265,816")
    cols[2].metric("Test baskets", "66,455")
    st.markdown(
        "<div class='note'>The median basket-value threshold is 0.06096600 in anonymized/scaled units. The data has no product costs or profit margins, so this predicts basket value, not profit margin.</div>",
        unsafe_allow_html=True,
    )
    importance = load_csv("classification_feature_importance.csv")
    if not importance.empty and {"feature", "importance"}.issubset(importance.columns):
        importance["importance"] = pd.to_numeric(importance["importance"], errors="coerce")
        importance = importance.dropna(subset=["importance"]).sort_values("importance", ascending=False)
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown("### What influenced predictions?")
            st.bar_chart(importance.set_index("feature")[["importance"]], color="#c8102e")
        with right:
            st.markdown("### Feature importance")
            st.dataframe(importance, use_container_width=True, hide_index=True)
    else:
        show_missing("classification_feature_importance.csv")
    with st.expander("Open classification metrics and confusion matrix"):
        st.code(classification_report or "classification_metrics.txt not found", language="text")
    predictions = load_csv("classification_test_predictions.csv")
    if not predictions.empty:
        st.markdown("### Test prediction sample")
        st.dataframe(predictions.head(50), use_container_width=True, hide_index=True)

elif page == "Clustering":
    page_header("STEP 4 · CLUSTERING", "Basket groups", "Explore the four groups created by K-Means and compare their average product count, quantity, and basket value.")
    cols = st.columns(3)
    cols[0].metric("Baskets clustered", f"{basket_count:,}")
    cols[1].metric("Clusters", "4")
    cols[2].metric("Silhouette score", f"{silhouette:.4f}")
    st.markdown("<div class='note'>The silhouette score suggests modest cluster separation with some overlap. Cluster numbers are labels, not a ranking.</div>", unsafe_allow_html=True)
    cluster_summary = load_csv("cluster_summary.csv")
    if not cluster_summary.empty:
        st.markdown("### Cluster profiles")
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
        measures = [c for c in ["mean_unique_items", "mean_total_quantity", "mean_basket_value"] if c in cluster_summary.columns]
        if "cluster_id" in cluster_summary.columns and measures:
            st.markdown("### Compare cluster averages")
            st.bar_chart(cluster_summary.set_index("cluster_id")[measures], color=["#c8102e", "#73839b", "#b5becb"][: len(measures)])
    else:
        show_missing("cluster_summary.csv")
    with st.expander("Open full clustering report"):
        st.code(clustering_report or "clustering_metrics.txt not found", language="text")

elif page == "Association rules":
    page_header("STEP 5 · ASSOCIATION RULES", "Product categories that appear together", "Explore common category combinations and use support, confidence, and lift to understand the strength of each rule.")
    cols = st.columns(3)
    cols[0].metric("Frequent itemsets", f"{frequent_count:,}")
    cols[1].metric("Retained rules", f"{rule_count:,}")
    cols[2].metric("Product groups", "18")
    st.caption("Settings: minimum support 0.5% · minimum confidence 10% · maximum itemset length 3 · retained rules have lift > 1.")
    rules = load_csv("association_rules.csv")
    if not rules.empty:
        for col in ["support", "confidence", "lift"]:
            if col in rules.columns:
                rules[col] = pd.to_numeric(rules[col], errors="coerce")
        if "lift" in rules.columns:
            max_lift = max(2.0, float(rules["lift"].dropna().max())) if rules["lift"].notna().any() else 2.0
            min_lift = st.slider("Filter by minimum lift", 1.0, max_lift, 1.0, 0.05)
            filtered = rules[rules["lift"] >= min_lift].copy()
        else:
            filtered = rules.copy()
        sort_by = next((col for col in ["lift", "confidence", "support"] if col in filtered.columns), None)
        if sort_by:
            filtered = filtered.sort_values(sort_by, ascending=False)
        cols_to_show = [c for c in ["antecedents", "consequents", "support", "confidence", "lift"] if c in filtered.columns]
        st.markdown("### Strongest category relationships")
        st.dataframe(filtered[cols_to_show].head(100), use_container_width=True, hide_index=True)
        st.caption(f"Showing {min(100, len(filtered)):,} of {len(filtered):,} rules matching the filter.")
    else:
        show_missing("association_rules.csv")
    with st.expander("Read the association-rule report"):
        st.code(association_report or "association_rule_metrics.txt not found", language="text")

elif page == "Next-item prediction":
    page_header("STEP 6 · NEXT-ITEM PREDICTION", "Predicting the next product category", "See which categories often follow earlier customer-day purchases and how often the top recommendations matched later events.")
    cols = st.columns(3)
    cols[0].metric("Top-1 hit rate", f"{top1:.2f}%")
    cols[1].metric("Top-3 hit rate", f"{top3:.2f}%")
    cols[2].metric("Mean recall at 3", f"{recall3:.2f}%")
    st.markdown("<div class='note'>Chronological training cutoff: 28 April 2020. Predictions are product groups, not exact article IDs, and the evaluation is historical/offline.</div>", unsafe_allow_html=True)
    transitions = load_csv("next_item_transition_counts.csv")
    if not transitions.empty:
        st.markdown("### Most common learned transitions")
        st.dataframe(transitions.head(15), use_container_width=True, hide_index=True)
        required = {"previous_product_group", "next_product_group", "transition_count"}
        if required.issubset(transitions.columns):
            chart = transitions.head(10).copy()
            chart["transition"] = chart["previous_product_group"].astype(str) + " → " + chart["next_product_group"].astype(str)
            chart = chart.set_index("transition")[["transition_count"]]
            st.markdown("### Top transition counts")
            st.bar_chart(chart, color="#c8102e")
    else:
        show_missing("next_item_transition_counts.csv")
    predictions = load_csv("next_item_predictions.csv")
    if not predictions.empty:
        st.markdown("### Prediction examples")
        view = predictions.copy()
        if "hit_at_3" in view.columns:
            match_filter = st.selectbox("Prediction outcome", ["All predictions", "Top-3 hit", "Top-3 miss"])
            if match_filter == "Top-3 hit":
                view = view[view["hit_at_3"] == 1]
            elif match_filter == "Top-3 miss":
                view = view[view["hit_at_3"] == 0]
        st.dataframe(view.head(50), use_container_width=True, hide_index=True)
    else:
        show_missing("next_item_predictions.csv")
    with st.expander("Open detailed prediction metrics"):
        st.code(prediction_report or "next_item_prediction_metrics.txt not found", language="text")

else:
    page_header("PROJECT NOTES", "Methodology and limitations", "Understand the assumptions behind the results, what the dataset cannot tell us, and which improvements would make the analysis stronger.")
    st.markdown("### Methodology limitations")
    st.markdown(
        "- **Proxy basket:** customer + date + sales channel, because an explicit order ID is unavailable. Separate same-day purchases can be merged.\n"
        "- **Value classification:** uses a median basket-value label; the dataset does not provide product costs or profit margins.\n"
        "- **Clustering:** four K-Means groups were created; a 0.2207 silhouette score indicates modest separation with overlap.\n"
        "- **Association rules:** categories appearing together indicate association, not causation.\n"
        "- **Next-item prediction:** predicts product groups rather than exact articles; the metrics are offline historical results."
    )
    st.markdown("### Improvements to consider")
    st.markdown(
        "- Compare classification algorithms and tune hyperparameters using validation data.\n"
        "- Compare multiple K values and test clusters without the channel feature.\n"
        "- Check association-rule stability across time periods.\n"
        "- Compare next-item predictions with a popularity baseline and evaluate on later data.\n"
        "- Add an exportable summary and screenshot evidence for the final demonstration."
    )
    with st.expander("View saved result reports"):
        st.markdown("**Classification**")
        st.code(classification_report or "Not found", language="text")
        st.markdown("**Clustering**")
        st.code(clustering_report or "Not found", language="text")
        st.markdown("**Association rules**")
        st.code(association_report or "Not found", language="text")
        st.markdown("**Next-item prediction**")
        st.code(prediction_report or "Not found", language="text")

st.markdown(
    "<div class='footer'>H&amp;M DWDM · Review 3 dashboard &nbsp; | &nbsp; Local analytics from saved outputs &nbsp; | &nbsp; No ETL or model training is run here.</div>",
    unsafe_allow_html=True,
)
