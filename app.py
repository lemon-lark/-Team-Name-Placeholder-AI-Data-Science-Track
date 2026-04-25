"""AI-partments Streamlit dashboard.

Run with:

    streamlit run app.py

The app always works in Mock Demo Mode. Switch the LLM provider in the
sidebar's Demo controls panel to use Ollama or IBM watsonx Granite.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.agents import answer_user_question
from src.database import (
    get_available_tables,
    get_table_row_count,
    initialize_database,
)
from src.ingestion.csv_loader import list_raw_files, save_uploaded_file
from src.ingestion.pipeline import IngestionPipeline
from src.llm_client import LLMClient
from src.seed_data import seed_demo_data
from src.tools.chart_tools import render_chart
from src.tools.map_tools import build_overview_deck, build_pydeck_layer
from src.tools.query_tools import (
    get_header_metrics,
    get_listing_profile,
    get_listing_profile_by_address,
    run_safe_sql,
    table_preview,
)


# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI-partments",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


EXAMPLE_QUERIES: list[str] = [
    "Find 1-bedroom apartments under $2,500 in low-crime neighborhoods near parks.",
    "Compare Astoria, Harlem, and Flushing by transit, parks, and housing supply.",
    "Which neighborhoods have the best combination of transit and affordability?",
    "Rank neighborhoods by renter_fit_score.",
    "Show neighborhoods with high availability score and good green space.",
    "Which neighborhoods have lots of HPD housing stock?",
    "Which neighborhoods have the best subway access and affordable rent?",
]


# Score columns we treat as "renter dimensions" for the Compare and Top
# Neighborhoods tabs. Any column missing in neighborhood_stats is skipped.
SCORE_DIMENSIONS: list[tuple[str, str]] = [
    ("renter_fit_score", "Renter fit"),
    ("affordability_score", "Affordability"),
    ("safety_score", "Safety"),
    ("transit_score", "Transit"),
    ("green_space_score", "Green space"),
    ("amenity_score", "Amenities"),
    ("availability_score", "Availability"),
]

SCORE_DEFINITIONS: dict[str, str] = {
    "renter_fit_score": "Composite score (30% affordability, 25% safety, 25% amenities, 20% availability).",
    "affordability_score": "Higher means lower typical rent in this neighborhood.",
    "safety_score": "Inverse of normalized crime-event volume in available data.",
    "transit_score": "Blend of subway access (distance tiers) and bus access (stop density + proximity).",
    "green_space_score": "Park access from park count, total acres, and nearest-park bonus.",
    "amenity_score": "Transit + green space + public services, with optional worship/grocery terms.",
    "availability_score": "Listings + housing stock signal (HPD), with a no-listings fallback formula.",
}

# Score-like columns that can appear in result tables. Used to hide
# non-requested score outputs while preserving non-score factual columns.
RESULT_SCORE_COLUMNS: set[str] = {
    "renter_fit_score",
    "safety_score",
    "transit_score",
    "green_space_score",
    "affordability_score",
    "amenity_score",
    "availability_score",
    "crime_score",
    "price_match_score",
    "safety_match_score",
    "transit_match_score",
    "proximity_match_score",
    "amenity_match_score",
    "overall_match_score",
}


CUSTOM_CSS = """
<style>
    :root {
        --ai-bg: #0b1220;
        --ai-bg-elevated: #121b2f;
        --ai-surface: #162238;
        --ai-surface-soft: #1a2945;
        --ai-border: rgba(148, 163, 184, 0.24);
        --ai-border-strong: rgba(96, 165, 250, 0.42);
        --ai-text: #e2e8f0;
        --ai-text-muted: #94a3b8;
        --ai-accent: #60a5fa;
        --ai-accent-2: #818cf8;
        --ai-success: #34d399;
        --ai-shadow: 0 12px 34px rgba(2, 6, 23, 0.5);
        --ai-radius: 14px;
    }
    .stApp {
        background:
            radial-gradient(1000px 450px at 6% -10%, rgba(129, 140, 248, 0.16), transparent 54%),
            radial-gradient(900px 360px at 96% 2%, rgba(56, 189, 248, 0.1), transparent 52%),
            linear-gradient(180deg, #0b1220 0%, #0a1020 100%);
        color: var(--ai-text);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.1rem;
    }
    h1, h2, h3, h4, h5 {
        color: #f8fafc;
        letter-spacing: -0.01em;
    }
    p, li, label, .stMarkdown, .stCaption {
        color: var(--ai-text);
    }
    /* Sidebar skin */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #101b31 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    section[data-testid="stSidebar"] * {
        color: var(--ai-text);
    }
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        white-space: normal;
        line-height: 1.25;
        font-size: 0.88rem;
        padding: 0.52rem 0.72rem;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(30, 41, 59, 0.55);
        color: #dbeafe;
        transition: all 120ms ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(96, 165, 250, 0.55);
        background: rgba(37, 99, 235, 0.22);
        transform: translateY(-1px);
    }
    /* Hero */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(22, 34, 56, 0.96), rgba(18, 27, 47, 0.92));
        padding: 0.7rem 1rem;
        border-radius: var(--ai-radius);
        border: 1px solid var(--ai-border);
        box-shadow: var(--ai-shadow);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: var(--ai-text-muted);
        opacity: 1;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
        color: #f8fafc;
    }
    [data-testid="stMetricDelta"] {
        color: var(--ai-success) !important;
    }
    /* Dataframe / table surface */
    [data-testid="stDataFrame"] {
        border-radius: var(--ai-radius);
        overflow: hidden;
        border: 1px solid var(--ai-border);
        box-shadow: var(--ai-shadow);
    }
    [data-testid="stDataFrame"] > div {
        background: var(--ai-surface) !important;
    }
    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.4rem;
        padding: 0.2rem;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 10px;
        height: 2.35rem;
        color: var(--ai-text-muted);
        font-weight: 600;
        transition: all 120ms ease;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #f8fafc !important;
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.38), rgba(99, 102, 241, 0.35)) !important;
        border: 1px solid var(--ai-border-strong) !important;
    }
    /* Inputs and controls */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        background: var(--ai-surface) !important;
        border: 1px solid var(--ai-border) !important;
        color: var(--ai-text) !important;
    }
    div[data-baseweb="select"] svg {
        fill: var(--ai-text-muted);
    }
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(96, 165, 250, 0.35);
        background: linear-gradient(180deg, rgba(37, 99, 235, 0.32), rgba(30, 64, 175, 0.42));
        color: #dbeafe;
        font-weight: 600;
        transition: all 120ms ease;
    }
    .stButton > button:hover {
        border-color: rgba(96, 165, 250, 0.72);
        background: linear-gradient(180deg, rgba(59, 130, 246, 0.4), rgba(37, 99, 235, 0.48));
        transform: translateY(-1px);
    }
    /* Chat + status blocks */
    [data-testid="stChatMessage"] {
        border-radius: var(--ai-radius);
        border: 1px solid var(--ai-border);
        background: linear-gradient(180deg, rgba(22, 34, 56, 0.8), rgba(18, 27, 47, 0.75));
        box-shadow: var(--ai-shadow);
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--ai-border);
        background: rgba(15, 23, 42, 0.72);
    }
    [data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid var(--ai-border);
        background: rgba(15, 23, 42, 0.58);
    }
    .ai-hero {
        margin-bottom: 0.8rem;
        padding: 1.05rem 1.2rem;
        border-radius: 18px;
        border: 1px solid var(--ai-border);
        background:
            radial-gradient(580px 200px at 0% 0%, rgba(96, 165, 250, 0.2), transparent 62%),
            linear-gradient(180deg, rgba(22, 34, 56, 0.98), rgba(18, 27, 47, 0.95));
        box-shadow: var(--ai-shadow);
    }
    .ai-hero h1 {
        font-size: 2.15rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.015em;
        color: #f8fafc;
    }
    .ai-hero p {
        margin: 0;
        color: var(--ai-text-muted);
        font-size: 1.03rem;
    }
    .ai-result-card {
        border-radius: 12px;
        border: 1px solid var(--ai-border);
        background: linear-gradient(180deg, rgba(26, 41, 69, 0.9), rgba(20, 31, 53, 0.9));
        box-shadow: var(--ai-shadow);
        padding: 0.75rem 0.85rem;
    }
    .ai-result-card h4 {
        margin: 0 0 0.2rem 0;
        font-size: 0.98rem;
        color: #f8fafc;
    }
    .ai-result-card p {
        margin: 0;
        color: var(--ai-text-muted);
        font-size: 0.85rem;
    }
</style>
"""


# ----------------------------------------------------------------------------
# Bootstrap: ensure DB exists; seed if empty.
# ----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _bootstrap_database() -> dict[str, int]:
    initialize_database()
    if get_table_row_count("neighborhoods") == 0:
        return seed_demo_data()
    return {"neighborhoods": get_table_row_count("neighborhoods")}


@st.cache_data(show_spinner=False)
def _cached_table_preview(table: str, limit: int, _token: float) -> pd.DataFrame:
    return table_preview(table, limit=limit)


def _refresh_token() -> float:
    """Bump the cache token so the next call re-fetches."""
    st.session_state["cache_token"] = st.session_state.get("cache_token", 0.0) + 1.0
    return st.session_state["cache_token"]


_bootstrap_database()
if "cache_token" not in st.session_state:
    st.session_state["cache_token"] = 0.0

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar: brand, example questions, hidden Demo controls
# ----------------------------------------------------------------------------


with st.sidebar:
    st.markdown("## AI-partments")
    st.caption("Smarter renter intelligence for NYC.")

    st.markdown("### Try an example")
    for i, ex in enumerate(EXAMPLE_QUERIES):
        if st.button(ex, key=f"ex_sidebar_{i}", use_container_width=True):
            st.session_state["queued_question"] = ex

    st.divider()

    with st.expander("Demo controls", expanded=False):
        st.caption(
            "Behind-the-scenes settings for live demos. Hidden by default so the "
            "main view stays focused on the renter experience."
        )

        provider_label_to_value = {
            "Mock Demo Mode": "mock",
            "Ollama (local)": "ollama",
            "IBM Granite / watsonx.ai": "watsonx",
        }
        default_label = next(
            (k for k, v in provider_label_to_value.items() if v == config.LLM_PROVIDER),
            "Mock Demo Mode",
        )
        provider_label = st.selectbox(
            "LLM provider",
            options=list(provider_label_to_value.keys()),
            index=list(provider_label_to_value.keys()).index(default_label),
        )
        provider = provider_label_to_value[provider_label]

        ollama_model = st.text_input(
            "Ollama model",
            value=config.OLLAMA_MODEL,
            help="Examples: llama3.1:8b, granite3.3:8b",
        )
        watsonx_model = st.text_input(
            "watsonx model",
            value=config.WATSONX_MODEL_ID,
            help="Default: ibm/granite-3-3-8b-instruct",
        )

        if st.button("Rebuild demo database", help="Regenerates synthetic NYC/NJ data."):
            seed_demo_data()
            _refresh_token()
            st.success("Demo database rebuilt.")

        st.checkbox(
            "Show data tools tab",
            key="show_data_tools",
            help="Reveals the NYC Data Pipeline tab with file ingestion controls.",
        )

        st.divider()
        st.caption("Database")
        st.code(str(config.DB_PATH), language="text")

        available_tables = get_available_tables()
        st.caption(f"Tables ({len(available_tables)})")
        for tbl in available_tables:
            rows = get_table_row_count(tbl)
            st.caption(f"- {tbl}  ({rows:,} rows)")


def get_llm() -> LLMClient:
    return LLMClient(
        provider=provider,
        ollama_model=ollama_model,
        watsonx_model=watsonx_model,
    )


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------


st.markdown(
    """
<div class="ai-hero">
  <h1>AI-partments</h1>
  <p>Smarter renter intelligence for NYC. Ask in plain English, get apartments, neighborhoods, charts, and maps.</p>
</div>
""",
    unsafe_allow_html=True,
)


# Header KPIs are intentionally uncached so they always reflect the latest
# ingested DB state, even when ingestion runs in another tab/session.
metrics = get_header_metrics()
m1, m2, m3, m4 = st.columns(4)

m1.metric("Listings", f"{metrics.get('total_listings', 0):,}")

median_rent = metrics.get("median_rent") or metrics.get("avg_rent")
m2.metric(
    "Median rent",
    f"${median_rent:,.0f}" if median_rent else "-",
)

best_fit = metrics.get("best_renter_fit_neighborhood") or "-"
best_fit_score = metrics.get("best_renter_fit_score")
m3.metric(
    "Top-rated neighborhood",
    best_fit,
    f"{best_fit_score:.0f} renter-fit" if best_fit_score is not None else None,
)

best_transit = metrics.get("best_transit_neighborhood") or "-"
best_transit_score = metrics.get("best_transit_score")
m4.metric(
    "Best subway access",
    best_transit,
    f"{best_transit_score:.0f} transit" if best_transit_score is not None else None,
)


# ----------------------------------------------------------------------------
# Tab construction (NYC Data Pipeline only when the demo toggle is on)
# ----------------------------------------------------------------------------


tab_labels = [
    "Find a place",
    "Compare neighborhoods",
    "Top neighborhoods",
    "Methodology",
    "About",
]
show_data_tools = bool(st.session_state.get("show_data_tools"))
if show_data_tools:
    tab_labels.append("NYC Data Pipeline")

tabs = st.tabs(tab_labels)
tab_ask = tabs[0]
tab_compare = tabs[1]
tab_rankings = tabs[2]
tab_methodology = tabs[3]
tab_about = tabs[4]
tab_pipeline = tabs[5] if show_data_tools else None


# ----------------------------------------------------------------------------
# Tab: Find a place (Ask AI)
# ----------------------------------------------------------------------------


def _apartment_result_column_config(
    df: pd.DataFrame, visible_score_columns: list[str] | None = None
) -> dict[str, Any]:
    """Build a Streamlit column_config that formats apartment-result columns
    (rent, distances, violation counts, scores, sqft) consistently."""
    config: dict[str, Any] = {}
    if "rent" in df.columns:
        config["rent"] = st.column_config.NumberColumn("Rent", format="$%d")
    if "sqft" in df.columns:
        config["sqft"] = st.column_config.NumberColumn("Sqft", format="%d")
    if "bedrooms" in df.columns:
        config["bedrooms"] = st.column_config.NumberColumn("Beds", format="%d")
    if "bathrooms" in df.columns:
        config["bathrooms"] = st.column_config.NumberColumn("Baths", format="%.1f")
    for col, label in (
        ("nearest_subway_distance", "Subway (mi)"),
        ("nearest_bus_stop_distance", "Bus (mi)"),
        ("nearest_park_distance", "Park (mi)"),
        ("distance_to_subway_miles", "Subway (mi)"),
        ("distance_to_bus_miles", "Bus (mi)"),
    ):
        if col in df.columns:
            config[col] = st.column_config.NumberColumn(label, format="%.2f mi")
    for col in df.columns:
        if col.startswith("distance_to_") and col.endswith("_miles") and col not in config:
            pretty = col.replace("distance_to_", "").replace("_miles", "").replace("_", " ").title()
            config[col] = st.column_config.NumberColumn(f"{pretty} (mi)", format="%.2f mi")
    if "hpd_violation_count" in df.columns:
        config["hpd_violation_count"] = st.column_config.NumberColumn(
            "HPD violations", format="%d"
        )
    allowed_scores = set(visible_score_columns or [])
    for score_col in RESULT_SCORE_COLUMNS:
        if allowed_scores and score_col not in allowed_scores:
            continue
        if score_col in df.columns:
            label = score_col.replace("_", " ").title()
            config[score_col] = st.column_config.NumberColumn(label, format="%.0f")
    return config


def _result_table_columns(df: pd.DataFrame, result: dict[str, Any]) -> list[str]:
    """Select uncluttered result-table columns:
    fixed listing fields + requested parameters + overall match."""
    fixed_order = [
        "neighborhood",
        "borough",
        "address",
        "rent",
        "bedrooms",
        "bathrooms",
        "sqft",
    ]
    out: list[str] = [c for c in fixed_order if c in df.columns]

    filters = ((result.get("router_output") or {}).get("filters") or {})
    requested_proximity = filters.get("proximity") or []
    targets = {str((p or {}).get("target") or "").lower() for p in requested_proximity}
    requested_params: list[str] = []
    transit_pref = str(filters.get("transit_preference") or "").lower()

    def _append_first_available(candidates: list[str]) -> None:
        for c in candidates:
            if c in df.columns and c not in requested_params:
                requested_params.append(c)
                return

    # Show raw proximity values for requested targets.
    if requested_proximity:
        if "subway" in targets:
            _append_first_available(["distance_to_subway_miles", "nearest_subway_distance"])
            _append_first_available(["distance_to_bus_miles", "nearest_bus_stop_distance"])
        if "bus" in targets:
            _append_first_available(["distance_to_bus_miles", "nearest_bus_stop_distance"])
            _append_first_available(["distance_to_subway_miles", "nearest_subway_distance"])
        if "park" in targets:
            requested_params.extend(["nearest_park_distance"])
        # Generic fallback if target is amenity/worship/etc.
        requested_params.extend(["distance_miles"])
    elif transit_pref == "any":
        _append_first_available(["distance_to_subway_miles", "nearest_subway_distance"])
        _append_first_available(["distance_to_bus_miles", "nearest_bus_stop_distance"])

    # Include requested score outputs from scoring layer.
    requested_params.extend(list(result.get("visible_score_columns") or []))

    for col in requested_params:
        if col in df.columns and col not in out:
            out.append(col)
    # Include any computed amenity-specific distance columns from SQL
    # (e.g., distance_to_mosque_miles) when proximity is requested.
    if requested_proximity:
        for col in df.columns:
            if col.startswith("distance_to_") and col.endswith("_miles") and col not in out:
                out.append(col)
    return out


def _score_sort_column(df: pd.DataFrame) -> str | None:
    if "overall_match_score" in df.columns:
        return "overall_match_score"
    if "renter_fit_score" in df.columns:
        return "renter_fit_score"
    return None


def _important_metric_column(df: pd.DataFrame) -> str | None:
    # Priority 1: known proximity/distance columns.
    for col in (
        "distance_to_subway_miles",
        "distance_to_bus_miles",
        "nearest_subway_distance",
        "nearest_bus_stop_distance",
        "nearest_park_distance",
        "distance_miles",
    ):
        if col in df.columns:
            return col
    # Priority 2: computed distance_to_*_miles columns.
    for col in df.columns:
        if col.startswith("distance_to_") and col.endswith("_miles"):
            return col
    # Priority 3: other useful score-like columns.
    for col in (
        "transit_score",
        "safety_score",
        "affordability_score",
        "value_score",
        "amenity_score",
        "availability_score",
    ):
        if col in df.columns:
            return col
    return None


def _metric_label(col: str) -> str:
    labels = {
        "distance_to_subway_miles": "Subway",
        "distance_to_bus_miles": "Bus",
        "nearest_subway_distance": "Subway",
        "nearest_bus_stop_distance": "Bus",
        "nearest_park_distance": "Park",
        "distance_miles": "Distance",
    }
    if col in labels:
        return labels[col]
    if col.startswith("distance_to_") and col.endswith("_miles"):
        return col.replace("distance_to_", "").replace("_miles", "").replace("_", " ").title()
    return col.replace("_", " ").title()


def _format_metric_value(col: str, value: Any) -> str:
    if pd.isna(value):
        return "-"
    if col.endswith("_miles") or "distance" in col:
        return f"{float(value):.2f} mi"
    return f"{float(value):.0f}"


def _format_profile_value(key: str, value: Any) -> str:
    if pd.isna(value):
        return "-"
    key_l = key.lower()
    if isinstance(value, (int, float)):
        if "distance" in key_l:
            return f"{float(value):.2f} mi"
        if "rent" in key_l or "income" in key_l:
            return f"${float(value):,.0f}"
        if "score" in key_l:
            return f"{float(value):.0f}"
        if key_l.endswith("_count") or "count" in key_l:
            return f"{int(value):,}"
        return f"{float(value):,.2f}" if not float(value).is_integer() else f"{int(value):,}"
    return str(value)


def _render_profile_section(
    title: str, profile: dict[str, Any], fields: list[tuple[str, str]]
) -> None:
    rows: list[dict[str, str]] = []
    for key, label in fields:
        if key not in profile:
            continue
        rows.append({"Field": label, "Value": _format_profile_value(key, profile[key])})
    if not rows:
        return
    st.markdown(f"#### {title}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_full_listing_profile(profile: dict[str, Any]) -> None:
    st.subheader("Full listing profile")
    apartment_id = profile.get("apartment_id")
    if apartment_id is not None:
        st.caption(f"Apartment ID: {int(apartment_id)}")

    _render_profile_section(
        "Listing basics",
        profile,
        [
            ("address", "Address"),
            ("neighborhood", "Neighborhood"),
            ("borough", "Borough"),
            ("city", "City"),
            ("state", "State"),
            ("rent", "Rent"),
            ("bedrooms", "Bedrooms"),
            ("bathrooms", "Bathrooms"),
            ("sqft", "Square feet"),
            ("available_date", "Available date"),
            ("listing_url", "Listing URL"),
            ("source", "Source"),
        ],
    )
    _render_profile_section(
        "Scores and fit",
        profile,
        [
            ("renter_fit_score", "Renter fit"),
            ("affordability_score", "Affordability"),
            ("safety_score", "Safety"),
            ("transit_score", "Transit"),
            ("green_space_score", "Green space"),
            ("amenity_score", "Amenity"),
            ("availability_score", "Availability"),
            ("value_score", "Value"),
            ("safety_affordability_score", "Safety + affordability"),
        ],
    )
    _render_profile_section(
        "Safety and housing risk",
        profile,
        [
            ("crime_score", "Crime score"),
            ("hpd_violation_count", "HPD violations"),
            ("housing_supply_score", "Housing supply score"),
            ("affordable_housing_signal", "Affordable housing signal"),
        ],
    )
    _render_profile_section(
        "Transit and access",
        profile,
        [
            ("listing_nearest_subway_distance", "Listing to nearest subway"),
            ("listing_nearest_bus_distance", "Listing to nearest bus"),
            ("nearest_park_distance", "Nearest park"),
            ("subway_access_score", "Subway access score"),
            ("bus_access_score", "Bus access score"),
            ("park_count", "Park count"),
            ("worship_count", "Worship places"),
            ("grocery_count", "Grocery stores"),
        ],
    )
    _render_profile_section(
        "Neighborhood context",
        profile,
        [
            ("median_listing_rent", "Neighborhood median listing rent"),
            ("neighborhood_avg_rent", "Neighborhood average rent"),
            ("neighborhood_min_rent", "Neighborhood minimum rent"),
            ("neighborhood_max_rent", "Neighborhood maximum rent"),
            ("neighborhood_median_rent", "Neighborhood median rent"),
            ("population", "Population"),
            ("median_income", "Median income"),
            ("listing_count", "Listing count"),
            ("nta_code", "NTA code"),
        ],
    )
    with st.expander("Raw profile data", expanded=False):
        st.json(profile)


@st.cache_data(show_spinner=False)
def _cached_global_listing_cards(_token: float) -> pd.DataFrame:
    """Global fallback source for Top 3 listing cards (no query yet)."""
    if get_table_row_count("apartments") == 0:
        return pd.DataFrame()
    sql = """
    SELECT
        a.apartment_id,
        a.address,
        a.rent,
        a.bedrooms,
        a.bathrooms,
        a.nearest_subway_distance,
        a.nearest_bus_stop_distance,
        n.name AS neighborhood,
        n.borough,
        ns.renter_fit_score
    FROM apartments a
    LEFT JOIN neighborhoods n ON n.neighborhood_id = a.neighborhood_id
    LEFT JOIN neighborhood_stats ns ON ns.neighborhood_id = a.neighborhood_id
    """
    try:
        return run_safe_sql(sql)
    except Exception:
        return pd.DataFrame()


def _render_top_listing_cards(df: pd.DataFrame, title: str) -> None:
    st.markdown("#### " + title)
    if df is None or df.empty:
        st.info("No listings available yet.")
        return

    working = df.copy()
    score_col = _score_sort_column(working)
    if score_col:
        ranked = working.sort_values(score_col, ascending=False, na_position="last")
    elif "rent" in working.columns:
        ranked = working.sort_values("rent", ascending=True, na_position="last")
    else:
        ranked = working
    ranked = ranked.head(3).reset_index(drop=True)

    metric_col = _important_metric_column(ranked)
    cols = st.columns(3)
    for i, col in enumerate(cols):
        if i >= len(ranked):
            with col:
                st.caption("No listing")
            continue
        row = ranked.iloc[i]
        primary = (
            row.get("address")
            or row.get("neighborhood")
            or row.get("name")
            or f"Listing #{i + 1}"
        )
        secondary_parts: list[str] = []
        if pd.notna(row.get("bedrooms")):
            secondary_parts.append(f"{int(row.get('bedrooms'))} bd")
        if pd.notna(row.get("bathrooms")):
            secondary_parts.append(f"{float(row.get('bathrooms')):.1f} ba")
        if row.get("neighborhood"):
            secondary_parts.append(str(row.get("neighborhood")))
        if row.get("borough"):
            secondary_parts.append(str(row.get("borough")))

        with col:
            st.markdown(
                f"""
<div class="ai-result-card">
  <h4>{primary}</h4>
  <p>{' • '.join(secondary_parts) if secondary_parts else 'Listing details'}</p>
</div>
""",
                unsafe_allow_html=True,
            )
            rent_val = row.get("rent")
            st.metric("Price", f"${float(rent_val):,.0f}" if pd.notna(rent_val) else "-")

            if metric_col and metric_col in ranked.columns:
                st.caption(
                    f"{_metric_label(metric_col)}: "
                    f"{_format_metric_value(metric_col, row.get(metric_col))}"
                )

            score_val = row.get("overall_match_score")
            if pd.isna(score_val):
                score_val = row.get("renter_fit_score")
            st.caption(
                f"Overall score: "
                f"{float(score_val):.0f}" if pd.notna(score_val) else "Overall score: -"
            )


def _render_result(result: dict[str, Any]) -> None:
    """Render a renter-friendly result block: recommendation first, then table,
    chart, and map. Pipeline internals are tucked into a single collapsed
    'Behind the scenes' expander for demo use."""
    recommendation = result.get("recommendation") or "Here are the results."
    st.subheader("Recommendation")
    st.write(recommendation)

    if "error" in result:
        st.warning(result["error"])

    df: pd.DataFrame = result.get("dataframe", pd.DataFrame())
    visible_score_columns = list(result.get("visible_score_columns") or [])
    overall_score_column = result.get("overall_score_column")
    if df is None or df.empty:
        st.info("No rows returned for this question.")
        overview = build_overview_deck()
        if overview is not None:
            st.subheader("Map")
            st.pydeck_chart(overview)
    else:
        display_df = df.copy()
        if visible_score_columns:
            hidden_score_cols = [
                c for c in display_df.columns if c in RESULT_SCORE_COLUMNS and c not in visible_score_columns
            ]
            if hidden_score_cols:
                display_df = display_df.drop(columns=hidden_score_cols)
        if overall_score_column and overall_score_column in display_df.columns:
            display_df = display_df.sort_values(
                overall_score_column, ascending=False, na_position="last"
            ).reset_index(drop=True)
        table_cols = _result_table_columns(display_df, result)
        table_df = display_df[table_cols] if table_cols else display_df

        st.subheader("Results")
        lookup_meta = pd.DataFrame(index=display_df.index)
        if "apartment_id" in display_df.columns:
            lookup_meta["apartment_id"] = display_df["apartment_id"]
        if "address" in display_df.columns:
            lookup_meta["address"] = display_df["address"]
        if "neighborhood" in display_df.columns:
            lookup_meta["neighborhood"] = display_df["neighborhood"]
        if "rent" in display_df.columns:
            lookup_meta["rent"] = display_df["rent"]

        table_interactive = table_df.copy()
        table_interactive.insert(0, "view_profile", False)
        table_editor_config = _apartment_result_column_config(
            table_df, visible_score_columns=visible_score_columns
        )
        table_editor_config["view_profile"] = st.column_config.CheckboxColumn(
            "View profile", help="Select one row to load full listing profile."
        )
        edited_table = st.data_editor(
            table_interactive,
            use_container_width=True,
            height=320,
            hide_index=True,
            disabled=[c for c in table_interactive.columns if c != "view_profile"],
            column_config=table_editor_config,
            key="results_profile_selector",
        )
        selected_idxs = edited_table.index[edited_table["view_profile"]].tolist()
        if len(selected_idxs) > 1:
            st.info("Select one listing at a time for full profile view.")
        selected_idx = selected_idxs[-1] if selected_idxs else None
        if selected_idx is not None:
            row_meta = lookup_meta.loc[selected_idx] if selected_idx in lookup_meta.index else pd.Series()
            apt_id = row_meta.get("apartment_id")
            if pd.notna(apt_id):
                profile = get_listing_profile(int(apt_id))
                st.session_state["selected_apartment_id"] = int(apt_id)
                st.session_state["selected_listing_profile"] = profile
                st.session_state["selected_profile_error"] = (
                    None if profile else "No full profile found for this listing."
                )
            else:
                address = str(row_meta.get("address") or "").strip()
                neighborhood = str(row_meta.get("neighborhood") or "").strip()
                rent = row_meta.get("rent")
                if address:
                    profile = get_listing_profile_by_address(
                        address=address,
                        neighborhood=neighborhood or None,
                        rent=float(rent) if pd.notna(rent) else None,
                    )
                    st.session_state["selected_apartment_id"] = (
                        int(profile["apartment_id"]) if profile and profile.get("apartment_id") is not None else None
                    )
                    st.session_state["selected_listing_profile"] = profile
                    st.session_state["selected_profile_error"] = (
                        None if profile else "No full profile found for this listing."
                    )
                else:
                    st.session_state["selected_apartment_id"] = None
                    st.session_state["selected_listing_profile"] = None
                    st.session_state["selected_profile_error"] = (
                        "This row does not include apartment_id or address for profile lookup."
                    )
        else:
            st.session_state["selected_profile_error"] = None

        chart_spec = result.get("chart_spec") or {}
        chart_type = chart_spec.get("chart_type", "table_only")
        if chart_type not in ("table_only",):
            fig = render_chart(display_df, chart_spec)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        selected_profile = st.session_state.get("selected_listing_profile")
        selected_error = st.session_state.get("selected_profile_error")
        if selected_profile:
            _render_full_listing_profile(selected_profile)
        elif selected_error:
            st.warning(selected_error)

        st.subheader("Map")
        deck = build_pydeck_layer(display_df, tooltip_fields=list(table_df.columns))
        if deck is not None:
            st.pydeck_chart(deck)
        else:
            overview = build_overview_deck()
            if overview is not None:
                st.caption(
                    "These results don't have point-level coordinates. "
                    "Showing an NYC neighborhood overview for context."
                )
                st.pydeck_chart(overview)

    with st.expander("Behind the scenes (router -> SQL -> safety)", expanded=False):
        st.caption(f"LLM provider: {result.get('provider_used', 'mock')}")

        st.markdown("**Router output**")
        st.json(result.get("router_output", {}))

        st.markdown("**Clarification check**")
        st.json(result.get("clarification") or {})

        st.markdown("**Schema selection**")
        sc = result.get("schema_context", {})
        st.write(f"Tables: {', '.join(sc.get('required_tables', []))}")
        if sc.get("notes"):
            st.caption(sc.get("notes", ""))
        st.json(sc)

        st.markdown("**Generated SQL**")
        st.code(result.get("sql", ""), language="sql")

        st.markdown("**SQL safety check**")
        sr = result.get("safety_result") or {}
        if sr.get("approved"):
            st.success(f"Approved. {sr.get('reason')}")
        else:
            st.error(f"Rejected: {sr.get('reason')}")
        st.json(sr)


def _reset_ask_conversation() -> None:
    st.session_state["ask_turns"] = []
    st.session_state["ask_history"] = []
    st.session_state["pending_clarification"] = None
    st.session_state["last_result"] = None
    st.session_state["selected_apartment_id"] = None
    st.session_state["selected_listing_profile"] = None
    st.session_state["selected_profile_error"] = None


def _process_ask_input(user_input: str) -> None:
    """Run the pipeline for a chat input, handling clarification short-circuits."""
    user_input = (user_input or "").strip()
    if not user_input:
        return
    st.session_state.setdefault("ask_history", []).append(
        {"role": "user", "content": user_input}
    )

    pending = st.session_state.get("pending_clarification")
    prior_turns: list[str] = list(st.session_state.get("ask_turns") or []) if pending else []

    try:
        result = answer_user_question(
            user_input, llm=get_llm(), prior_turns=prior_turns or None
        )
    except Exception as exc:
        st.session_state["ask_history"].append(
            {"role": "assistant", "content": f"Pipeline crashed: {exc}"}
        )
        st.session_state["pending_clarification"] = None
        st.session_state["last_result"] = None
        st.session_state["selected_apartment_id"] = None
        st.session_state["selected_listing_profile"] = None
        st.session_state["selected_profile_error"] = None
        return

    turns = list(result.get("turns") or (prior_turns + [user_input]))
    st.session_state["ask_turns"] = turns

    if result.get("needs_clarification"):
        clar = result.get("clarification") or {}
        follow_up = clar.get("follow_up_text") or "Could you tell me a bit more?"
        st.session_state["pending_clarification"] = clar
        st.session_state["last_result"] = None
        st.session_state["selected_apartment_id"] = None
        st.session_state["selected_listing_profile"] = None
        st.session_state["selected_profile_error"] = None
        st.session_state["ask_history"].append(
            {"role": "assistant", "content": follow_up, "is_clarification": True}
        )
        return

    st.session_state["pending_clarification"] = None
    st.session_state["last_result"] = result
    st.session_state["selected_apartment_id"] = None
    st.session_state["selected_listing_profile"] = None
    st.session_state["selected_profile_error"] = None
    summary = result.get("recommendation") or "Here are the results."
    st.session_state["ask_history"].append(
        {"role": "assistant", "content": summary, "is_clarification": False}
    )


def _render_ask_tab() -> None:
    st.subheader("Ask a question")
    st.caption(
        "Try one of the example questions in the sidebar, or type your own below."
    )

    st.session_state.setdefault("ask_turns", [])
    st.session_state.setdefault("ask_history", [])
    st.session_state.setdefault("pending_clarification", None)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("queued_question", None)
    st.session_state.setdefault("selected_apartment_id", None)
    st.session_state.setdefault("selected_listing_profile", None)
    st.session_state.setdefault("selected_profile_error", None)

    if st.session_state["ask_history"]:
        if st.button("New question", help="Reset the chat history."):
            _reset_ask_conversation()
            st.rerun()

    for entry in st.session_state["ask_history"]:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

    queued = st.session_state.get("queued_question")
    if queued:
        st.session_state["queued_question"] = None
        with st.chat_message("user"):
            st.write(queued)
        with st.spinner("Searching neighborhoods..."):
            _process_ask_input(queued)
        st.rerun()

    placeholder = (
        "Reply to the assistant's question..."
        if st.session_state.get("pending_clarification")
        else "E.g. Find 1-bedroom apartments under $2,500 near parks in Astoria."
    )
    user_input = st.chat_input(placeholder)
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.spinner("Searching neighborhoods..."):
            _process_ask_input(user_input)
        st.rerun()

    st.divider()
    if st.session_state.get("last_result"):
        result_df = (st.session_state["last_result"] or {}).get("dataframe", pd.DataFrame())
        _render_top_listing_cards(result_df, title="Top 3 results")
    else:
        if st.session_state.get("pending_clarification"):
            st.info("Add those details above to see top query-specific results.")
        _render_top_listing_cards(
            _cached_global_listing_cards(st.session_state["cache_token"]),
            title="Top 3 listings overall",
        )

    if st.session_state.get("last_result"):
        _render_result(st.session_state["last_result"])
    elif not st.session_state.get("ask_history"):
        st.subheader("NYC overview")
        st.caption(
            "Every dot is a neighborhood in the database. Ask a question above "
            "to filter and zoom in."
        )
        overview = build_overview_deck()
        if overview is not None:
            st.pydeck_chart(overview)


with tab_ask:
    _render_ask_tab()


# ----------------------------------------------------------------------------
# Tab: Compare neighborhoods
# ----------------------------------------------------------------------------


def _load_neighborhood_scoreboard() -> pd.DataFrame:
    """Return a joined neighborhood + score dataframe for the Compare/Rankings
    tabs. Empty if either source table is empty."""
    if (
        get_table_row_count("neighborhood_stats") == 0
        or get_table_row_count("neighborhoods") == 0
    ):
        return pd.DataFrame()
    ns = _cached_table_preview(
        "neighborhood_stats", 500, st.session_state["cache_token"]
    )
    nb = _cached_table_preview("neighborhoods", 500, st.session_state["cache_token"])
    if ns.empty or nb.empty:
        return pd.DataFrame()
    keep_cols = ["neighborhood_id", "name", "borough"]
    keep_cols = [c for c in keep_cols if c in nb.columns]
    joined = ns.merge(nb[keep_cols], on="neighborhood_id", how="left")
    return joined.rename(columns={"name": "neighborhood"})


def _get_data_freshness_caption() -> str:
    """Best-effort freshness signal for neighborhood metrics."""
    processed_path = Path("data/processed/neighborhood_stats.csv")
    file_part = "Processed CSV not found"
    if processed_path.exists():
        mtime = datetime.fromtimestamp(
            processed_path.stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
        file_part = f"Neighborhood metrics file updated: {mtime}"

    registry_part = "No ingestion history in raw_file_registry"
    try:
        recent_ingest = run_safe_sql(
            """
            SELECT MAX(ingested_at) AS latest_ingested_at
            FROM raw_file_registry
            """
        )
        if not recent_ingest.empty and recent_ingest.loc[0, "latest_ingested_at"]:
            registry_part = (
                f"Latest source ingest: {recent_ingest.loc[0, 'latest_ingested_at']}"
            )
    except Exception:
        registry_part = "Ingestion history unavailable"

    return f"{file_part}. {registry_part}."


def _render_compare_tab() -> None:
    st.subheader("Compare neighborhoods")
    st.caption(
        "Pick two to four neighborhoods and see how they stack up across the "
        "renter-fit dimensions."
    )

    df = _load_neighborhood_scoreboard()
    if df.empty or "neighborhood" not in df.columns:
        st.info("No neighborhood data yet. Use Demo controls to rebuild the demo database.")
        return

    available = sorted(df["neighborhood"].dropna().unique().tolist())
    if len(available) < 2:
        st.info("Need at least two neighborhoods to compare.")
        return

    default_picks = [n for n in ("Astoria", "Harlem", "Williamsburg") if n in available]
    if len(default_picks) < 2:
        default_picks = available[: min(3, len(available))]

    selected = st.multiselect(
        "Neighborhoods",
        options=available,
        default=default_picks,
        max_selections=4,
        help="Choose 2 to 4 neighborhoods.",
    )
    if len(selected) < 2:
        st.info("Pick at least two neighborhoods to see a comparison.")
        return

    subset = df[df["neighborhood"].isin(selected)].copy()

    available_dims = [(c, l) for c, l in SCORE_DIMENSIONS if c in subset.columns]
    if not available_dims:
        st.info("Score columns are missing from neighborhood_stats.")
        return

    st.caption(_get_data_freshness_caption())
    with st.expander("Metric definitions", expanded=False):
        for col, label in available_dims:
            definition = SCORE_DEFINITIONS.get(col, "Composite neighborhood score.")
            st.markdown(f"- **{label}**: {definition}")

    # KPI tile grid: one row per neighborhood, four key metrics each.
    st.markdown("#### At a glance")
    for _, row in subset.iterrows():
        st.markdown(f"**{row['neighborhood']}**")
        cols = st.columns(4)
        rent = row.get("median_listing_rent")
        cols[0].metric(
            "Median rent",
            f"${rent:,.0f}" if pd.notna(rent) else "-",
        )
        for i, (col, label) in enumerate(
            [
                ("renter_fit_score", "Renter fit"),
                ("safety_score", "Safety"),
                ("transit_score", "Transit"),
            ]
        ):
            val = row.get(col)
            cols[i + 1].metric(
                label,
                f"{val:.0f}" if pd.notna(val) else "-",
            )

    # Radar chart (Plotly Scatterpolar) overlaying each selected neighborhood.
    st.markdown("#### Score profile")
    dim_cols = [c for c, _ in available_dims]
    dim_labels = [l for _, l in available_dims]

    fig = go.Figure()
    for _, row in subset.iterrows():
        values = [row.get(c) if pd.notna(row.get(c)) else 0 for c in dim_cols]
        # Close the polygon by repeating the first value/label.
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=dim_labels + [dim_labels[0]],
                fill="toself",
                name=str(row["neighborhood"]),
                opacity=0.55,
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor="#121b2f",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(148,163,184,0.2)",
                linecolor="rgba(148,163,184,0.25)",
                tickfont=dict(color="#cbd5e1"),
            ),
            angularaxis=dict(
                gridcolor="rgba(148,163,184,0.12)",
                linecolor="rgba(148,163,184,0.22)",
                tickfont=dict(color="#cbd5e1"),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(15,23,42,0.5)",
            bordercolor="rgba(148,163,184,0.28)",
            borderwidth=1,
        ),
        margin=dict(t=30, l=20, r=20, b=20),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Underlying numbers", expanded=False):
        display_cols = ["neighborhood"] + (
            ["borough"] if "borough" in subset.columns else []
        )
        if "median_listing_rent" in subset.columns:
            display_cols.append("median_listing_rent")
        display_cols += dim_cols
        st.dataframe(
            subset[display_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


with tab_compare:
    _render_compare_tab()


# ----------------------------------------------------------------------------
# Tab: Top neighborhoods
# ----------------------------------------------------------------------------


def _render_rankings_tab() -> None:
    st.subheader("Top neighborhoods")
    st.caption(
        "Rank every neighborhood in the database by the metric you care about most."
    )

    df = _load_neighborhood_scoreboard()
    if df.empty or "neighborhood" not in df.columns:
        st.info("No neighborhood data yet. Use Demo controls to rebuild the demo database.")
        return

    available_dims = [(c, l) for c, l in SCORE_DIMENSIONS if c in df.columns]
    if not available_dims:
        st.info("Score columns are missing from neighborhood_stats.")
        return

    labels = [l for _, l in available_dims]
    cols = [c for c, _ in available_dims]
    default_idx = cols.index("renter_fit_score") if "renter_fit_score" in cols else 0

    chosen_label = st.selectbox(
        "Rank by",
        options=labels,
        index=default_idx,
    )
    chosen_col = cols[labels.index(chosen_label)]
    metric_definition = SCORE_DEFINITIONS.get(
        chosen_col, "Composite neighborhood score."
    )
    st.caption(f"{chosen_label}: {metric_definition}")
    st.caption(
        "Rankings sort highest to lowest, exclude rows with missing values, and may "
        "show ties with identical scores."
    )
    st.caption(_get_data_freshness_caption())

    ranked = (
        df[["neighborhood", chosen_col]]
        .dropna(subset=[chosen_col])
        .sort_values(chosen_col, ascending=False)
        .reset_index(drop=True)
    )
    if ranked.empty:
        st.info(f"No values found for {chosen_label}.")
        return

    st.markdown("#### Top 5")
    top5 = ranked.head(5)
    card_cols = st.columns(len(top5))
    for i, (_, row) in enumerate(top5.iterrows()):
        card_cols[i].metric(
            row["neighborhood"],
            f"{row[chosen_col]:.0f}",
        )

    st.markdown("#### Full ranking")
    spec = {
        "chart_type": "bar",
        "x": "neighborhood",
        "y": chosen_col,
        "color": None,
        "title": f"{chosen_label} by neighborhood",
    }
    fig = render_chart(ranked, spec)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full ranking table", expanded=False):
        display = ranked.rename(columns={chosen_col: chosen_label})
        st.dataframe(display, use_container_width=True, hide_index=True)


with tab_rankings:
    _render_rankings_tab()


# ----------------------------------------------------------------------------
# Tab: Methodology
# ----------------------------------------------------------------------------


with tab_methodology:
    st.subheader("How AI-partments works")
    st.markdown(
        """
- **Data scope:** Demo mode is synthetic unless you ingest CSV/XLSX files in the NYC Data Pipeline tab.
- **LLM boundary:** LLMs generate intent/SQL/recommendation text, but never execute queries directly.
- **SQL safety:** A deterministic validator allows only read-only `SELECT`/`WITH` against known tables/columns.
- **HPD interpretation:** `hpd_unit_count` is a housing-stock signal, not live vacancy.
- **Fallback policy:** If a source dataset is missing, affected scores fall back to neutral `50.0`.
- **Access signals:** Transit and amenity metrics represent access/proximity, not guaranteed service quality.

### Score weights

`amenity_score = 0.30*transit + 0.25*green_space + 0.20*public_service + 0.15*worship + 0.10*grocery`
(weights are redistributed if worship/grocery datasets are absent).

`availability_score = 0.60*listings_norm + 0.25*housing_supply + 0.15*affordable_housing_signal`
(falls back to `0.70*housing_supply + 0.30*affordable_housing_signal` when there are no live listings).

`renter_fit_score = 0.30*affordability + 0.25*safety + 0.25*amenity + 0.20*availability`

`value_score = 0.50*affordability + 0.30*safety + 0.20*amenity`

`safety_affordability_score = 0.5*affordability + 0.5*safety`

### Data coverage and known limitations

- Current rankings rely on the latest `neighborhood_stats` materialization in DuckDB/processed CSV.
- Crime and transit scores are neighborhood-level aggregates, not building-level guarantees.
- Availability blends listing volume and HPD housing-stock signals; it is not a real-time vacancy feed.
- Neighborhood comparisons are best used as directional guidance, then validated with current listings.
        """
    )
    st.caption(_get_data_freshness_caption())


# ----------------------------------------------------------------------------
# Tab: About
# ----------------------------------------------------------------------------


with tab_about:
    st.subheader("About AI-partments")
    st.markdown(
        """
AI-partments is a renter decision-support dashboard built on local analytics + AI assistance.
It blends apartment listings, neighborhood metrics, transit access, parks, public facilities,
and housing-stock signals in DuckDB so users can compare areas and query data with natural language.

### The agent pipeline

1. **Router (AI-assisted):** classifies intent and extracts structured filters.
2. **Schema selection (deterministic):** constrains table/column choices to the canonical schema.
3. **SQL generation (AI-assisted):** drafts query text for a read-only data request.
4. **Safety validation (deterministic):** enforces one safe `SELECT`/`WITH` statement.
5. **Execution (deterministic):** runs validated SQL in DuckDB.
6. **Visualization + explanation (mixed):** charting is deterministic; narrative recommendations are AI-assisted.

### Deterministic vs AI-assisted

- **Deterministic core:** schema guardrails, SQL safety checks, query execution, score math, chart rendering.
- **AI-assisted layers:** intent parsing, SQL drafting, user-facing recommendations.
- **Result:** AI adds flexibility while deterministic layers protect correctness and safety.

### Stack

- App/UI: Streamlit + Plotly + PyDeck
- Data/compute: DuckDB + pandas + numpy
- LLM options: local Ollama or IBM Granite (watsonx.ai), with mock mode for offline reproducibility
        """
    )


# ----------------------------------------------------------------------------
# Tab: NYC Data Pipeline (only when the demo toggle is on)
# ----------------------------------------------------------------------------


def _render_profile(profile: dict[str, Any]) -> None:
    if "error" in profile:
        st.error(profile["error"])
        return
    cols = st.columns(4)
    cols[0].metric("Rows", f"{profile.get('row_count', 0):,}")
    cols[1].metric("Columns", profile.get("column_count", 0))
    cols[2].metric("Detected dataset", profile.get("dataset_type") or "-")
    cols[3].metric("Has lat/lng", "yes" if profile.get("has_lat_lng") else "no")
    if profile.get("warnings"):
        for w in profile["warnings"]:
            st.warning(w)
    with st.expander("Important columns"):
        st.json(profile.get("important_columns", {}))
    with st.expander("Sample rows"):
        sample = profile.get("sample_rows", [])
        if sample:
            st.dataframe(pd.DataFrame(sample), use_container_width=True)
    if profile.get("missing_value_summary"):
        with st.expander("Missing value summary"):
            st.json(profile["missing_value_summary"])


def _render_pipeline_tab() -> None:
    st.subheader("Real NYC data ingestion")
    st.caption(
        "Drop CSV/XLSX/XLS files into data/raw/ (or upload below). The pipeline "
        "auto-detects datasets, cleans them, matches them to neighborhoods, and "
        "loads them into DuckDB. For very large CSVs, this app uses DuckDB "
        "read_csv_auto where possible instead of loading everything into pandas."
    )

    pipeline = IngestionPipeline()

    st.markdown("### Upload files")
    uploaded = st.file_uploader(
        "Upload one or more CSV/XLSX/XLS files",
        accept_multiple_files=True,
        type=["csv", "xlsx", "xls"],
    )
    auto_detect = st.checkbox("Auto-detect dataset type from filename", value=True)
    manual_dt = st.selectbox(
        "Dataset type (used when auto-detect is off)",
        [
            "apartments",
            "demographics",
            "crime",
            "transit",
            "transit_bus",
            "transit_subway",
            "amenities",
            "parks",
            "worship",
            "housing",
            "facilities",
            "population",
        ],
        index=0,
        disabled=auto_detect,
    )

    col_p, col_i = st.columns(2)
    if col_p.button("Profile uploaded files", disabled=not uploaded):
        for upl in uploaded:
            saved_path = save_uploaded_file(upl)
            dt = None if auto_detect else manual_dt
            with st.expander(f"Profile: {upl.name}", expanded=True):
                profile = pipeline.profile_csv(saved_path, dataset_type=dt)
                _render_profile(profile)

    if col_i.button("Ingest uploaded files", disabled=not uploaded, type="primary"):
        progress = st.progress(0.0, text="Starting ingestion...")
        for idx, upl in enumerate(uploaded, start=1):
            saved_path = save_uploaded_file(upl)
            dt = None if auto_detect else manual_dt
            progress.progress(idx / max(1, len(uploaded)), text=f"Ingesting {upl.name}")
            res = pipeline.ingest_csv(saved_path, dataset_type=dt)
            with st.expander(f"Result: {upl.name}", expanded=False):
                st.json(res)
        progress.empty()
        rebuild = pipeline.rebuild_processed_tables()
        st.success(f"Ingestion complete. neighborhood_stats now has {rebuild['rows']} rows.")
        for w in rebuild.get("warnings", []) or []:
            st.warning(w)
        _refresh_token()

    st.divider()

    st.markdown("### data/raw folder")
    files = list_raw_files()
    if not files:
        st.caption("No files found under data/raw/. Drop CSV/XLSX/XLS files into the subfolders.")
    else:
        rows: list[dict[str, Any]] = []
        for f in files:
            rel = f.relative_to(config.RAW_DIR)
            try:
                size_mb = f.stat().st_size / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            rows.append({"path": str(rel), "size_mb": round(size_mb, 2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    col_pr, col_in, col_rb = st.columns(3)
    if col_pr.button("Profile raw folder", disabled=not files):
        for f in files:
            with st.expander(f"Profile: {f.name}", expanded=False):
                _render_profile(pipeline.profile_csv(f))

    if col_in.button("Ingest all raw files", disabled=not files):
        progress = st.progress(0.0, text="Starting ingestion...")
        for idx, f in enumerate(files, start=1):
            progress.progress(idx / max(1, len(files)), text=f"Ingesting {f.name}")
            res = pipeline.ingest_csv(f)
            with st.expander(f"Result: {f.name}", expanded=False):
                st.json(res)
        progress.empty()
        rebuild = pipeline.rebuild_processed_tables()
        st.success(f"All raw files ingested. neighborhood_stats has {rebuild['rows']} rows.")
        for w in rebuild.get("warnings", []) or []:
            st.warning(w)
        _refresh_token()

    if col_rb.button("Rebuild metrics only"):
        rebuild = pipeline.rebuild_processed_tables()
        st.success(f"Rebuilt neighborhood_stats with {rebuild['rows']} rows.")
        for w in rebuild.get("warnings", []) or []:
            st.warning(w)
        _refresh_token()

    st.divider()

    st.markdown("### Processed table row counts")
    counts = {
        "neighborhoods": get_table_row_count("neighborhoods"),
        "neighborhood_stats": get_table_row_count("neighborhood_stats"),
        "apartments": get_table_row_count("apartments"),
        "amenities": get_table_row_count("amenities"),
        "crime_events": get_table_row_count("crime_events"),
        "hpd_buildings": get_table_row_count("hpd_buildings"),
        "facilities": get_table_row_count("facilities"),
        "parks": get_table_row_count("parks"),
        "transit_bus_stops": get_table_row_count("transit_bus_stops"),
        "transit_subway_stations": get_table_row_count("transit_subway_stations"),
        "raw_file_registry": get_table_row_count("raw_file_registry"),
    }
    st.dataframe(
        pd.DataFrame([{"table": k, "rows": v} for k, v in counts.items()]),
        use_container_width=True,
        hide_index=True,
    )

    if get_table_row_count("raw_file_registry") > 0:
        st.markdown("### Recent ingestions")
        st.dataframe(
            _cached_table_preview("raw_file_registry", 100, st.session_state["cache_token"]),
            use_container_width=True,
            hide_index=True,
        )

    if get_table_row_count("neighborhood_stats") > 0:
        st.markdown("### neighborhood_stats preview")
        st.dataframe(
            _cached_table_preview("neighborhood_stats", 50, st.session_state["cache_token"]),
            use_container_width=True,
            height=240,
        )


if tab_pipeline is not None:
    with tab_pipeline:
        _render_pipeline_tab()
