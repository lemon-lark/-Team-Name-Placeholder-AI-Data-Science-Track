"""Visualization agent: pick a chart spec from a result DataFrame."""
from __future__ import annotations

from typing import Any

import pandas as pd


SCORE_COLUMNS = (
    "renter_fit_score",
    "availability_score",
    "green_space_score",
    "transit_score",
    "safety_score",
    "amenity_score",
    "affordability_score",
    "value_score",
    "subway_access_score",
    "bus_access_score",
    "public_service_score",
)


def visualization_agent(question: str, df: pd.DataFrame) -> dict[str, Any]:
    """Choose a chart for the result DataFrame.

    Returns a dict ``{chart_type, x, y, color, title}`` consumable by
    ``src.tools.chart_tools.render_chart``. ``chart_type`` is one of
    ``bar | scatter | line | map | table_only``.
    """
    if df is None or df.empty:
        return {"chart_type": "table_only", "title": "No results"}

    columns = list(df.columns)
    has_lat = "lat" in columns
    has_lng = "lng" in columns
    has_center_lat = "center_lat" in columns
    has_center_lng = "center_lng" in columns
    has_neighborhood = "neighborhood" in columns or "name" in columns
    nb_col = "neighborhood" if "neighborhood" in columns else ("name" if "name" in columns else None)

    score_cols = [c for c in SCORE_COLUMNS if c in columns]
    has_rent = "rent" in columns or "median_listing_rent" in columns or "avg_rent" in columns
    rent_col = (
        "rent"
        if "rent" in columns
        else ("median_listing_rent" if "median_listing_rent" in columns else ("avg_rent" if "avg_rent" in columns else None))
    )

    # Apartments - rent vs safety scatter.
    if has_rent and "safety_score" in columns and len(df) > 1:
        return {
            "chart_type": "scatter",
            "x": rent_col,
            "y": "safety_score",
            "color": "neighborhood" if has_neighborhood else None,
            "title": "Rent vs. Safety Score",
        }

    # Apartments / neighborhoods with rent and transit.
    if rent_col and "transit_score" in columns and len(df) > 1:
        return {
            "chart_type": "scatter",
            "x": rent_col,
            "y": "transit_score",
            "color": "neighborhood" if has_neighborhood else None,
            "title": "Rent vs. Transit Score",
        }

    # Neighborhood ranking by renter_fit_score / availability / green space.
    for score in (
        "renter_fit_score",
        "availability_score",
        "green_space_score",
        "transit_score",
        "safety_score",
        "amenity_score",
        "affordability_score",
    ):
        if score in columns and nb_col:
            pretty = score.replace("_", " ").title()
            return {
                "chart_type": "bar",
                "x": nb_col,
                "y": score,
                "color": "borough" if "borough" in columns else None,
                "title": f"{pretty} by Neighborhood",
            }

    # If we only have a numeric and a label column, draw a bar chart.
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if nb_col and numeric_cols:
        y = numeric_cols[0]
        return {
            "chart_type": "bar",
            "x": nb_col,
            "y": y,
            "color": None,
            "title": f"{y.replace('_', ' ').title()} by {nb_col.title()}",
        }

    # If only lat/lng available, return map type and let the map tool render.
    if (has_lat and has_lng) or (has_center_lat and has_center_lng):
        return {"chart_type": "map", "title": "Locations"}

    return {"chart_type": "table_only", "title": "Results"}
