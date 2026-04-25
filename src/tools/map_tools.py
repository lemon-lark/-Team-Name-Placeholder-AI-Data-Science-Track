"""pydeck map rendering for result dataframes."""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

try:
    import pydeck as pdk
except Exception:  # pragma: no cover - pydeck may not be present in headless runs
    pdk = None  # type: ignore


# Default viewport centered on NYC for empty / no-lat-lng states.
NYC_DEFAULT_LAT = 40.7128
NYC_DEFAULT_LNG = -74.0060
NYC_DEFAULT_ZOOM = 9.6


# Columns we surface in the tooltip when present.
TOOLTIP_FIELDS = (
    "neighborhood",
    "name",
    "address",
    "rent",
    "bedrooms",
    "bathrooms",
    "sqft",
    "available_date",
    "type",
    "amenity_name",
    "station_name",
    "stop_name",
    "facility_type",
    "park_type",
    "borough",
    "renter_fit_score",
    "safety_score",
    "transit_score",
    "green_space_score",
    "nearest_subway_distance",
    "nearest_bus_stop_distance",
    "hpd_violation_count",
)

DARK_MAP_TOOLTIP_STYLE = {
    "backgroundColor": "#0f172a",
    "color": "#e2e8f0",
    "fontSize": "12px",
    "borderRadius": "8px",
    "border": "1px solid rgba(96, 165, 250, 0.45)",
}


def _prepare_points(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return a copy of ``df`` with ``lat``/``lng`` columns suitable for pydeck."""
    if df is None or df.empty:
        return None
    df = df.copy()
    if "lat" not in df.columns and "center_lat" in df.columns:
        df["lat"] = df["center_lat"]
    if "lng" not in df.columns and "center_lng" in df.columns:
        df["lng"] = df["center_lng"]
    if "lat" not in df.columns or "lng" not in df.columns:
        return None
    df = df.dropna(subset=["lat", "lng"])
    if df.empty:
        return None
    return df


def _build_tooltip(
    df: pd.DataFrame, tooltip_fields: Sequence[str] | None = None
) -> dict:
    candidate_fields = list(tooltip_fields) if tooltip_fields else list(TOOLTIP_FIELDS)
    fields = [c for c in candidate_fields if c in df.columns]
    if not fields:
        return {"html": "<b>{lat}, {lng}</b>"}
    rows = "<br/>".join(
        f"<b>{c}:</b> {{{c}}}" for c in fields
    )
    return {
        "html": rows,
        "style": DARK_MAP_TOOLTIP_STYLE,
    }


def build_pydeck_layer(
    df: pd.DataFrame, tooltip_fields: Sequence[str] | None = None
):
    """Build a pydeck Deck object from a DataFrame with ``lat``/``lng``.

    Returns ``None`` if the DataFrame has no usable coordinates.
    """
    if pdk is None:
        return None
    points = _prepare_points(df)
    if points is None:
        return None
    center_lat = float(points["lat"].mean())
    center_lng = float(points["lng"].mean())

    tooltip = _build_tooltip(points, tooltip_fields=tooltip_fields)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[lng, lat]",
        get_radius=110,
        get_fill_color=[56, 189, 248, 214],
        pickable=True,
        opacity=0.9,
        radius_min_pixels=4,
        radius_max_pixels=15,
    )
    view = pdk.ViewState(
        longitude=center_lng,
        latitude=center_lat,
        zoom=11,
        pitch=0,
    )
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_provider="carto",
        map_style="dark",
    )


def render_map(df: pd.DataFrame):
    """Alias for ``build_pydeck_layer`` (for symmetry with ``render_chart``)."""
    return build_pydeck_layer(df)


def build_overview_deck() -> Optional["pdk.Deck"]:
    """Default NYC overview deck plotting every neighborhood centroid.

    Used as the empty state for the Find a place tab so the map area is always
    visible. Falls back to a plain NYC viewport when neighborhood centroids
    are unavailable.
    """
    if pdk is None:
        return None

    df: Optional[pd.DataFrame] = None
    try:
        # Local import to avoid a hard module-level dependency on the database
        # layer in headless / unit-test contexts.
        from src.database import run_query

        df = run_query(
            "SELECT name AS neighborhood, borough, "
            "center_lat AS lat, center_lng AS lng "
            "FROM neighborhoods "
            "WHERE center_lat IS NOT NULL AND center_lng IS NOT NULL"
        )
    except Exception:
        df = None

    layers = []
    center_lat = NYC_DEFAULT_LAT
    center_lng = NYC_DEFAULT_LNG
    tooltip: Optional[dict] = None

    if df is not None and not df.empty:
        center_lat = float(df["lat"].mean())
        center_lng = float(df["lng"].mean())
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lng, lat]",
                get_radius=180,
                get_fill_color=[129, 140, 248, 186],
                pickable=True,
                opacity=0.75,
                radius_min_pixels=4,
                radius_max_pixels=11,
            )
        )
        tooltip = {
            "html": "<b>{neighborhood}</b><br/>{borough}",
            "style": DARK_MAP_TOOLTIP_STYLE,
        }

    view = pdk.ViewState(
        longitude=center_lng,
        latitude=center_lat,
        zoom=NYC_DEFAULT_ZOOM,
        pitch=0,
    )
    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip=tooltip or {"html": "", "style": {}},
        map_provider="carto",
        map_style="dark",
    )
