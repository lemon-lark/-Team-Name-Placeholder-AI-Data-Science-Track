"""Plotly chart rendering keyed off the visualization agent's chart spec."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#121b2f",
    font=dict(color="#e2e8f0"),
    title=dict(font=dict(color="#f8fafc", size=20)),
    legend=dict(
        bgcolor="rgba(15,23,42,0.5)",
        bordercolor="rgba(148,163,184,0.28)",
        borderwidth=1,
    ),
    margin=dict(t=60, l=20, r=20, b=60),
)


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**DARK_CHART_THEME)
    fig.update_xaxes(
        showgrid=False,
        linecolor="rgba(148,163,184,0.35)",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#cbd5e1"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#cbd5e1"),
    )
    return fig


def render_chart(df: pd.DataFrame, chart_spec: dict[str, Any]) -> Optional[go.Figure]:
    """Build a Plotly figure for ``df`` according to ``chart_spec``.

    Returns ``None`` if no chart should be rendered (table_only or empty df).
    """
    if df is None or df.empty:
        return None
    chart_type = (chart_spec or {}).get("chart_type", "table_only")
    title = (chart_spec or {}).get("title", "")
    x = chart_spec.get("x")
    y = chart_spec.get("y")
    color = chart_spec.get("color")

    if chart_type == "table_only" or chart_type == "map":
        return None

    if not x or not y or x not in df.columns or y not in df.columns:
        return None

    try:
        if chart_type == "bar":
            data = df[[x, y] + ([color] if color and color in df.columns else [])].dropna(subset=[y])
            if pd.api.types.is_numeric_dtype(data[y]):
                data = data.sort_values(y, ascending=False).head(40)
            fig = px.bar(
                data,
                x=x,
                y=y,
                color=color if color and color in df.columns else None,
                title=title,
                color_discrete_sequence=px.colors.sequential.Plasma,
            )
            fig.update_layout(xaxis_tickangle=-30, margin=dict(t=60, l=20, r=20, b=80))
            return _apply_dark_theme(fig)

        if chart_type == "scatter":
            data = df[[x, y] + ([color] if color and color in df.columns else [])].dropna(subset=[x, y])
            fig = px.scatter(
                data,
                x=x,
                y=y,
                color=color if color and color in df.columns else None,
                title=title,
                hover_data=[c for c in df.columns if c not in (x, y)][:6],
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_traces(marker=dict(size=10, opacity=0.85))
            fig.update_layout(margin=dict(t=60, l=20, r=20, b=40))
            return _apply_dark_theme(fig)

        if chart_type == "line":
            fig = px.line(
                df,
                x=x,
                y=y,
                color=color if color and color in df.columns else None,
                title=title,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_traces(line=dict(width=3))
            return _apply_dark_theme(fig)
    except Exception:
        return None

    return None
