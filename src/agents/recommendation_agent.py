"""Recommendation agent: short renter-friendly explanation of the result."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.llm_client import LLMClient


SYSTEM_PROMPT = (
    "You are a friendly NYC renter assistant. Given a small results table from "
    "a renter intelligence dashboard, write a 3-5 sentence summary that "
    "highlights the top matches, surfaces a tradeoff, and includes a brief "
    "uncertainty disclaimer. Do not invent numbers - reuse the values that "
    "appear in the table. Do not use markdown."
)


def _top_neighborhoods(df: pd.DataFrame) -> list[str]:
    for col in ("neighborhood", "name"):
        if col in df.columns:
            return [str(v) for v in df[col].head(3).tolist()]
    return []


def _format_row_summary(df: pd.DataFrame) -> str:
    """Stringify the top 5 rows for prompt context."""
    if df.empty:
        return "(empty result)"
    sample = df.head(5)
    lines: list[str] = []
    for _, row in sample.iterrows():
        parts = []
        for col, value in row.items():
            if pd.isna(value):
                continue
            parts.append(f"{col}={value}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def _mock_recommendation(question: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return (
            "No matches were found for your question. Try loosening the filters "
            "(higher max rent, broader bedroom count, or fewer required amenities). "
            "Demo data is synthetic, so live market conditions will differ."
        )

    tops = _top_neighborhoods(df)
    bullets: list[str] = []
    columns = df.columns

    if "renter_fit_score" in columns and tops:
        first = df.iloc[0]
        bullets.append(
            f"{first.get('neighborhood', tops[0])} leads on renter_fit_score "
            f"({first.get('renter_fit_score')})."
        )
    if "rent" in columns and "neighborhood" in columns:
        cheapest = df.sort_values("rent").iloc[0]
        bullets.append(
            f"The most affordable listing is {cheapest.get('rent')} in "
            f"{cheapest.get('neighborhood')} ({cheapest.get('bedrooms', '?')}-bed)."
        )
    if "transit_score" in columns and tops:
        best_transit = df.sort_values("transit_score", ascending=False).iloc[0]
        nb = best_transit.get("neighborhood") or best_transit.get("name") or tops[0]
        bullets.append(
            f"Best transit access goes to {nb} (transit_score={best_transit.get('transit_score')})."
        )
    if "green_space_score" in columns and tops:
        best_green = df.sort_values("green_space_score", ascending=False).iloc[0]
        nb = best_green.get("neighborhood") or best_green.get("name") or tops[0]
        bullets.append(
            f"For parks/green space, {nb} stands out "
            f"(green_space_score={best_green.get('green_space_score')})."
        )
    if "hpd_unit_count" in columns and tops:
        most_hpd = df.sort_values("hpd_unit_count", ascending=False).iloc[0]
        nb = most_hpd.get("neighborhood") or most_hpd.get("name") or tops[0]
        bullets.append(
            f"{nb} carries the largest HPD housing-stock signal "
            f"(units={most_hpd.get('hpd_unit_count')})."
        )

    if not bullets and tops:
        bullets.append(f"Top matches: {', '.join(tops)}.")

    disclaimer = (
        "These results come from the local AI-partments dataset (demo or ingested CSVs); "
        "live availability and prices may differ."
    )
    return " ".join(bullets + [disclaimer])


def recommendation_agent(
    question: str,
    df: pd.DataFrame,
    chart_spec: dict[str, Any],
    llm: Optional[LLMClient] = None,
) -> str:
    """Return a short natural-language explanation of the results."""
    if llm is None or llm.provider == "mock":
        return _mock_recommendation(question, df)

    summary_rows = _format_row_summary(df)
    prompt = (
        f"User question:\n{question}\n\n"
        f"Top result rows:\n{summary_rows}\n\n"
        f"Chart context: {chart_spec.get('title')}\n\n"
        "Write a 3-5 sentence renter-friendly explanation."
    )
    try:
        result = llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.2)
        if result.text and not result.fell_back:
            return result.text.strip()
    except Exception:
        pass
    return _mock_recommendation(question, df)
