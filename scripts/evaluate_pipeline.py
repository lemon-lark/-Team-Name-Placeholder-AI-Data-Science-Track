"""Static evaluation harness for the AI-partments agentic pipeline.

Runs a curated set of questions through ``answer_user_question`` and
asserts properties of every pipeline stage (router -> clarification ->
schema -> SQL -> safety -> execution -> visualization -> recommendation).

Outputs:

* ``reports/eval_<timestamp>.json`` - full per-case payload (one record per case).
* ``reports/eval_<timestamp>.csv``  - flat summary suitable for diffing in CI.

Usage:

    python scripts/evaluate_pipeline.py [--provider mock|ollama|watsonx]

Exit code is ``0`` when every case passed, ``1`` otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents import answer_user_question  # noqa: E402
from src.agents.clarification_agent import count_dimensions  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.schema import ALLOWED_TABLES  # noqa: E402


REPORTS_DIR = ROOT / "reports"


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    question: str
    expects_clarification: bool = False
    expected_intent: str | None = None
    min_rows: int = 0
    expected_chart_types: tuple[str, ...] = ()
    expected_filters: dict[str, Any] = field(default_factory=dict)
    expected_proximity: list[dict[str, Any]] = field(default_factory=list)
    expected_sql_substrings: tuple[str, ...] = ()
    expected_required_tables: tuple[str, ...] = ()
    expected_distance_column: str | None = None
    expected_distance_max: float | None = None
    expect_safety_approved: bool = True
    require_recommendation: bool = True


CASES: list[Case] = [
    Case(
        name="rental_search_full",
        question=(
            "Find 1-bedroom apartments under $2,500 in low-crime neighborhoods "
            "near parks."
        ),
        expects_clarification=False,
        expected_intent="rental_search",
        min_rows=1,
        expected_chart_types=("scatter", "bar"),
        expected_filters={
            "max_rent": 2500,
            "exact_bedrooms": 1,
            "amenities_contains": "park",
            "safety_preference": "low_crime",
        },
        expected_sql_substrings=("apartments", "WHERE", "rent <="),
        expected_required_tables=("apartments", "neighborhood_stats"),
    ),
    Case(
        name="under_specified_show_apartments",
        question="Show apartments.",
        expects_clarification=True,
    ),
    Case(
        name="under_specified_price_only",
        question="Apartments under $2500.",
        expects_clarification=True,
    ),
    Case(
        name="two_dim_price_location",
        question="Apartments under $2500 in Astoria.",
        expects_clarification=False,
        expected_intent="rental_search",
        expected_filters={"max_rent": 2500, "neighborhoods_contains": "Astoria"},
        expected_sql_substrings=("apartments", "Astoria", "rent <="),
    ),
    Case(
        name="amenity_proximity_mosque",
        question="Within 0.5 mile of a mosque, under $3000.",
        expects_clarification=False,
        min_rows=1,
        expected_filters={"max_rent": 3000},
        expected_proximity=[
            {"target": "mosque", "max_distance_miles": 0.5, "kind": "amenity"},
        ],
        expected_sql_substrings=("EXISTS", "amenities", "distance_miles", "mosque"),
        expected_required_tables=("amenities",),
    ),
    Case(
        name="transit_proximity_subway_default",
        question="1-bedroom apartments near subway under $2800.",
        expects_clarification=False,
        expected_intent="rental_search",
        min_rows=0,  # 0.25 mi is tight - allow zero rows but distance bound still asserted
        expected_filters={"max_rent": 2800, "exact_bedrooms": 1},
        expected_proximity=[
            {"target": "subway", "max_distance_miles": 0.25, "kind": "transit"},
        ],
        expected_sql_substrings=(
            "transit_subway_stations",
            "distance_to_subway_miles",
            "RADIANS",
        ),
        expected_required_tables=("transit_subway_stations", "apartments"),
        expected_distance_column="distance_to_subway_miles",
        expected_distance_max=0.25,
    ),
    Case(
        name="transit_proximity_bus_with_location",
        question="Apartments close to a bus stop in Astoria.",
        expects_clarification=False,
        expected_intent="rental_search",
        min_rows=0,
        expected_filters={"neighborhoods_contains": "Astoria"},
        expected_proximity=[
            {"target": "bus", "max_distance_miles": 0.25, "kind": "transit"},
        ],
        expected_sql_substrings=(
            "transit_bus_stops",
            "distance_to_bus_miles",
            "Astoria",
        ),
        expected_required_tables=("transit_bus_stops", "apartments"),
        expected_distance_column="distance_to_bus_miles",
        expected_distance_max=0.25,
    ),
    Case(
        name="comparison_two_neighborhoods",
        question="Compare Astoria and Harlem.",
        expects_clarification=False,
        expected_intent="neighborhood_comparison",
        min_rows=1,
        expected_filters={"neighborhoods_contains": "Astoria"},
        expected_sql_substrings=("Astoria", "Harlem", "neighborhood_stats"),
    ),
    Case(
        name="ranking_only",
        question="Rank neighborhoods by renter_fit_score.",
        expects_clarification=False,
        expected_intent="neighborhood_comparison",
        min_rows=1,
        expected_chart_types=("scatter", "bar"),
        expected_sql_substrings=("renter_fit_score", "ORDER BY"),
    ),
    # Existing canonical questions from the smoke test set.
    Case(
        name="smoke_compare_three",
        question=(
            "Compare Astoria, Harlem, and Flushing by transit, parks, "
            "and housing supply."
        ),
        expects_clarification=False,
        expected_intent="neighborhood_comparison",
        min_rows=1,
    ),
    Case(
        name="smoke_transit_affordability",
        question="Which neighborhoods have the best combination of transit and affordability?",
        expects_clarification=False,
        min_rows=1,
    ),
    Case(
        name="smoke_high_availability",
        question="Show neighborhoods with high availability score and good green space.",
        expects_clarification=False,
        min_rows=1,
    ),
    Case(
        name="smoke_hpd",
        question="Which neighborhoods have lots of HPD housing stock?",
        expects_clarification=False,
        expected_intent="housing_supply_analysis",
        min_rows=1,
    ),
    Case(
        name="smoke_subway_affordable",
        question="Which neighborhoods have the best subway access and affordable rent?",
        expects_clarification=False,
        min_rows=1,
    ),
]


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _check_filters(
    expected: dict[str, Any], actual: dict[str, Any], failures: list[str]
) -> None:
    for key, want in expected.items():
        if key.endswith("_contains"):
            base = key.removesuffix("_contains")
            haystack = actual.get(base) or []
            if not isinstance(haystack, list):
                haystack = [haystack]
            if want not in haystack:
                failures.append(
                    f"filter[{base}] missing {want!r} (got {haystack!r})"
                )
        else:
            if actual.get(key) != want:
                failures.append(
                    f"filter[{key}] expected {want!r}, got {actual.get(key)!r}"
                )


def _check_proximity(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    failures: list[str],
) -> None:
    for want in expected:
        match = next(
            (
                p
                for p in actual
                if p.get("target") == want.get("target")
                and abs(
                    float(p.get("max_distance_miles") or 0)
                    - float(want.get("max_distance_miles") or 0)
                )
                < 1e-6
                and p.get("kind") == want.get("kind")
            ),
            None,
        )
        if match is None:
            failures.append(
                f"proximity missing entry {want!r} (got {actual!r})"
            )


def _check_sql_substrings(
    needles: tuple[str, ...], sql: str, failures: list[str]
) -> None:
    sql_l = sql.lower()
    for needle in needles:
        if needle.lower() not in sql_l:
            failures.append(f"sql missing substring {needle!r}")


def _check_required_tables(
    needed: tuple[str, ...], required_tables: list[str], failures: list[str]
) -> None:
    for table in needed:
        if table not in required_tables:
            failures.append(f"schema missing required table {table!r}")
    extras = [t for t in required_tables if t not in ALLOWED_TABLES]
    if extras:
        failures.append(f"schema includes disallowed tables: {extras}")


def _check_distance_column(
    column: str | None,
    max_distance: float | None,
    df,
    failures: list[str],
) -> None:
    if column is None:
        return
    if column not in df.columns:
        failures.append(f"result df missing distance column {column!r}")
        return
    series = df[column].dropna()
    if series.empty:
        return  # no rows; min_rows=0 was specified
    if series.isna().any():
        failures.append(f"distance column {column!r} contains NaN values")
    if max_distance is not None and (series > max_distance + 1e-6).any():
        failures.append(
            f"distance column {column!r} exceeded max {max_distance} "
            f"(max in result: {series.max():.4f})"
        )


def _check_chart(expected: tuple[str, ...], actual: str, failures: list[str]) -> None:
    if not expected:
        return
    if actual not in expected:
        failures.append(
            f"chart_type {actual!r} not in expected {list(expected)!r}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _stage_timer() -> Callable[[str], None]:
    """Return a closure used to record cumulative-then-incremental stage times."""
    start = time.perf_counter()
    last = start
    timings: dict[str, float] = {}

    def record(stage: str) -> None:
        nonlocal last
        now = time.perf_counter()
        timings[stage] = round((now - last) * 1000.0, 2)
        last = now

    record.timings = timings  # type: ignore[attr-defined]
    record.start = start  # type: ignore[attr-defined]
    return record


def run_case(case: Case, llm: LLMClient) -> dict[str, Any]:
    failures: list[str] = []
    started = time.perf_counter()
    try:
        result = answer_user_question(case.question, llm=llm)
        crashed: str | None = None
    except Exception as exc:  # pragma: no cover - safety net
        result = {}
        crashed = f"{type(exc).__name__}: {exc}"
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

    if crashed:
        failures.append(f"pipeline crashed: {crashed}")

    router_output = result.get("router_output", {}) or {}
    filters = router_output.get("filters", {}) or {}
    intent = router_output.get("intent")
    clar = result.get("clarification") or {}
    needs_clarification = bool(result.get("needs_clarification"))
    schema_context = result.get("schema_context") or {}
    required_tables = list(schema_context.get("required_tables") or [])
    sql = result.get("sql") or ""
    safety_result = result.get("safety_result") or {}
    df = result.get("dataframe")
    chart_spec = result.get("chart_spec") or {}
    chart_type = chart_spec.get("chart_type", "table_only")
    recommendation = result.get("recommendation") or ""
    provider_used = result.get("provider_used", "mock")

    if case.expects_clarification:
        if not needs_clarification:
            failures.append("expected clarification but pipeline ran SQL")
        if not clar.get("follow_up_text"):
            failures.append("clarification turn produced no follow_up_text")
    else:
        if needs_clarification:
            failures.append(
                f"unexpected clarification: {clar.get('follow_up_text')!r}"
            )
        if case.expected_intent and intent != case.expected_intent:
            failures.append(
                f"intent {intent!r} != expected {case.expected_intent!r}"
            )
        _check_filters(case.expected_filters, filters, failures)
        _check_proximity(
            case.expected_proximity, filters.get("proximity") or [], failures
        )
        _check_sql_substrings(case.expected_sql_substrings, sql, failures)
        _check_required_tables(
            case.expected_required_tables, required_tables, failures
        )
        if case.expect_safety_approved and not safety_result.get("approved"):
            failures.append(
                f"safety not approved: {safety_result.get('reason')}"
            )
        if df is not None and len(df) < case.min_rows:
            failures.append(
                f"result has {len(df)} rows; expected >= {case.min_rows}"
            )
        _check_distance_column(
            case.expected_distance_column,
            case.expected_distance_max,
            df if df is not None else None,
            failures,
        )
        _check_chart(case.expected_chart_types, chart_type, failures)
        if case.require_recommendation and not recommendation.strip():
            failures.append("recommendation was empty")

    supplied_dims = count_dimensions(filters, intent=intent)

    return {
        "case": case.name,
        "question": case.question,
        "passed": not failures,
        "failures": failures,
        "needs_clarification": needs_clarification,
        "intent": intent,
        "supplied_dimensions": supplied_dims,
        "filters": filters,
        "required_tables": required_tables,
        "sql": sql,
        "safety_approved": bool(safety_result.get("approved")),
        "safety_reason": safety_result.get("reason"),
        "row_count": int(len(df)) if df is not None else 0,
        "result_columns": list(df.columns) if df is not None else [],
        "chart_type": chart_type,
        "recommendation_length": len(recommendation),
        "provider_used": provider_used,
        "fell_back": bool(result.get("fell_back")),
        "exec_error": result.get("error"),
        "elapsed_ms_total": elapsed_ms,
        "follow_up_text": clar.get("follow_up_text"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        default="mock",
        choices=("mock", "ollama", "watsonx"),
        help="LLM provider for the run.",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(REPORTS_DIR),
        help="Directory to write JSON/CSV reports to.",
    )
    args = parser.parse_args(argv)

    llm = LLMClient(provider=args.provider)
    rows: list[dict[str, Any]] = []

    print(f"Running {len(CASES)} pipeline evaluation cases (provider={args.provider})\n")
    passed_count = 0
    for case in CASES:
        record = run_case(case, llm)
        rows.append(record)
        flag = "PASS" if record["passed"] else "FAIL"
        if record["passed"]:
            passed_count += 1
        print(f"[{flag}] {record['case']:38s} | {record['question']}")
        for f in record["failures"]:
            print(f"        - {f}")

    summary = {
        "total": len(rows),
        "passed": passed_count,
        "failed": len(rows) - passed_count,
        "provider": args.provider,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    print()
    print(
        f"{summary['passed']}/{summary['total']} passed, "
        f"{summary['failed']} failed."
    )

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"eval_{summary['timestamp']}.json"
    csv_path = reports_dir / f"eval_{summary['timestamp']}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "cases": rows}, fh, indent=2, default=str)

    csv_columns = [
        "case",
        "question",
        "passed",
        "needs_clarification",
        "intent",
        "supplied_dimensions",
        "row_count",
        "chart_type",
        "safety_approved",
        "safety_reason",
        "exec_error",
        "recommendation_length",
        "provider_used",
        "elapsed_ms_total",
        "failures",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "case": r["case"],
                    "question": r["question"],
                    "passed": r["passed"],
                    "needs_clarification": r["needs_clarification"],
                    "intent": r["intent"],
                    "supplied_dimensions": "|".join(r["supplied_dimensions"]),
                    "row_count": r["row_count"],
                    "chart_type": r["chart_type"],
                    "safety_approved": r["safety_approved"],
                    "safety_reason": r["safety_reason"],
                    "exec_error": r["exec_error"],
                    "recommendation_length": r["recommendation_length"],
                    "provider_used": r["provider_used"],
                    "elapsed_ms_total": r["elapsed_ms_total"],
                    "failures": "; ".join(r["failures"]),
                }
            )

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
