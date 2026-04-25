"""Unit-style tests for the clarification + proximity-extraction layer."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.clarification_agent import (  # noqa: E402
    MAX_CLARIFICATION_ROUNDS,
    MIN_PARAMETERS,
    clarification_agent,
    count_dimensions,
)
from src.agents.router_agent import (  # noqa: E402
    DEFAULT_PROXIMITY_MILES,
    _extract_boroughs,
    _extract_opt_outs,
    _extract_proximity,
    combine_turns,
    keyword_route,
    router_agent,
)
from src.agents.sql_agent import sql_agent  # noqa: E402


# ---------------------------------------------------------------------------
# count_dimensions
# ---------------------------------------------------------------------------

DIMENSION_TESTS: list[tuple[str, dict, str | None, set[str]]] = [
    (
        "max_rent only",
        {"max_rent": 2500},
        None,
        {"max_rent"},
    ),
    (
        "rent + bedrooms",
        {"max_rent": 2500, "exact_bedrooms": 1},
        None,
        {"max_rent", "bedrooms"},
    ),
    (
        "neighborhood + amenity",
        {"neighborhoods": ["Astoria"], "amenities": ["park"]},
        None,
        {"neighborhoods", "amenities"},
    ),
    (
        "proximity only",
        {"proximity": [{"target": "subway", "max_distance_miles": 0.25, "kind": "transit"}]},
        None,
        {"proximity"},
    ),
    (
        "ranking signal + analysis intent",
        {"ranking_signal": True},
        "housing_supply_analysis",
        {"ranking_signal", "analysis_intent"},
    ),
    (
        "ignores empty / falsy",
        {
            "max_rent": None,
            "neighborhoods": [],
            "amenities": [],
            "proximity": [],
            "ranking_signal": False,
            "min_bedrooms": None,
        },
        None,
        set(),
    ),
    (
        "boroughs satisfy neighborhoods slot",
        {"boroughs": ["Brooklyn"], "max_rent": 2500},
        None,
        {"max_rent", "neighborhoods"},
    ),
    (
        "opt_outs count toward supplied",
        {
            "max_rent": 2000,
            "opt_outs": ["safety_preference", "amenities"],
        },
        None,
        {"max_rent", "safety_preference", "amenities"},
    ),
]


def test_count_dimensions() -> int:
    failures = 0
    for label, filters, intent, expected in DIMENSION_TESTS:
        got = set(count_dimensions(filters, intent=intent))
        if got != expected:
            print(f"  [FAIL] count_dimensions: {label}: got {got!r} expected {expected!r}")
            failures += 1
        else:
            print(f"  [OK]   count_dimensions: {label}")
    return failures


# ---------------------------------------------------------------------------
# _extract_proximity
# ---------------------------------------------------------------------------

PROXIMITY_TESTS: list[tuple[str, list[dict]]] = [
    (
        "Within 0.5 miles of a mosque",
        [{"target": "mosque", "max_distance_miles": 0.5, "kind": "amenity"}],
    ),
    (
        "Find a park within 1 mile",
        [],  # the keyword "park" precedes the distance phrase, current impl skips
    ),
    (
        "Apartments within 3 blocks of a church",
        [{"target": "church", "max_distance_miles": 0.15, "kind": "amenity"}],
    ),
    (
        "Listings within 2 km of a grocery",
        [
            {
                "target": "grocery",
                "max_distance_miles": round(2 * 0.621371, 3),
                "kind": "amenity",
            }
        ],
    ),
    (
        "1-bedroom near subway under $2800",
        [
            {
                "target": "subway",
                "max_distance_miles": DEFAULT_PROXIMITY_MILES,
                "kind": "transit",
            }
        ],
    ),
    (
        "Apartments close to a bus stop in Astoria",
        [
            {
                "target": "bus",
                "max_distance_miles": DEFAULT_PROXIMITY_MILES,
                "kind": "transit",
            }
        ],
    ),
    (
        "Walking distance to a train",
        [
            {
                "target": "subway",
                "max_distance_miles": DEFAULT_PROXIMITY_MILES,
                "kind": "transit",
            }
        ],
    ),
    (
        "Within 0.4 mi of a train",
        [{"target": "subway", "max_distance_miles": 0.4, "kind": "transit"}],
    ),
    (
        "No proximity here, just a neighborhood",
        [],
    ),
    (
        "near a mosque AND within 0.5 mile of a grocery",
        [
            {"target": "grocery", "max_distance_miles": 0.5, "kind": "amenity"},
            {"target": "mosque", "max_distance_miles": DEFAULT_PROXIMITY_MILES, "kind": "amenity"},
        ],
    ),
]


def _proximity_equal(a: list[dict], b: list[dict]) -> bool:
    if len(a) != len(b):
        return False
    a_sorted = sorted(a, key=lambda p: p.get("target") or "")
    b_sorted = sorted(b, key=lambda p: p.get("target") or "")
    for x, y in zip(a_sorted, b_sorted):
        if x.get("target") != y.get("target"):
            return False
        if x.get("kind") != y.get("kind"):
            return False
        if abs(
            float(x.get("max_distance_miles") or 0)
            - float(y.get("max_distance_miles") or 0)
        ) > 1e-3:
            return False
    return True


def test_extract_proximity() -> int:
    failures = 0
    for question, expected in PROXIMITY_TESTS:
        got = _extract_proximity(question)
        if not _proximity_equal(got, expected):
            print(f"  [FAIL] proximity: {question!r}")
            print(f"          got      = {got}")
            print(f"          expected = {expected}")
            failures += 1
        else:
            print(f"  [OK]   proximity: {question!r}")
    return failures


# ---------------------------------------------------------------------------
# clarification_agent + combine_turns
# ---------------------------------------------------------------------------

CLARIFICATION_TESTS: list[tuple[str, bool]] = [
    ("Find 1-bedroom apartments under $2,500 in low-crime neighborhoods near parks.", False),
    ("Show apartments.", True),
    ("Apartments under $2500.", True),
    ("Apartments under $2500 in Astoria.", False),
    ("1-bedroom near subway under $2800.", False),
    ("Within 0.5 mile of a mosque, under $3000.", False),
    ("Compare Astoria and Harlem.", False),
    ("Rank neighborhoods by renter_fit_score.", False),
    ("Which neighborhoods have lots of HPD housing stock?", False),
]


def test_clarification() -> int:
    failures = 0
    for question, expects_clarification in CLARIFICATION_TESTS:
        router_output = router_agent(question)
        clar = clarification_agent(question, router_output)
        actual = bool(clar.get("needs_clarification"))
        if actual != expects_clarification:
            print(
                f"  [FAIL] clarification: {question!r}: "
                f"expected needs_clarification={expects_clarification}, got {actual}"
            )
            failures += 1
        else:
            print(f"  [OK]   clarification: {question!r} -> {actual}")
    return failures


def test_combine_turns_and_followup() -> int:
    failures = 0
    initial = "Show apartments."
    reply = "Under $2500 in Astoria."
    merged = combine_turns([initial, reply])
    if "Astoria" not in merged or "$2500" not in merged:
        print(f"  [FAIL] combine_turns: missing parts in {merged!r}")
        failures += 1
    else:
        print(f"  [OK]   combine_turns merged turns into {merged!r}")

    router_after = router_agent(merged)
    clar_after = clarification_agent(merged, router_after, prior_turns=[initial, reply])
    if clar_after.get("needs_clarification"):
        print(
            f"  [FAIL] clarification: merged turns still flagged as needing clarification: "
            f"{clar_after!r}"
        )
        failures += 1
    else:
        print("  [OK]   merged turns satisfy MIN_PARAMETERS")

    # Mock-mode follow-up text on an under-specified question.
    router_initial = router_agent(initial)
    clar_initial = clarification_agent(initial, router_initial)
    if not clar_initial.get("needs_clarification"):
        print("  [FAIL] mock follow-up: expected clarification needed for 'Show apartments.'")
        failures += 1
    elif not (clar_initial.get("follow_up_text") or "").strip():
        print("  [FAIL] mock follow-up: missing follow_up_text")
        failures += 1
    else:
        print(
            f"  [OK]   mock follow-up text emitted "
            f"({len(clar_initial['follow_up_text'])} chars)"
        )

    return failures


# ---------------------------------------------------------------------------
# Borough extraction + keyword_route
# ---------------------------------------------------------------------------

BOROUGH_TESTS: list[tuple[str, list[str]]] = [
    ("apartments in brooklyn", ["Brooklyn"]),
    ("Find places in The Bronx near a subway", ["Bronx"]),
    ("Manhattan or Queens please", ["Manhattan", "Queens"]),
    ("staten island under 2000", ["Staten Island"]),
    ("astoria specifically", []),  # neighborhood, not a borough
]


def test_extract_boroughs() -> int:
    failures = 0
    for question, expected in BOROUGH_TESTS:
        got = _extract_boroughs(question)
        if sorted(got) != sorted(expected):
            print(f"  [FAIL] _extract_boroughs: {question!r}: got {got!r} expected {expected!r}")
            failures += 1
        else:
            print(f"  [OK]   _extract_boroughs: {question!r} -> {got}")
    return failures


def test_keyword_route_boroughs() -> int:
    failures = 0
    out = keyword_route("brooklyn")
    if out["filters"].get("boroughs") != ["Brooklyn"]:
        print(f"  [FAIL] keyword_route('brooklyn').filters.boroughs = {out['filters'].get('boroughs')!r}")
        failures += 1
    else:
        print("  [OK]   keyword_route exposes boroughs in filters")

    # Regression: housing-intent phrasing with amenity proximity should still
    # route to apartment search rather than amenity table lookup.
    live_query = "I wanna live in manhattan nearby a mosque"
    live_out = keyword_route(live_query)
    if live_out.get("intent") != "rental_search":
        print(
            f"  [FAIL] keyword_route live-intent: expected rental_search, "
            f"got {live_out.get('intent')!r}"
        )
        failures += 1
    else:
        print("  [OK]   keyword_route treats 'live in ... near ...' as rental_search")
    return failures


# ---------------------------------------------------------------------------
# Opt-out extraction
# ---------------------------------------------------------------------------

OPT_OUT_TESTS: list[tuple[str, set[str]]] = [
    ("i dont really care about crime", {"safety_preference"}),
    ("none are important", set()),  # bare opt-out, no topic - empty
    ("i don't care about a nearby grocery store", {"amenities", "proximity"}),
    ("any transit is fine", {"transit_preference"}),
    ("safety doesn't matter", {"safety_preference"}),
    ("Find 1-bedroom apartments under $2,500", set()),  # no opt-out phrase
    ("no preference for neighborhoods", {"neighborhoods"}),
]


def test_extract_opt_outs() -> int:
    failures = 0
    for question, expected in OPT_OUT_TESTS:
        got = set(_extract_opt_outs(question))
        if got != expected:
            print(f"  [FAIL] _extract_opt_outs: {question!r}: got {got!r} expected {expected!r}")
            failures += 1
        else:
            print(f"  [OK]   _extract_opt_outs: {question!r} -> {got}")
    return failures


# ---------------------------------------------------------------------------
# Regression: the loop bug from the screenshot. With max_rent + bedrooms
# already supplied, the deterministic clarification path must short-circuit
# without calling the LLM, even when prior_turns is set.
# ---------------------------------------------------------------------------


class _StubLLM:
    """Pretends to be an Ollama/watsonx provider that always returns
    is_specific_enough=False so we can prove the deterministic short-circuit
    fires before the LLM is even consulted."""

    provider = "ollama"

    def generate_validated(self, *_args: object, **_kwargs: object) -> tuple[object, dict[str, object]]:
        raise AssertionError("clarification_agent should not have called the LLM")


def test_loop_short_circuit() -> int:
    failures = 0
    initial = "find 1 bedroom under 2000"
    router_output = router_agent(initial)
    clar = clarification_agent(
        combine_turns([initial]),
        router_output,
        llm=_StubLLM(),  # type: ignore[arg-type]
        prior_turns=[initial],
    )
    if clar.get("needs_clarification"):
        print(f"  [FAIL] loop short-circuit: still needs_clarification, got {clar!r}")
        failures += 1
    elif clar.get("provider_used") not in (None,):
        print(f"  [FAIL] loop short-circuit: expected provider_used=None, got {clar.get('provider_used')!r}")
        failures += 1
    else:
        print("  [OK]   loop short-circuit: 1BR + max_rent skipped LLM and shipped SQL")

    # Same scenario with Brooklyn instead of a known neighborhood: boroughs
    # should now satisfy the 'neighborhoods' slot too.
    second = "brooklyn"
    merged = combine_turns([initial, second])
    router_output = router_agent(merged)
    clar = clarification_agent(
        merged,
        router_output,
        llm=_StubLLM(),  # type: ignore[arg-type]
        prior_turns=[initial, second],
    )
    if clar.get("needs_clarification"):
        print(f"  [FAIL] borough short-circuit: still needs_clarification, got {clar!r}")
        failures += 1
    else:
        print("  [OK]   borough short-circuit: brooklyn satisfies neighborhoods slot")
    return failures


def test_round_cap() -> int:
    failures = 0
    # All three turns are deliberately under-specified so deterministic count
    # never hits MIN_PARAMETERS - only the round cap can break the loop.
    turns = ["uhh", "i don't know", "whatever you think"]
    if len(turns) <= MAX_CLARIFICATION_ROUNDS:
        # Defensive: if MAX_CLARIFICATION_ROUNDS ever grows, extend the list.
        turns = turns + ["still no idea"] * (MAX_CLARIFICATION_ROUNDS - len(turns) + 1)
    router_output = router_agent(combine_turns(turns))
    clar = clarification_agent(
        combine_turns(turns),
        router_output,
        prior_turns=turns,
    )
    if clar.get("needs_clarification"):
        print(f"  [FAIL] round cap: still asking after {len(turns)} turns: {clar!r}")
        failures += 1
    elif clar.get("provider_used") != "round_cap":
        print(f"  [FAIL] round cap: expected provider_used='round_cap', got {clar.get('provider_used')!r}")
        failures += 1
    else:
        print(f"  [OK]   round cap: forced SQL after {len(turns)} turns")
    return failures


# ---------------------------------------------------------------------------
# SQL semantic guard: if LLM SQL drops required location predicates, sql_agent
# must fall back to deterministic SQL that preserves router filters.
# ---------------------------------------------------------------------------


class _StubSQLResult:
    def __init__(self, text: str) -> None:
        self.text = text
        self.fell_back = False


class _StubSQLLLM:
    provider = "ollama"

    def __init__(self, sql_text: str) -> None:
        self._sql_text = sql_text

    def generate(self, *_args: object, **_kwargs: object) -> _StubSQLResult:
        return _StubSQLResult(self._sql_text)


def test_sql_semantic_location_guard() -> int:
    failures = 0
    question = "I am looking for 4 bedroom apartment in the bronx"
    router_output = {
        "intent": "rental_search",
        "filters": {
            "exact_bedrooms": 4,
            "min_bedrooms": 4,
            "max_rent": None,
            "neighborhoods": ["Bronx"],
            "boroughs": ["Bronx"],
            "amenities": [],
            "proximity": [],
            "safety_preference": None,
            "transit_preference": "any",
            "sort_preference": "renter_fit",
            "ranking_signal": False,
            "opt_outs": [],
        },
    }
    schema_context = {
        "required_tables": ["apartments", "neighborhoods", "neighborhood_stats"],
        "notes": "Apartment search",
    }
    llm_sql_missing_location = (
        "SELECT n.name AS neighborhood, n.borough, a.address, a.rent, a.bedrooms "
        "FROM apartments a "
        "JOIN neighborhoods n ON a.neighborhood_id = n.neighborhood_id "
        "JOIN neighborhood_stats ns ON n.neighborhood_id = ns.neighborhood_id "
        "WHERE a.bedrooms = 4 "
        "ORDER BY ns.renter_fit_score DESC LIMIT 50"
    )
    out_sql = sql_agent(
        question=question,
        schema_context=schema_context,
        router_output=router_output,
        llm=_StubSQLLLM(llm_sql_missing_location),  # type: ignore[arg-type]
    )
    lowered = out_sql.lower()
    if "n.borough in ('bronx')" not in lowered:
        print("  [FAIL] sql semantic guard: missing borough predicate after fallback")
        failures += 1
    else:
        # "Bronx" is a borough label, not a canonical neighborhood name, so
        # fallback SQL may intentionally keep only n.borough predicate.
        print("  [OK]   sql semantic guard: fallback preserved required location predicate")
    return failures


def main() -> int:
    print(f"MIN_PARAMETERS = {MIN_PARAMETERS}")
    print(f"MAX_CLARIFICATION_ROUNDS = {MAX_CLARIFICATION_ROUNDS}\n")
    failures = 0
    print("--- count_dimensions ---")
    failures += test_count_dimensions()
    print("\n--- _extract_proximity ---")
    failures += test_extract_proximity()
    print("\n--- _extract_boroughs ---")
    failures += test_extract_boroughs()
    print("\n--- keyword_route boroughs ---")
    failures += test_keyword_route_boroughs()
    print("\n--- _extract_opt_outs ---")
    failures += test_extract_opt_outs()
    print("\n--- clarification_agent ---")
    failures += test_clarification()
    print("\n--- combine_turns + follow-up ---")
    failures += test_combine_turns_and_followup()
    print("\n--- loop short-circuit ---")
    failures += test_loop_short_circuit()
    print("\n--- round cap ---")
    failures += test_round_cap()
    print("\n--- sql semantic location guard ---")
    failures += test_sql_semantic_location_guard()
    print()
    if failures == 0:
        print("All clarification tests passed.")
        return 0
    print(f"{failures} clarification test failure(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
