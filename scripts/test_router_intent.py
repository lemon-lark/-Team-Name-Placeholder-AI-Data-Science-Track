"""Side-by-side smoke test for the LLM-first router agent.

Runs every ``EXAMPLE_QUERIES`` entry from ``app.py`` through ``router_agent``
in two modes - ``mock`` (keyword-only) and ``ollama`` (LLM-first with
deterministic fallback) - and prints the resulting intent + extracted
filters for visual comparison.

Usage::

    python scripts/test_router_intent.py
    python scripts/test_router_intent.py --provider ollama
    python scripts/test_router_intent.py --provider both          # default

The script never raises if Ollama is unreachable: ``router_agent`` falls
back to keyword extraction on any LLM failure, so a missing local server
just means the ``ollama`` column will look identical to ``mock``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.router_agent import router_agent  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402


# Mirrors EXAMPLE_QUERIES at app.py:46-54. Keep this list in sync if the
# canned chips in the dashboard change.
EXAMPLE_QUERIES: list[str] = [
    "Find 1-bedroom apartments under $2,500 in low-crime neighborhoods near parks.",
    "Compare Astoria, Harlem, and Flushing by transit, parks, and housing supply.",
    "Which neighborhoods have the best combination of transit and affordability?",
    "Rank neighborhoods by renter_fit_score.",
    "Show neighborhoods with high availability score and good green space.",
    "Which neighborhoods have lots of HPD housing stock?",
    "Which neighborhoods have the best subway access and affordable rent?",
]


def _format_filters(filters: dict) -> str:
    """One-line, sorted-key dump of non-empty filter values."""
    keep = {k: v for k, v in filters.items() if v not in (None, [], "", False)}
    return json.dumps(keep, sort_keys=True, separators=(", ", ": "))


def _print_run(label: str, result: dict) -> None:
    intent = result.get("intent")
    filters = result.get("filters") or {}
    print(f"  [{label:6s}] intent={intent}")
    print(f"           filters={_format_filters(filters)}")


def _run_one(question: str, providers: list[str]) -> None:
    print(f"\nQ: {question}")
    for provider in providers:
        llm = LLMClient(provider=provider) if provider != "mock" else None
        try:
            result = router_agent(question, llm=llm)
        except Exception as exc:
            print(f"  [{provider:6s}] ERROR: {exc}")
            continue
        _print_run(provider, result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--provider",
        choices=("mock", "ollama", "both"),
        default="both",
        help="Which LLM provider(s) to compare (default: both).",
    )
    args = parser.parse_args()

    if args.provider == "both":
        providers = ["mock", "ollama"]
    else:
        providers = [args.provider]

    for question in EXAMPLE_QUERIES:
        _run_one(question, providers)

    print()
    print(
        "Note: router_agent silently falls back to keyword extraction when the "
        "LLM is unreachable or returns invalid JSON, so identical mock/ollama "
        "rows usually mean the LLM was not actually consulted."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
