"""Quick smoke test for the agent pipeline.

Run with:

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without installing the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents import answer_user_question
from src.llm_client import LLMClient


QUESTIONS = [
    "Find 1-bedroom apartments under $2,500 in low-crime neighborhoods near parks.",
    "Compare Astoria, Harlem, and Flushing by transit, parks, and housing supply.",
    "Which neighborhoods have the best combination of transit and affordability?",
    "Rank neighborhoods by renter_fit_score.",
    "Show neighborhoods with high availability score and good green space.",
    "Which neighborhoods have lots of HPD housing stock?",
    "Which neighborhoods have the best subway access and affordable rent?",
]


def main() -> int:
    llm = LLMClient(provider="mock")
    failed = 0
    for q in QUESTIONS:
        result = answer_user_question(q, llm=llm)
        ok = result["safety_result"].get("approved")
        rows = len(result["dataframe"])
        chart = result["chart_spec"].get("chart_type")
        intent = result["router_output"]["intent"]
        status = "OK" if ok and rows > 0 else "WARN"
        if not ok:
            status = "FAIL"
            failed += 1
        print(f"[{status}] intent={intent:28s} chart={chart:10s} rows={rows:3d} | {q}")
        if not ok:
            print(f"    reason: {result['safety_result'].get('reason')}")
            print(f"    sql: {result['sql'][:200]}")

    print()
    print(f"{len(QUESTIONS)-failed}/{len(QUESTIONS)} questions passed safety + execution.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
