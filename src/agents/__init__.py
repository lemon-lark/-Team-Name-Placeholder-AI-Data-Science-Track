"""Agentic pipeline for AI-partments.

Each agent is a small function/class with a clear responsibility. The
``answer_user_question`` orchestrator glues them together and is the single
entry point used by ``app.py``.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.agents.clarification_agent import clarification_agent
from src.agents.recommendation_agent import recommendation_agent
from src.agents.router_agent import combine_turns, router_agent
from src.agents.safety_agent import safety_agent
from src.agents.schema_agent import schema_agent
from src.agents.sql_agent import sql_agent
from src.agents.visualization_agent import visualization_agent
from src.llm_client import LLMClient
from src.tools.match_scoring import score_results_for_request
from src.tools.query_tools import run_safe_sql


def answer_user_question(
    question: str,
    llm: Optional[LLMClient] = None,
    prior_turns: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run the full router -> clarify -> schema -> SQL -> safety -> exec -> viz -> rec pipeline.

    Parameters
    ----------
    question:
        The user's latest message. When ``prior_turns`` is provided it is
        treated as a follow-up reply to a clarification question.
    llm:
        Optional :class:`LLMClient`. Defaults to a mock provider so the app
        always works offline.
    prior_turns:
        Earlier user messages from the same conversation. Combined with
        ``question`` so the router can re-extract structured filters across
        the whole exchange.
    """
    llm = llm or LLMClient(provider="mock")

    turns: list[str] = list(prior_turns or []) + [question]
    merged_question = combine_turns(turns)

    router_output = router_agent(merged_question, llm=llm)

    clarification = clarification_agent(
        merged_question,
        router_output,
        llm=llm,
        prior_turns=turns,
    )

    base_payload: dict[str, Any] = {
        "question": question,
        "turns": turns,
        "merged_question": merged_question,
        "router_output": router_output,
        "clarification": clarification,
        "provider_used": getattr(llm, "provider", "mock"),
    }

    if clarification.get("needs_clarification"):
        return {
            **base_payload,
            "needs_clarification": True,
            "schema_context": {},
            "sql": "",
            "safety_result": {"approved": False, "reason": "Awaiting clarification."},
            "dataframe": pd.DataFrame(),
            "chart_spec": {"chart_type": "table_only", "title": "Awaiting details"},
            "recommendation": clarification.get("follow_up_text") or "",
        }

    schema_context = schema_agent(merged_question, router_output)
    generated_sql = sql_agent(merged_question, schema_context, router_output, llm=llm)
    safety_result = safety_agent(generated_sql)

    base_payload.update(
        {
            "schema_context": schema_context,
            "sql": generated_sql,
            "safety_result": safety_result,
            "needs_clarification": False,
        }
    )

    if not safety_result.get("approved"):
        return {
            **base_payload,
            "error": safety_result.get("reason", "Query rejected."),
            "dataframe": pd.DataFrame(),
            "chart_spec": {"chart_type": "table_only"},
            "recommendation": "The generated SQL was rejected by the safety layer. See the SQL and safety panels for details.",
        }

    safe_sql = safety_result.get("safe_sql") or generated_sql
    try:
        df = run_safe_sql(safe_sql)
        exec_error: Optional[str] = None
    except Exception as exc:
        df = pd.DataFrame()
        exec_error = str(exc)

    score_meta: dict[str, Any] = {
        "requested_dimensions": [],
        "visible_score_columns": [],
        "overall_score_column": "overall_match_score",
    }
    if not df.empty:
        try:
            df, score_meta = score_results_for_request(
                df,
                filters=router_output.get("filters") or {},
                requested_boost=2.0,
            )
        except Exception:
            # Scoring is additive; never block base query results on scoring errors.
            score_meta = {
                "requested_dimensions": [],
                "visible_score_columns": [],
                "overall_score_column": "overall_match_score",
            }

    chart_spec = visualization_agent(merged_question, df)
    recommendation = recommendation_agent(merged_question, df, chart_spec, llm=llm)

    result = {
        **base_payload,
        "sql": safe_sql,
        "dataframe": df,
        "chart_spec": chart_spec,
        "recommendation": recommendation,
        "requested_dimensions": score_meta.get("requested_dimensions", []),
        "visible_score_columns": score_meta.get("visible_score_columns", []),
        "overall_score_column": score_meta.get("overall_score_column", "overall_match_score"),
    }
    if exec_error:
        result["error"] = exec_error
    return result


__all__ = [
    "answer_user_question",
    "router_agent",
    "schema_agent",
    "sql_agent",
    "safety_agent",
    "visualization_agent",
    "recommendation_agent",
    "clarification_agent",
]
