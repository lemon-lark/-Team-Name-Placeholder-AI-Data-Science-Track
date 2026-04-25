"""Clarification agent: ensure we have enough query parameters before SQL.

Renter questions are often underspecified ("show me apartments"). The
clarification agent decides whether the user has been specific enough to
run SQL, and if not, generates a focused follow-up question.

LLM-first behavior: when an LLM provider is available we ask the model in
a single call for a structured ``ClarificationDecision`` (is_specific_enough
+ missing_dimensions + follow_up_text). The deterministic ``count_dimensions``
check still runs as a sanity check - if the LLM claims the question is
specific enough but the regex extracted **zero** dimensions that's a
hallucination and we override.

Mock Demo Mode (``provider == "mock"``) uses the deterministic
``count_dimensions`` + template path, so the app still works without any
external service.
"""
from __future__ import annotations

from typing import Any, Optional

from src.agents.schemas import ClarificationDecision
from src.llm_client import LLMClient


MIN_PARAMETERS: int = 2

# Hard ceiling on clarification rounds. After this many user turns we stop
# asking and run SQL with whatever we have, even if the LLM still wants more.
# Counts the user's initial question + follow-up replies, so the value here
# means "initial + this many follow-ups".
MAX_CLARIFICATION_ROUNDS: int = 2


# Specific analysis intents already encode the dataset / metric the user
# cares about (e.g. "show me HPD housing stock"). When the router lands on
# one of these we count it as one of the supplied dimensions so users do not
# have to re-specify the obvious.
ANALYSIS_INTENTS: frozenset[str] = frozenset(
    {
        "neighborhood_comparison",
        "amenity_search",
        "transit_analysis",
        "housing_supply_analysis",
        "availability_prediction",
    }
)


# Order matters: if we still need more dimensions, we ask about the first
# missing one in this list. Higher-priority dimensions come first.
DIMENSION_PRIORITY: tuple[str, ...] = (
    "max_rent",
    "bedrooms",
    "neighborhoods",
    "proximity",
    "amenities",
    "safety_preference",
    "transit_preference",
)


# Friendly question text used by the mock fallback and as a hint for the LLM.
DIMENSION_PROMPTS: dict[str, str] = {
    "max_rent": "what's your maximum monthly rent (in USD)",
    "bedrooms": "how many bedrooms you need (studio, 1, 2, or 3+)",
    "neighborhoods": "which neighborhoods or boroughs you'd like to focus on",
    "proximity": (
        "anything you want to live near and how close - for example, "
        "'within 0.25 mi of a subway' or 'close to a park'"
    ),
    "amenities": "any nearby amenities that matter (parks, places of worship, grocery, schools)",
    "safety_preference": "whether low crime / high safety is important to you",
    "transit_preference": "preferred transit mode (subway, bus, or any)",
}


def count_dimensions(
    filters_or_router: dict[str, Any],
    intent: Optional[str] = None,
) -> list[str]:
    """Return distinct dimension keys supplied by the user.

    Accepts either a ``filters`` dict or a full ``router_output`` dict (in
    which case ``intent`` is read from it automatically). Each dimension
    counts at most once even if it has multiple values.
    """
    if not filters_or_router:
        return []

    # Allow either a router_output or a raw filters dict for convenience.
    if "filters" in filters_or_router and isinstance(
        filters_or_router.get("filters"), dict
    ):
        filters = filters_or_router["filters"]
        if intent is None:
            intent = filters_or_router.get("intent")
    else:
        filters = filters_or_router

    supplied: list[str] = []
    if filters.get("max_rent") is not None:
        supplied.append("max_rent")
    if (
        filters.get("exact_bedrooms") is not None
        or filters.get("min_bedrooms") is not None
    ):
        supplied.append("bedrooms")
    # Boroughs satisfy the same "where do you want to live" slot as
    # neighborhoods - the SQL agent ORs them into the WHERE clause.
    if filters.get("neighborhoods") or filters.get("boroughs"):
        supplied.append("neighborhoods")
    if filters.get("proximity"):
        supplied.append("proximity")
    if filters.get("amenities"):
        supplied.append("amenities")
    if filters.get("safety_preference"):
        supplied.append("safety_preference")
    if filters.get("transit_preference"):
        supplied.append("transit_preference")
    if filters.get("sort_preference"):
        supplied.append("sort_preference")
    if filters.get("ranking_signal"):
        supplied.append("ranking_signal")
    if intent in ANALYSIS_INTENTS:
        supplied.append("analysis_intent")
    # Explicit "I don't care about X" replies count toward the dimension
    # they target - treats opt-outs as supplied so the bot stops asking.
    for dim in filters.get("opt_outs") or []:
        if dim and dim not in supplied:
            supplied.append(dim)
    return supplied


def _missing_dimensions(supplied: list[str]) -> list[str]:
    return [d for d in DIMENSION_PRIORITY if d not in supplied]


def _mock_follow_up(missing: list[str], needed: int) -> str:
    """Deterministic follow-up question when no real LLM is available.

    Asks for exactly ``needed`` missing dimensions (clamped to >= 1) so we
    don't pad the question with extras the user didn't ask for.
    """
    asks = [DIMENSION_PROMPTS[d] for d in missing[: max(needed, 1)]]
    if not asks:
        return "Could you give me a bit more detail about what you're looking for?"
    if len(asks) == 1:
        body = asks[0]
    elif len(asks) == 2:
        body = f"{asks[0]}, and {asks[1]}"
    else:
        body = ", ".join(asks[:-1]) + f", and {asks[-1]}"
    return (
        "Before I search I need a couple more details. Could you tell me "
        f"{body}?"
    )


def _build_clarification_prompt(
    question: str,
    supplied: list[str],
    missing: list[str],
    needed: int,
    prior_turns: Optional[list[str]],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the LLM-first clarification call."""
    system = (
        "You are a friendly NYC renter assistant. Decide whether the user has "
        "supplied enough constraints to run a precise apartment search, and if "
        "not, write a short follow-up question that asks for the most useful "
        "missing details. Return ONLY valid JSON matching this schema: "
        '{"is_specific_enough": bool, "missing_dimensions": [str], '
        '"follow_up_text": str|null}. '
        "If is_specific_enough is true, set follow_up_text to null. "
        "If is_specific_enough is false, write ONE conversational sentence in "
        "follow_up_text - no markdown, no bullets, no greetings. "
        "Pick missing_dimensions only from the provided list - never invent new ones. "
        "If 'Already supplied dimensions' lists 2 or more entries, set "
        "is_specific_enough = true and follow_up_text = null. "
        "If the user has said they don't care about a dimension (for example "
        "'I don't care about crime', 'none are important', 'doesn't matter'), "
        "treat that dimension as supplied and never re-ask about it. "
        "Do not ask the user to repeat information they already provided."
    )
    dimension_catalog = "\n".join(
        f"- {key}: {prompt}" for key, prompt in DIMENSION_PROMPTS.items()
    )
    missing_lines = "\n".join(
        f"- {d}: {DIMENSION_PROMPTS[d]}" for d in missing[: max(needed, 2) + 1]
    ) or "- (the user already covered every dimension)"
    supplied_lines = ", ".join(supplied) if supplied else "(none yet)"
    history = ""
    if prior_turns:
        history = "Conversation so far:\n" + "\n".join(
            f"- user: {t}" for t in prior_turns
        ) + "\n\n"
    user_prompt = (
        f"{history}Latest message: {question}\n\n"
        f"Already supplied dimensions: {supplied_lines}\n"
        f"Suggested dimensions to ask about (pick the {needed} most useful):\n"
        f"{missing_lines}\n\n"
        f"Available dimension keys for missing_dimensions:\n{dimension_catalog}\n\n"
        "Return JSON only."
    )
    return system, user_prompt


def _filter_missing_dimensions(values: list[str]) -> list[str]:
    """Drop unknown keys the LLM might have invented."""
    valid = set(DIMENSION_PROMPTS.keys())
    return [v for v in values if v in valid]


def clarification_agent(
    question: str,
    router_output: dict[str, Any],
    llm: Optional[LLMClient] = None,
    prior_turns: Optional[list[str]] = None,
    min_parameters: int = MIN_PARAMETERS,
) -> dict[str, Any]:
    """Decide whether the pipeline can run, or whether we need to ask more.

    LLM-first: when an LLM is available we ask it for a single
    ``ClarificationDecision`` (judgment + missing list + follow-up text).
    The deterministic ``count_dimensions`` count still runs as a sanity
    check that vetoes obvious LLM hallucinations (claiming the question is
    specific when zero dimensions were extracted).

    Returns a dict with keys:

    ``needs_clarification``: bool
        True when the user has not supplied enough distinct dimensions.
    ``supplied``: list[str]
        Dimension keys already populated by the deterministic extractor.
    ``missing``: list[str]
        Dimension keys still missing, ordered by priority.
    ``follow_up_text``: str | None
        Natural-language sentence to show the user (only when needs
        clarification is True).
    ``min_parameters``: int
    ``provider_used``: str | None
        ``"llm"`` when the decision + follow-up text came from the LLM, else
        ``"mock"``.
    """
    filters = (router_output or {}).get("filters") or {}
    intent = (router_output or {}).get("intent")
    supplied = count_dimensions(filters, intent=intent)
    deterministic_missing = _missing_dimensions(supplied)

    # Symmetric sanity override: if the deterministic count already meets
    # min_parameters we never call the LLM, which prevents the follow-up
    # loop where a chatty model keeps asking for "one more" dimension.
    if len(supplied) >= min_parameters:
        return {
            "needs_clarification": False,
            "supplied": supplied,
            "missing": deterministic_missing,
            "follow_up_text": None,
            "min_parameters": min_parameters,
            "provider_used": None,
        }

    # Defensive round cap: prior_turns is the user's initial question plus all
    # follow-up replies. Once we've asked enough times we stop pestering and
    # run SQL with whatever we managed to extract.
    if prior_turns is not None and len(prior_turns) > MAX_CLARIFICATION_ROUNDS:
        return {
            "needs_clarification": False,
            "supplied": supplied,
            "missing": deterministic_missing,
            "follow_up_text": None,
            "min_parameters": min_parameters,
            "provider_used": "round_cap",
        }

    use_llm = llm is not None and getattr(llm, "provider", "mock") != "mock"
    if not use_llm:
        return _deterministic_clarification(
            supplied, deterministic_missing, min_parameters
        )

    needed = max(1, min_parameters - len(supplied))
    system, user_prompt = _build_clarification_prompt(
        question, supplied, deterministic_missing, needed, prior_turns
    )
    try:
        decision, _ = llm.generate_validated(  # type: ignore[union-attr]
            prompt=user_prompt,
            system_prompt=system,
            model_cls=ClarificationDecision,
            temperature=0.2,
        )
    except Exception:
        decision = None

    if decision is None:
        return _deterministic_clarification(
            supplied, deterministic_missing, min_parameters
        )

    # Sanity check: the LLM cannot claim the question is specific enough
    # when the deterministic extractor found zero supplied dimensions.
    if decision.is_specific_enough and len(supplied) == 0:
        return _deterministic_clarification(
            supplied, deterministic_missing, min_parameters
        )

    # Defense in depth for the loop bug: if the LLM insists we need more
    # detail but the regex already sees enough dimensions, override and ship.
    if not decision.is_specific_enough and len(supplied) >= min_parameters:
        return {
            "needs_clarification": False,
            "supplied": supplied,
            "missing": deterministic_missing,
            "follow_up_text": None,
            "min_parameters": min_parameters,
            "provider_used": "llm_override",
        }

    if decision.is_specific_enough:
        return {
            "needs_clarification": False,
            "supplied": supplied,
            "missing": deterministic_missing,
            "follow_up_text": None,
            "min_parameters": min_parameters,
            "provider_used": "llm",
        }

    # LLM says we need clarification. Use its missing list (filtered to known
    # keys) and its follow-up text - falling back to the deterministic
    # template if the LLM omitted the text.
    llm_missing = _filter_missing_dimensions(list(decision.missing_dimensions))
    if not llm_missing:
        llm_missing = deterministic_missing
    follow_up_text = decision.follow_up_text or _mock_follow_up(
        deterministic_missing, needed
    )
    return {
        "needs_clarification": True,
        "supplied": supplied,
        "missing": llm_missing,
        "follow_up_text": follow_up_text,
        "min_parameters": min_parameters,
        "provider_used": "llm",
    }


def _deterministic_clarification(
    supplied: list[str],
    missing: list[str],
    min_parameters: int,
) -> dict[str, Any]:
    """Pure ``count_dimensions``-based decision (mock fallback path)."""
    if len(supplied) >= min_parameters:
        return {
            "needs_clarification": False,
            "supplied": supplied,
            "missing": missing,
            "follow_up_text": None,
            "min_parameters": min_parameters,
            "provider_used": None,
        }
    needed = max(1, min_parameters - len(supplied))
    return {
        "needs_clarification": True,
        "supplied": supplied,
        "missing": missing,
        "follow_up_text": _mock_follow_up(missing, needed),
        "min_parameters": min_parameters,
        "provider_used": "mock",
    }


__all__ = [
    "MIN_PARAMETERS",
    "MAX_CLARIFICATION_ROUNDS",
    "DIMENSION_PRIORITY",
    "DIMENSION_PROMPTS",
    "clarification_agent",
    "count_dimensions",
]
