"""Router agent: classify intent and extract structured filters from a question.

The router is **LLM-first** when an LLM provider is available: it asks the
model to produce a JSON-structured ``RouterOutput`` (validated by Pydantic)
and only falls back to the deterministic ``keyword_route`` extractor when
the LLM is unreachable, returns invalid JSON, or hallucinates a non-conformant
shape.

A small "fact veto" still runs over the LLM result: for high-stakes facts
(``max_rent``, ``min_bedrooms``, ``exact_bedrooms``, ``neighborhoods``) the
regex-extracted value wins when it disagrees with the LLM. This catches the
classic small-model failure of fudging numbers and named entities while
still letting Ollama drive intent and the more nuanced fields.

Mock Demo Mode (``provider == "mock"``) skips the LLM entirely and runs the
keyword extractor, so the app still works without any external service.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from src.agents.schemas import RouterOutput
from src.llm_client import LLMClient


# Allowed intent labels. Keep in sync with the spec.
INTENTS = (
    "rental_search",
    "neighborhood_comparison",
    "availability_prediction",
    "amenity_search",
    "transit_analysis",
    "housing_supply_analysis",
    "general_analysis",
)


# Lowercased neighborhood keywords -> display name (matches src/seed_data.py).
NEIGHBORHOOD_KEYWORDS: dict[str, str] = {
    "astoria": "Astoria",
    "sunnyside": "Sunnyside",
    "harlem": "Harlem",
    "washington heights": "Washington Heights",
    "williamsburg": "Williamsburg",
    "bushwick": "Bushwick",
    "park slope": "Park Slope",
    "jersey city heights": "Jersey City Heights",
    "journal square": "Journal Square",
    "hoboken": "Hoboken",
    "mott haven": "Mott Haven",
    "flushing": "Flushing",
}


# Borough keywords. Matches the borough column written by seed_data + ingestion.
# Listing both "the bronx" and "bronx" so longest-first iteration handles either
# spelling deterministically.
BOROUGH_KEYWORDS: dict[str, str] = {
    "brooklyn": "Brooklyn",
    "manhattan": "Manhattan",
    "queens": "Queens",
    "the bronx": "Bronx",
    "bronx": "Bronx",
    "staten island": "Staten Island",
    "hudson": "Hudson",
    # NYC sub-neighborhood phrases that should at least scope to borough.
    "upper east side": "Manhattan",
    "ues": "Manhattan",
    "upper west side": "Manhattan",
    "midtown": "Manhattan",
    "soho": "Manhattan",
    "tribeca": "Manhattan",
    "financial district": "Manhattan",
    "east village": "Manhattan",
    "west village": "Manhattan",
    "chelsea": "Manhattan",
}


AMENITY_KEYWORDS: dict[str, str] = {
    "park": "park",
    "parks": "park",
    "green space": "park",
    "mosque": "mosque",
    "masjid": "mosque",
    "church": "church",
    "synagogue": "synagogue",
    "temple": "temple",
    "place of worship": "worship",
    "places of worship": "worship",
    "worship": "worship",
    "grocery": "grocery",
    "grocer": "grocery",
    "subway": "subway",
    "train": "subway",
    "metro": "subway",
    "bus": "bus",
    "school": "school",
    "schools": "school",
    "healthcare": "healthcare",
    "hospital": "healthcare",
    "clinic": "healthcare",
    "library": "library",
    "libraries": "library",
    "community": "community_center",
    "gym": "gym",
    "laundromat": "laundromat",
}


# --- Proximity extraction ----------------------------------------------------
# Targets that are computed via apartment-level haversine against the
# transit_* tables (rather than the cheap amenities EXISTS subquery).
TRANSIT_TARGETS: frozenset[str] = frozenset({"subway", "bus"})

# Default radius (in miles) when the user says "near X" / "close to X" without
# providing a number. Matches the tiering documented in the README.
DEFAULT_PROXIMITY_MILES: float = 0.25

# Sorted longest-first so multi-word keys like "place of worship" match before
# their single-word substrings.
_AMENITY_KEYS_LONG_FIRST: list[str] = sorted(
    AMENITY_KEYWORDS.keys(), key=lambda s: (-len(s), s)
)
_AMENITY_PATTERN: str = "|".join(re.escape(k) for k in _AMENITY_KEYS_LONG_FIRST)

# Distance unit multipliers (-> miles).
_UNIT_TO_MILES: dict[str, float] = {
    "mile": 1.0,
    "miles": 1.0,
    "mi": 1.0,
    "km": 0.621371,
    "kilometer": 0.621371,
    "kilometers": 0.621371,
    "block": 0.05,
    "blocks": 0.05,
}

_DISTANCE_RE = re.compile(
    r"\b(?:within|under|closer than|less than|no more than|up to)\s+"
    r"(\d+(?:\.\d+)?)\s*"
    r"(miles?|mi|km|kilometers?|blocks?)\b",
    re.IGNORECASE,
)
_BARE_PROX_RE = re.compile(
    r"\b(?:near|next to|by|close to|walking distance from|walking distance to|"
    r"nearby|right by|right next to|adjacent to)\b",
    re.IGNORECASE,
)


def _convert_to_miles(amount: float, unit: str) -> float:
    multiplier = _UNIT_TO_MILES.get(unit.lower(), 1.0)
    return round(amount * multiplier, 3)


def _find_target_in_window(text: str) -> Optional[str]:
    """Return the canonical target name of the earliest amenity keyword match.

    Iterating longest-first means a multi-word key like ``"place of worship"``
    wins over its substring ``"worship"`` at the same start position; among
    matches at different positions we still prefer the leftmost.
    """
    best_pos: Optional[int] = None
    best_target: Optional[str] = None
    for key in _AMENITY_KEYS_LONG_FIRST:
        m = re.search(r"\b" + re.escape(key) + r"\b", text)
        if m is None:
            continue
        if best_pos is None or m.start() < best_pos:
            best_pos = m.start()
            best_target = AMENITY_KEYWORDS[key]
    return best_target


def _extract_proximity(question: str) -> list[dict[str, Any]]:
    """Pull zero or more proximity constraints from a free-text question.

    Each entry: ``{"target": str, "max_distance_miles": float,
    "kind": "amenity" | "transit"}``.
    """
    q = question.lower()
    results: list[dict[str, Any]] = []
    seen_targets: set[str] = set()

    for match in _DISTANCE_RE.finditer(q):
        amount = float(match.group(1))
        unit = match.group(2)
        miles = _convert_to_miles(amount, unit)
        tail = q[match.end() : match.end() + 80]
        target = _find_target_in_window(tail)
        if target is None or target in seen_targets:
            continue
        seen_targets.add(target)
        results.append(
            {
                "target": target,
                "max_distance_miles": miles,
                "kind": "transit" if target in TRANSIT_TARGETS else "amenity",
            }
        )

    for match in _BARE_PROX_RE.finditer(q):
        tail = q[match.end() : match.end() + 60]
        target = _find_target_in_window(tail)
        if target is None or target in seen_targets:
            continue
        seen_targets.add(target)
        results.append(
            {
                "target": target,
                "max_distance_miles": DEFAULT_PROXIMITY_MILES,
                "kind": "transit" if target in TRANSIT_TARGETS else "amenity",
            }
        )

    return results


def combine_turns(turns: list[str]) -> str:
    """Join the original question + clarification replies into one string.

    The router runs its keyword extraction over this combined text so a user
    can supply parameters across multiple turns without losing prior context.
    """
    if not turns:
        return ""
    parts = [t.strip() for t in turns if t and t.strip()]
    return " ; ".join(parts)


def _extract_max_rent(question: str) -> Optional[int]:
    q = question.lower().replace(",", "")
    matches = re.findall(r"\$?\s*(\d{3,5})\b", q)
    if not matches:
        return None
    candidates = [int(m) for m in matches if 500 <= int(m) <= 20000]
    if not candidates:
        return None
    if any(word in q for word in ("under", "below", "less than", "max", "up to", "<=", "<")):
        return min(candidates)
    return min(candidates)


def _extract_bedrooms(question: str) -> tuple[Optional[int], Optional[int]]:
    q = question.lower()
    if "studio" in q:
        return 0, 0
    match = re.search(r"(\d+)\s*[- ]?\s*(?:bed|br|bedroom)", q)
    if match:
        n = int(match.group(1))
        return n, n
    return None, None


def _extract_neighborhoods(question: str) -> list[str]:
    q = question.lower()
    found: list[str] = []
    for key, name in NEIGHBORHOOD_KEYWORDS.items():
        if key in q and name not in found:
            found.append(name)
    return found


def _extract_boroughs(question: str) -> list[str]:
    """Pull borough names from the question.

    Iterates longest-key-first so "the bronx" wins over the "bronx" substring
    at the same position; later we de-duplicate by display name anyway.
    """
    q = question.lower()
    found: list[str] = []
    for key in sorted(BOROUGH_KEYWORDS, key=lambda s: (-len(s), s)):
        if re.search(r"\b" + re.escape(key) + r"\b", q):
            name = BOROUGH_KEYWORDS[key]
            if name not in found:
                found.append(name)
    return found


# --- Opt-out extraction ------------------------------------------------------
# Phrases like "i don't care" / "doesn't matter" / "none are important" used in
# combination with a topic word should mark the corresponding dimension as
# "user explicitly waived" so the clarification agent stops asking about it.

_OPT_OUT_RE = re.compile(
    r"\b(?:"
    r"i\s*don'?t\s*(?:really\s+)?care(?:\s+about)?"
    r"|doesn'?t\s+matter"
    r"|not\s+important"
    r"|no\s+preference"
    r"|whatever"
    r"|none\s+(?:are|is)\s+important"
    r"|none"
    # "any[thing] [topic] [is] fine" - allows an optional topic word in
    # between, e.g. "any transit is fine".
    r"|any(?:thing)?\s+(?:\w+\s+)?(?:is\s+)?fine"
    r")\b",
    re.IGNORECASE,
)

_DIMENSION_TOPIC_RE: dict[str, "re.Pattern[str]"] = {
    "safety_preference": re.compile(r"\b(?:crime|safety|safe|safer)\b", re.IGNORECASE),
    "amenities": re.compile(
        r"\b(?:amenit\w*|grocer\w*|school\w*|park|parks|worship|mosque|"
        r"church|synagogue|temple|gym|library|libraries|laundromat)\b",
        re.IGNORECASE,
    ),
    "proximity": re.compile(
        r"\b(?:proximity|near|nearby|close\s+to|walking\s+distance)\b",
        re.IGNORECASE,
    ),
    "transit_preference": re.compile(
        r"\b(?:transit|subway|bus|train|metro)\b", re.IGNORECASE
    ),
    "neighborhoods": re.compile(
        r"\b(?:neighborhood\w*|borough\w*|area)\b", re.IGNORECASE
    ),
}


def _extract_opt_outs(question: str) -> list[str]:
    """Return dimension keys the user has explicitly said they do not care about.

    Requires both an opt-out phrase ("don't care", "none", "doesn't matter",
    etc.) and a topic word ("crime", "amenities", "proximity", ...). When the
    user says only "I don't care" with no topic the result is empty - the
    clarification agent treats that conservatively.
    """
    if not _OPT_OUT_RE.search(question):
        return []
    out: list[str] = []
    for dim, topic_re in _DIMENSION_TOPIC_RE.items():
        if topic_re.search(question):
            out.append(dim)
    return out


def _extract_amenities(question: str) -> list[str]:
    q = question.lower()
    found: list[str] = []
    for key, name in AMENITY_KEYWORDS.items():
        if re.search(r"\b" + re.escape(key) + r"\b", q) and name not in found:
            found.append(name)
    return found


def _extract_safety_pref(question: str) -> Optional[str]:
    q = question.lower()
    if any(t in q for t in ("low crime", "low-crime", "safe", "safer", "high safety")):
        if "low" in q or "safer" in q:
            return "low_crime"
        return "high_safety"
    return None


def _extract_transit_pref(question: str) -> Optional[str]:
    q = question.lower()
    has_subway = "subway" in q or "train" in q or "metro" in q
    has_bus = "bus" in q
    if has_subway and not has_bus:
        return "subway"
    if has_bus and not has_subway:
        return "bus"
    if has_subway and has_bus:
        return "any"
    if "transit" in q or "public transit" in q:
        return "any"
    return None


_RANKING_SINGLE_WORDS = (
    "rank",
    "top",
    "best",
    "worst",
    "highest",
    "lowest",
    "most",
    "fewest",
    "leading",
    "biggest",
    "largest",
    "smallest",
    "cheapest",
)
_RANKING_RE = re.compile(
    r"\b(?:" + "|".join(_RANKING_SINGLE_WORDS) + r")\b", re.IGNORECASE
)
_RANKING_PHRASES_RE = re.compile(
    r"\b(?:lots? of|loads? of|tons? of|plenty of|a lot of|the most|the best)\b",
    re.IGNORECASE,
)


def _extract_ranking_signal(question: str) -> bool:
    """Return True when the question implies a ranking / leaderboard view."""
    q = question.lower()
    return bool(_RANKING_RE.search(q) or _RANKING_PHRASES_RE.search(q))


def _extract_sort_pref(question: str) -> Optional[str]:
    q = question.lower()
    if "renter_fit" in q or "renter fit" in q or "rank" in q:
        return "renter_fit"
    if "transit" in q and ("best" in q or "rank" in q):
        return "transit"
    if "green" in q or "park" in q:
        return "green_space"
    if "afford" in q:
        return "rent"
    if "safe" in q:
        return "safety"
    if "available" in q or "availability" in q:
        return "availability"
    return None


def _classify_intent(question: str, neighborhoods: list[str], amenities: list[str]) -> str:
    q = question.lower()
    has_word = lambda w: re.search(r"\b" + re.escape(w) + r"\b", q) is not None

    if "compare" in q and len(neighborhoods) >= 2:
        return "neighborhood_comparison"
    if has_word("rank"):
        return "neighborhood_comparison"
    if "hpd" in q or "housing stock" in q or "housing supply" in q:
        return "housing_supply_analysis"
    if has_word("availability") or has_word("available") or has_word("supply"):
        return "availability_prediction"
    rental_keywords = (
        "apartment",
        "apartments",
        "1-bedroom",
        "2-bedroom",
        "studio",
        "listings",
        "listing",
        "find",
        "live",
        "living",
        "move",
    )
    rental_phrases = ("1 bed", "2 bed", "3 bed", "bedroom")
    if any(has_word(k) for k in rental_keywords) or any(p in q for p in rental_phrases):
        return "rental_search"
    if has_word("subway") or has_word("bus") or has_word("transit"):
        return "transit_analysis"
    if amenities and ("near" in q or has_word("with")):
        # "live in X near Y" is usually a housing request, not a POI list.
        if neighborhoods or _extract_boroughs(question):
            return "rental_search"
        return "amenity_search"
    if has_word("best") or has_word("top"):
        return "neighborhood_comparison"
    return "general_analysis"


def keyword_route(question: str) -> dict[str, Any]:
    """Pure keyword-based routing - always works, never raises."""
    neighborhoods = _extract_neighborhoods(question)
    boroughs = _extract_boroughs(question)
    amenities = _extract_amenities(question)
    intent = _classify_intent(question, neighborhoods, amenities)
    min_bedrooms, exact_bedrooms = _extract_bedrooms(question)
    return {
        "intent": intent,
        "needs_sql": True,
        "needs_chart": True,
        "needs_map": intent in (
            "rental_search",
            "amenity_search",
            "transit_analysis",
            "neighborhood_comparison",
        ),
        "filters": {
            "max_rent": _extract_max_rent(question),
            "min_bedrooms": min_bedrooms,
            "exact_bedrooms": exact_bedrooms,
            "neighborhoods": neighborhoods,
            "boroughs": boroughs,
            "amenities": amenities,
            "proximity": _extract_proximity(question),
            "safety_preference": _extract_safety_pref(question),
            "transit_preference": _extract_transit_pref(question),
            "sort_preference": _extract_sort_pref(question),
            "ranking_signal": _extract_ranking_signal(question),
            "opt_outs": _extract_opt_outs(question),
        },
    }


# Fields where the regex extractor is more trustworthy than a small local LLM.
# When the regex finds a value AND it disagrees with the LLM, the regex value
# wins for that field only. Other fields (intent, amenities, preferences, etc.)
# trust the LLM unconditionally because the LLM is better at intent.
_FACT_VETO_FIELDS: tuple[str, ...] = (
    "max_rent",
    "min_bedrooms",
    "exact_bedrooms",
    "neighborhoods",
    "boroughs",
    "opt_outs",
)


def _build_router_prompt(question: str) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the LLM-first router."""
    system = (
        "You are a routing classifier for a renter intelligence dashboard. "
        "Return ONLY valid JSON matching the requested schema. "
        "Do not invent fields, do not include explanations, do not wrap in markdown."
    )
    schema_hint = (
        '{"intent": one of '
        + ", ".join(repr(i) for i in INTENTS)
        + ', "needs_sql": true, "needs_chart": true, "needs_map": true|false, '
        '"filters": {"max_rent": int|null (500..20000), '
        '"min_bedrooms": int|null (0..6), "exact_bedrooms": int|null (0..6), '
        '"neighborhoods": [str], "amenities": [str], '
        '"proximity": [{"target": str, "max_distance_miles": float, '
        '"kind": "amenity"|"transit"}], '
        '"safety_preference": "low_crime"|"high_safety"|null, '
        '"transit_preference": "bus"|"subway"|"any"|null, '
        '"sort_preference": "rent"|"safety"|"availability"|"renter_fit"|"transit"|"green_space"|null, '
        '"ranking_signal": bool}}'
    )
    prompt = (
        f"User question: {question}\n\n"
        f"Schema: {schema_hint}\n\n"
        "Return JSON only."
    )
    return system, prompt


def _values_disagree(regex_value: Any, llm_value: Any) -> bool:
    """Return True when the regex value is non-empty and contradicts the LLM."""
    if regex_value in (None, "", []):
        return False
    if isinstance(regex_value, list):
        if not regex_value:
            return False
        if not isinstance(llm_value, list):
            return True
        return set(map(str, regex_value)) != set(map(str, llm_value))
    return regex_value != llm_value


def _apply_fact_veto(llm_dict: dict[str, Any], kw_dict: dict[str, Any]) -> dict[str, Any]:
    """Override LLM filters with regex values on disagreement for vetoed fields."""
    llm_filters = dict(llm_dict.get("filters") or {})
    kw_filters = kw_dict.get("filters") or {}
    for field in _FACT_VETO_FIELDS:
        regex_value = kw_filters.get(field)
        llm_value = llm_filters.get(field)
        if _values_disagree(regex_value, llm_value):
            llm_filters[field] = regex_value
    out = dict(llm_dict)
    out["filters"] = llm_filters
    return out


def router_agent(question: str, llm: Optional[LLMClient] = None) -> dict[str, Any]:
    """Classify the user's intent and pull out structured filters.

    LLM-first: the model is asked for a structured ``RouterOutput`` and the
    keyword extractor only kicks in when the LLM is unavailable, returns
    invalid JSON, or contradicts the regex on a fact-veto field. Always
    returns a valid router dict - the public shape is identical to the
    historical keyword-only version.
    """
    kw = keyword_route(question)

    if llm is None or llm.provider == "mock":
        return kw

    system, prompt = _build_router_prompt(question)
    try:
        parsed, _ = llm.generate_validated(
            prompt=prompt,
            system_prompt=system,
            model_cls=RouterOutput,
            temperature=0.0,
        )
    except Exception:
        parsed = None

    if parsed is None:
        return kw

    return _apply_fact_veto(parsed.to_dict(), kw)
