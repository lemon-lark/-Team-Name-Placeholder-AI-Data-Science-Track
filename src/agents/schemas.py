"""Pydantic schemas for LLM-structured agent outputs.

Single source of truth for the JSON contract every LLM-first agent expects.
Keep these schemas in sync with the deterministic fallback functions in
:mod:`src.agents.router_agent` and :mod:`src.agents.clarification_agent` so
the rest of the pipeline can treat both code paths interchangeably.

Fields use ``Optional`` defaults and broad ``ge``/``le`` bounds so a small
local model (llama3.1:8b) can produce a partially-filled object without
tripping ``ValidationError``. The bounds are tight enough to reject
hallucinated values such as "max_rent: 25" or "exact_bedrooms: 99".
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Enum-like type aliases -------------------------------------------------------

IntentLiteral = Literal[
    "rental_search",
    "neighborhood_comparison",
    "availability_prediction",
    "amenity_search",
    "transit_analysis",
    "housing_supply_analysis",
    "general_analysis",
]

SafetyPreferenceLiteral = Literal["low_crime", "high_safety"]
TransitPreferenceLiteral = Literal["bus", "subway", "any"]
SortPreferenceLiteral = Literal[
    "rent", "safety", "availability", "renter_fit", "transit", "green_space"
]
ProximityKindLiteral = Literal["amenity", "transit"]


# --- Sub-models ---------------------------------------------------------------


class ProximityConstraint(BaseModel):
    """A single "near X within Y miles" constraint extracted from the question."""

    model_config = ConfigDict(extra="ignore")

    target: str = Field(..., min_length=1, max_length=64)
    max_distance_miles: float = Field(..., gt=0, le=10)
    kind: ProximityKindLiteral

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "max_distance_miles": float(self.max_distance_miles),
            "kind": self.kind,
        }


class Filters(BaseModel):
    """Structured filters extracted from the user question.

    Mirrors the dict produced by ``keyword_route`` in the router agent.
    """

    model_config = ConfigDict(extra="ignore")

    max_rent: Optional[int] = Field(default=None, ge=500, le=20000)
    min_bedrooms: Optional[int] = Field(default=None, ge=0, le=6)
    exact_bedrooms: Optional[int] = Field(default=None, ge=0, le=6)
    neighborhoods: List[str] = Field(default_factory=list)
    boroughs: List[str] = Field(default_factory=list)
    amenities: List[str] = Field(default_factory=list)
    proximity: List[ProximityConstraint] = Field(default_factory=list)
    safety_preference: Optional[SafetyPreferenceLiteral] = None
    transit_preference: Optional[TransitPreferenceLiteral] = None
    sort_preference: Optional[SortPreferenceLiteral] = None
    ranking_signal: bool = False
    opt_outs: List[str] = Field(default_factory=list)

    @field_validator("neighborhoods", "boroughs", "amenities", "opt_outs", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> Any:
        # Some small models emit a single string instead of a list.
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v

    @field_validator("proximity", mode="before")
    @classmethod
    def _coerce_proximity_list(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return v

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_rent": self.max_rent,
            "min_bedrooms": self.min_bedrooms,
            "exact_bedrooms": self.exact_bedrooms,
            "neighborhoods": list(self.neighborhoods),
            "boroughs": list(self.boroughs),
            "amenities": list(self.amenities),
            "proximity": [p.to_dict() for p in self.proximity],
            "safety_preference": self.safety_preference,
            "transit_preference": self.transit_preference,
            "sort_preference": self.sort_preference,
            "ranking_signal": bool(self.ranking_signal),
            "opt_outs": list(self.opt_outs),
        }


# --- Top-level agent outputs --------------------------------------------------


class RouterOutput(BaseModel):
    """LLM-first router agent output."""

    model_config = ConfigDict(extra="ignore")

    intent: IntentLiteral
    needs_sql: bool = True
    needs_chart: bool = True
    needs_map: bool = False
    filters: Filters = Field(default_factory=Filters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "needs_sql": bool(self.needs_sql),
            "needs_chart": bool(self.needs_chart),
            "needs_map": bool(self.needs_map),
            "filters": self.filters.to_dict(),
        }


class ClarificationDecision(BaseModel):
    """LLM-first clarification agent output.

    The deterministic ``count_dimensions`` check still runs as a sanity check
    on top of this; see :mod:`src.agents.clarification_agent`.
    """

    model_config = ConfigDict(extra="ignore")

    is_specific_enough: bool
    missing_dimensions: List[str] = Field(default_factory=list)
    follow_up_text: Optional[str] = None

    @field_validator("missing_dimensions", mode="before")
    @classmethod
    def _coerce_missing(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v

    @field_validator("follow_up_text", mode="before")
    @classmethod
    def _coerce_follow_up(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped or None
        return str(v)
