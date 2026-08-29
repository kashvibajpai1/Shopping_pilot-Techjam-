"""Pydantic schemas mirroring docs/agent_api_contract.json.

Every payload that crosses the Agent boundary (incoming turn/reset requests,
outgoing responses) is validated here. Validation failures never raise past
the Agent — callers get a coerced, safe default instead. See
`src/orchestrator.py` for how these are used as the input-validation layer
required by the security & robustness spec.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ALLOWED_ASK_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")  # tolerate additional organizer fields without crashing

    purchase_frequency: str = ""
    average_prior_rating: Optional[float] = None
    rating_style: str = ""
    preference_tags: list[str] = Field(default_factory=list)
    summary: str = ""


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    parent_asin: str
    score: Optional[float] = None


class TurnResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str = ""
    ask_attribute: Optional[str] = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    usage: Optional[dict] = None

    def as_contract_dict(self) -> dict[str, Any]:
        """Serialize back to the exact wire shape the evaluator expects."""
        ask = self.ask_attribute if self.ask_attribute in ALLOWED_ASK_ATTRIBUTES else None
        payload: dict[str, Any] = {
            "message": self.message,
            "ask_attribute": ask,
            "recommendations": [
                {"parent_asin": rec.parent_asin} for rec in self.recommendations[:100]
            ],
        }
        if self.usage is not None:
            payload["usage"] = self.usage
        return payload


def safe_parse_user_profile(raw: Any) -> UserProfile:
    """Never raises. Malformed/partial profiles degrade to safe defaults."""
    if not isinstance(raw, dict):
        return UserProfile()
    try:
        return UserProfile(**raw)
    except ValidationError:
        # Best-effort coercion: keep whatever fields are individually valid.
        cleaned: dict[str, Any] = {}
        for key in ("purchase_frequency", "rating_style", "summary"):
            value = raw.get(key)
            if isinstance(value, str):
                cleaned[key] = value
        tags = raw.get("preference_tags")
        if isinstance(tags, list):
            cleaned["preference_tags"] = [str(t) for t in tags]
        rating = raw.get("average_prior_rating")
        if isinstance(rating, (int, float)):
            cleaned["average_prior_rating"] = float(rating)
        return UserProfile(**cleaned)


def safe_user_message(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if raw is None:
        return ""
    return str(raw)


def safe_turn(raw: Any) -> int:
    try:
        turn = int(raw)
    except (TypeError, ValueError):
        return 1
    return max(1, min(10, turn))


def safe_top_k(raw: Any) -> int:
    try:
        top_k = int(raw)
    except (TypeError, ValueError):
        return 10
    return max(1, min(10, top_k))
