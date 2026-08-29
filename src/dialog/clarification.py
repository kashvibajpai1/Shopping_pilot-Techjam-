"""Clarification generator — build-brief section F.

Picks the single most under-specified, most *discriminative* missing slot
to ask about: among slot types not yet held with confidence, we look at
how much the values of that attribute vary across the current candidate
pool (normalized entropy) and ask about whichever one would split the pool
most evenly — a generic "what are you looking for?" is never returned.

Guards against asking about the same unresolved slot twice in a row (the
orchestrator is responsible for telling us when that's the situation, via
`avoid_slot`) — repeating an identical question burns a turn without
gaining information, so we pivot to the next-best candidate slot, or
signal "no good clarification available" so the orchestrator forces a
best-effort ranked answer instead.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from src.catalog.loader import Catalog
from src.dialog.state_tracker import DialogState

QUESTION_TEMPLATES: dict[str, str] = {
    "category": "What kind of item are you looking for exactly?",
    "material": "Do you have a material preference, like cotton, leather, or something synthetic?",
    "color": "Any color you'd like it in?",
    "size": "What size do you need?",
    "style": "What style are you going for — casual, formal, athletic, something else?",
    "brand": "Do you have a brand in mind, or are you open to any?",
    "budget": "What's your budget range for this?",
    "feature": "Is there a specific feature that matters most, like water-resistance or breathability?",
    "use_case": "What will you mainly be using this for?",
    "other": "Could you tell me a bit more about what you're looking for?",
}

_ATTRIBUTE_GETTERS = {
    "category": lambda p: p.category_tokens[-1] if p.category_tokens else None,
    "material": lambda p: next(iter(p.materials), None),
    "color": lambda p: next(iter(p.colors), None),
    "size": lambda p: next(iter(p.sizes), None),
    "brand": lambda p: p.store.lower() if p.store else None,
    "budget": lambda p: p.price_bucket,
}
_FREE_TEXT_SLOTS = ("style", "use_case", "feature")


def _normalized_entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    distinct = len(counts)
    if distinct <= 1 or total == 0:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(distinct)
    return entropy / max_entropy if max_entropy else 0.0


def most_discriminative_missing_slot(
    candidate_ids: list[str],
    catalog: Catalog,
    state: DialogState,
    avoid_slot: Optional[str] = None,
) -> Optional[str]:
    confident = state.confident_slots()
    products = [catalog.by_id[pid] for pid in candidate_ids if pid in catalog.by_id]

    scores: dict[str, float] = {}
    for name, getter in _ATTRIBUTE_GETTERS.items():
        if name in confident:
            continue
        values = [v for v in (getter(p) for p in products) if v]
        score = _normalized_entropy(values)
        if score > 0:
            scores[name] = score

    for name in _FREE_TEXT_SLOTS:
        if name not in confident and name not in scores:
            scores.setdefault(name, 0.25)  # flat baseline: cheap to ask, can't estimate entropy locally

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for name, _score in ranked:
        if name != avoid_slot:
            return name
    # Every candidate slot is the one we just asked about and it's still
    # unresolved: never re-ask the identical question — signal "nothing to
    # clarify" so the caller forces a best-effort answer instead.
    return None if avoid_slot is not None else (ranked[0][0] if ranked else None)


def build_clarification(
    candidate_ids: list[str],
    catalog: Catalog,
    state: DialogState,
    avoid_slot: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Returns (ask_attribute, message) or None if nothing useful to ask."""
    slot_name = most_discriminative_missing_slot(candidate_ids, catalog, state, avoid_slot=avoid_slot)
    if slot_name is None:
        return None
    return slot_name, QUESTION_TEMPLATES.get(slot_name, QUESTION_TEMPLATES["other"])
