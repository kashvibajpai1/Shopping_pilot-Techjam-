"""Dialog state tracker (slot manager) — build-brief section C.

The `DialogState` object is the single source of truth for "what does the
user currently want". Every turn:
  1. `decay()` ages every slot that was not reinforced last turn.
  2. `merge_turn(text, turn)` extracts new slot values from the raw text
     (regex for structured values, keyword vocab for freer ones) and merges
     them in. Because each slot name holds exactly one active value, a new
     extraction for an already-held slot *replaces* it — which is what
     "clear the conflicting slot rather than stacking" reduces to. Explicit
     override language ("actually", "never mind", "instead", ...) is still
     detected separately so it can (a) reset confidence/consumed-count for
     the replaced slot cleanly and (b) be logged as an intent-override
     event for the session profile's strategy log.

Slot names line up 1:1 with the contract's `ask_attribute` enum
(category, material, color, size, style, brand, budget, feature, use_case)
plus an internal `other` bucket for anything unclassified, so a slot name
can be handed straight to `ask_attribute` with no translation layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.catalog.loader import COLOR_VOCAB, MATERIAL_VOCAB

SLOT_NAMES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)

CONFIDENT_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.15
DECAY_FACTOR = 0.82
REINFORCE_BOOST = 0.3
INITIAL_CONFIDENCE = 0.65

CATEGORY_VOCAB = (
    "shoes", "sneakers", "boots", "sandals", "heels", "flats", "loafers",
    "shirt", "t-shirt", "blouse", "dress", "jacket", "coat", "jeans",
    "pants", "trousers", "shorts", "skirt", "leggings", "sweater",
    "hoodie", "cardigan", "suit", "swimsuit", "romper", "jumpsuit",
    "necklace", "ring", "earrings", "bracelet", "watch", "anklet",
    "bag", "backpack", "purse", "wallet", "hat", "cap", "scarf",
    "socks", "belt", "gloves", "sunglasses", "tie",
)
STYLE_VOCAB = (
    "casual", "formal", "athletic", "vintage", "bohemian", "classic",
    "sporty", "elegant", "minimalist", "trendy", "chic", "professional",
    "streetwear", "preppy", "edgy",
)
USE_CASE_VOCAB = (
    "running", "hiking", "gym", "workout", "work", "travel", "wedding",
    "party", "everyday", "outdoor", "winter", "summer", "beach", "office",
    "school", "date night", "vacation", "camping", "walking", "yoga",
)
FEATURE_VOCAB = (
    "waterproof", "water-resistant", "water resistant", "breathable",
    "lightweight", "non-slip", "slip-resistant", "adjustable", "stretchy",
    "machine washable", "wrinkle-free", "moisture-wicking", "insulated",
    "reversible", "packable",
)

_vocab_re = lambda words: re.compile(  # noqa: E731
    r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE
)
MATERIAL_RE = _vocab_re(MATERIAL_VOCAB)
COLOR_RE = _vocab_re(COLOR_VOCAB)
CATEGORY_RE = _vocab_re(CATEGORY_VOCAB)
STYLE_RE = _vocab_re(STYLE_VOCAB)
USE_CASE_RE = _vocab_re(USE_CASE_VOCAB)
FEATURE_RE = _vocab_re(FEATURE_VOCAB)
SIZE_RE = re.compile(
    r"\bsize\s+([a-z0-9.]+)\b|\b(xx-small|xx-large|x-small|x-large|small|"
    r"medium|large|xxs|xs|xxl|xl)\b",
    re.IGNORECASE,
)
BRAND_RE = re.compile(
    r"\bbrand(?:\s+is|\s*[:,]|\s+like|\s+called)?\s+([A-Z][\w&'\-]*(?:\s+[A-Z][\w&'\-]*){0,2})"
    r"|\bby\s+([A-Z][\w&'\-]*(?:\s+[A-Z][\w&'\-]*){0,2})\b"
)
OVERRIDE_RE = re.compile(
    r"\b(actually|never\s*mind|instead|not that|scratch that|forget (?:that|it)|"
    r"change (?:my|that) (?:mind|to)|ignore (?:my|that|the) (?:earlier|previous|last))\b",
    re.IGNORECASE,
)
PRICE_RANGE_RE = re.compile(
    r"\$?\s*(\d+(?:\.\d+)?)\s*(?:-|to|and)\s*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
PRICE_UNDER_RE = re.compile(
    r"(?:under|below|less than|up to|no more than|max(?:imum)?(?: of)?|<=?)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PRICE_OVER_RE = re.compile(
    r"(?:over|above|more than|at least|min(?:imum)?(?: of)?|>=?)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PRICE_AROUND_RE = re.compile(
    r"(?:budget(?: of| around| is)?|around|about)\s*\$?\s*(\d+(?:\.\d+)?)"
    r"|\$\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass
class Slot:
    value: str
    confidence: float = INITIAL_CONFIDENCE
    confirmed_count: int = 1
    turns_since_reinforced: int = 0
    first_turn: int = 1
    last_turn: int = 1


@dataclass
class ExtractionEvent:
    turn: int
    slot: str
    old_value: Optional[str]
    new_value: str
    is_override: bool
    reinforced: bool


@dataclass
class DialogState:
    slots: dict[str, Slot] = field(default_factory=dict)
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    budget_slot: Optional[Slot] = None
    free_text_history: list[str] = field(default_factory=list)
    turn: int = 0
    last_ask_attribute: Optional[str] = None
    consecutive_same_ask: int = 0
    override_events: list[ExtractionEvent] = field(default_factory=list)
    extraction_log: list[ExtractionEvent] = field(default_factory=list)

    # -- decay -------------------------------------------------------
    def decay(self, reinforced_this_turn: set[str]) -> None:
        for name, slot in list(self.slots.items()):
            if name in reinforced_this_turn:
                continue
            slot.turns_since_reinforced += 1
            slot.confidence = max(MIN_CONFIDENCE, slot.confidence * DECAY_FACTOR)
        if self.budget_slot is not None and "budget" not in reinforced_this_turn:
            self.budget_slot.turns_since_reinforced += 1
            self.budget_slot.confidence = max(MIN_CONFIDENCE, self.budget_slot.confidence * DECAY_FACTOR)

    # -- merging one turn of raw text ---------------------------------
    def merge_turn(self, text: str, turn: int) -> list[ExtractionEvent]:
        self.turn = turn
        text = text or ""
        self.free_text_history.append(text)
        is_override_turn = bool(OVERRIDE_RE.search(text))

        extracted: dict[str, str] = {}
        m = MATERIAL_RE.search(text)
        if m:
            extracted["material"] = m.group(1).lower()
        c = COLOR_RE.search(text)
        if c:
            extracted["color"] = c.group(1).lower()
        cat = CATEGORY_RE.search(text)
        if cat:
            extracted["category"] = cat.group(1).lower()
        style = STYLE_RE.search(text)
        if style:
            extracted["style"] = style.group(1).lower()
        use_case = USE_CASE_RE.search(text)
        if use_case:
            extracted["use_case"] = use_case.group(1).lower()
        feature = FEATURE_RE.search(text)
        if feature:
            extracted["feature"] = feature.group(1).lower()
        size = SIZE_RE.search(text)
        if size:
            extracted["size"] = (size.group(1) or size.group(2) or "").lower()
        brand = BRAND_RE.search(text)
        if brand:
            extracted["brand"] = (brand.group(1) or brand.group(2) or "").strip()

        reinforced: set[str] = set()
        events: list[ExtractionEvent] = []
        for name, value in extracted.items():
            if not value:
                continue
            event = self._apply_slot(name, value, turn, is_override_turn)
            events.append(event)
            reinforced.add(name)

        price_event = self._apply_price(text, turn, is_override_turn)
        if price_event:
            events.append(price_event)
            reinforced.add("budget")

        self.decay(reinforced)
        self.extraction_log.extend(events)
        self.override_events.extend(e for e in events if e.is_override)
        return events

    def _apply_slot(self, name: str, value: str, turn: int, is_override_turn: bool) -> ExtractionEvent:
        existing = self.slots.get(name)
        if existing is None:
            self.slots[name] = Slot(value=value, first_turn=turn, last_turn=turn)
            return ExtractionEvent(turn, name, None, value, is_override_turn, reinforced=False)
        same_value = existing.value == value
        old_value = existing.value
        if is_override_turn and not same_value:
            self.slots[name] = Slot(value=value, first_turn=turn, last_turn=turn)
            return ExtractionEvent(turn, name, old_value, value, True, reinforced=False)
        if same_value:
            existing.confidence = min(1.0, existing.confidence + REINFORCE_BOOST)
            existing.confirmed_count += 1
            existing.turns_since_reinforced = 0
            existing.last_turn = turn
            return ExtractionEvent(turn, name, old_value, value, False, reinforced=True)
        # New (non-override) mention of a slot that already has a different
        # value: still replace — a slot holds one active value — but do not
        # grant the override confidence reset, since no explicit override
        # language was present (mild ambiguity signal).
        self.slots[name] = Slot(value=value, confidence=INITIAL_CONFIDENCE, first_turn=turn, last_turn=turn)
        return ExtractionEvent(turn, name, old_value, value, False, reinforced=False)

    def _apply_price(self, text: str, turn: int, is_override_turn: bool) -> Optional[ExtractionEvent]:
        new_min, new_max = None, None
        range_match = PRICE_RANGE_RE.search(text)
        under_match = PRICE_UNDER_RE.search(text)
        over_match = PRICE_OVER_RE.search(text)
        around_match = PRICE_AROUND_RE.search(text)
        if range_match:
            lo, hi = float(range_match.group(1)), float(range_match.group(2))
            new_min, new_max = min(lo, hi), max(lo, hi)
        elif under_match:
            new_max = float(under_match.group(1))
        elif over_match:
            new_min = float(over_match.group(1))
        elif around_match:
            value = float(around_match.group(1) or around_match.group(2))
            new_min, new_max = max(0.0, value * 0.7), value * 1.3

        if new_min is None and new_max is None:
            return None

        old_value = None
        if self.budget_slot is not None:
            old_value = f"{self.price_min}-{self.price_max}"
        reinforced = self.budget_slot is not None and (self.price_min, self.price_max) == (new_min, new_max)
        self.price_min, self.price_max = new_min, new_max
        label = f"{new_min if new_min is not None else 0:.0f}-{new_max if new_max is not None else '∞'}"
        if reinforced and self.budget_slot:
            self.budget_slot.confidence = min(1.0, self.budget_slot.confidence + REINFORCE_BOOST)
            self.budget_slot.confirmed_count += 1
            self.budget_slot.turns_since_reinforced = 0
            self.budget_slot.last_turn = turn
        else:
            self.budget_slot = Slot(value=label, first_turn=turn, last_turn=turn)
        return ExtractionEvent(turn, "budget", old_value, label, is_override_turn, reinforced)

    # -- reads used by retrieval/router/clarification -----------------
    def confident_slots(self, threshold: float = CONFIDENT_THRESHOLD) -> dict[str, Slot]:
        result = {name: slot for name, slot in self.slots.items() if slot.confidence >= threshold}
        if self.budget_slot is not None and self.budget_slot.confidence >= threshold:
            result["budget"] = self.budget_slot
        return result

    def n_confident_slots(self, threshold: float = CONFIDENT_THRESHOLD) -> int:
        return len(self.confident_slots(threshold))

    def query_text(self) -> str:
        parts = [slot.value for slot in self.slots.values()]
        if self.budget_slot:
            parts.append(self.budget_slot.value)
        parts.extend(self.free_text_history[-2:])  # bounded recency window
        return " ".join(p for p in parts if p)

    def has_any_hard_constraint(self) -> bool:
        return bool(self.slots) or self.price_min is not None or self.price_max is not None
