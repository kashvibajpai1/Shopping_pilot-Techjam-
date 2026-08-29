"""Route 2: category/attribute hard filtering — dominant on the Buying track.

Provides:
  * exact/substring lookups against the catalog's inverted indices
    (O(1)-ish dict access, per build-brief section A),
  * `filter_with_relaxation`, which implements the "zero/near-zero
    candidate" handling from section D: if the strict intersection of all
    active slots is too small, progressively drop the least-confident slot
    and retry rather than ever returning an empty candidate pool,
  * `soft_match_score`, a continuous 0..1 signal (fraction of requested
    slots a product satisfies) used by fusion even when the hard filter
    has been relaxed away — every route's opinion still counts, it just
    stops being a strict gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from src.catalog.loader import Catalog

# Slot names that map onto a catalog inverted index built at load time.
INDEXED_SLOTS = ("category", "brand", "material", "color", "size")


@dataclass
class SlotQuery:
    name: str
    value: str
    confidence: float


@dataclass
class FilterResult:
    candidate_ids: frozenset[str]
    relaxed_slots: list[str] = field(default_factory=list)
    active_slots: list[str] = field(default_factory=list)  # slots actually applied (post-relaxation)
    requested_slots: list[SlotQuery] = field(default_factory=list)  # original, pre-relaxation


class AttributeFilter:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self._indexes: Mapping[str, Mapping[str, frozenset]] = {
            "category": catalog.category_index,
            "brand": catalog.brand_index,
            "material": catalog.material_index,
            "color": catalog.color_index,
            "size": catalog.size_index,
        }

    def candidates_for_slot(self, slot_name: str, value: str) -> Optional[frozenset[str]]:
        index = self._indexes.get(slot_name)
        if index is None or not value:
            return None
        needle = value.strip().lower()
        exact = index.get(needle)
        if exact:
            return exact
        matched: set[str] = set()
        for key, ids in index.items():
            if needle in key or key in needle:
                matched |= ids
        return frozenset(matched)

    def budget_candidates(self, price_min: Optional[float], price_max: Optional[float]) -> frozenset[str]:
        if price_min is None and price_max is None:
            return frozenset(self.catalog.ids)
        ids: set[str] = set()
        for p in self.catalog.products:
            if p.price is None:
                continue
            if price_min is not None and p.price < price_min:
                continue
            if price_max is not None and p.price > price_max:
                continue
            ids.add(p.parent_asin)
        return frozenset(ids)

    def filter_with_relaxation(
        self,
        slot_queries: Iterable[SlotQuery],
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        min_results: int = 5,
    ) -> FilterResult:
        requested = list(slot_queries)
        active = list(requested)
        has_budget = price_min is not None or price_max is not None
        relaxed: list[str] = []

        while True:
            sets: list[frozenset[str]] = []
            for slot in active:
                candidates = self.candidates_for_slot(slot.name, slot.value)
                if candidates is not None:
                    sets.append(candidates)
            if has_budget and "budget" not in relaxed:
                sets.append(self.budget_candidates(price_min, price_max))

            candidate_ids = frozenset.intersection(*sets) if sets else frozenset(self.catalog.ids)

            if len(candidate_ids) >= min_results or (not active and not has_budget):
                return FilterResult(
                    candidate_ids=candidate_ids,
                    relaxed_slots=relaxed,
                    active_slots=[s.name for s in active] + (["budget"] if has_budget else []),
                    requested_slots=requested,
                )

            # Relax the least-confident constraint next (budget relaxed last —
            # it is usually the most load-bearing hard constraint on Buying).
            if active:
                active.sort(key=lambda s: s.confidence)
                dropped = active.pop(0)
                relaxed.append(dropped.name)
            elif has_budget:
                relaxed.append("budget")
                has_budget = False
            else:
                return FilterResult(
                    candidate_ids=candidate_ids,
                    relaxed_slots=relaxed,
                    active_slots=[],
                    requested_slots=requested,
                )

    def soft_match_score(self, parent_asin: str, slot_queries: Iterable[SlotQuery]) -> float:
        """Fraction of the *originally requested* slots this product satisfies."""
        queries = list(slot_queries)
        if not queries:
            return 0.0
        matched = 0
        for slot in queries:
            candidates = self.candidates_for_slot(slot.name, slot.value)
            if candidates is not None and parent_asin in candidates:
                matched += 1
        return matched / len(queries)
