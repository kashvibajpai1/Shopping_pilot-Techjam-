"""Multi-route fusion: BM25 + attribute filter + dense similarity → ranked pool.

Implements build-brief section D:
  * weighted Reciprocal Rank Fusion (RRF) across the three routes, with
    track-dependent weights (Buying leans on filter+keyword, Browsing
    leans on dense similarity for cross-category diversity),
  * dynamic top-K truncation as a function of how constrained the query is,
  * an over-generality flag (candidate pool still large while slots remain
    under-specified) that the orchestrator uses to route to clarification
    instead of the reranker,
  * a guaranteed non-empty fallback pool (sorted by a popularity prior) so
    a query that matches nothing in BM25/dense never dead-ends downstream.

All thresholds below are placeholders to calibrate against a held-out slice
of the 200 public dev sessions (see build-brief section D) — they are
exposed as module-level constants specifically so that tuning does not
require touching the fusion logic itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.catalog.loader import Catalog
from src.retrieval.attribute_filter import AttributeFilter, SlotQuery
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_index import VectorIndex

RRF_K = 60.0

TRACK_WEIGHTS = {
    "buying": {"bm25": 0.35, "dense": 0.15, "filter": 0.50},
    "browsing": {"bm25": 0.30, "dense": 0.55, "filter": 0.15},
}
RATING_PRIOR_WEIGHT = 0.001

K_TIGHT, K_MEDIUM, K_LOOSE = 10, 25, 50
OVER_GENERALITY_THRESHOLD = {"buying": 20, "browsing": 60}
CANDIDATE_POOL_CAP = 300
ROUTE_TOP_N = 150


@dataclass
class FusionResult:
    ranked: list[tuple[str, float]]
    dynamic_top_k: int
    over_generality: bool
    relaxed_slots: list[str]
    pool_size: int
    route_sizes: dict[str, int] = field(default_factory=dict)


def _popularity(catalog: Catalog, parent_asin: str) -> float:
    product = catalog.by_id.get(parent_asin)
    if not product:
        return 0.0
    rating = product.average_rating or 0.0
    return rating * math.log1p(product.rating_number)


def fuse(
    catalog: Catalog,
    bm25_index: BM25Index,
    vector_index: VectorIndex,
    attribute_filter: AttributeFilter,
    query_text: str,
    slot_queries: list[SlotQuery],
    price_min: Optional[float],
    price_max: Optional[float],
    track: str,
    n_confident_slots: int,
    widen_pool: bool = False,
) -> FusionResult:
    weights = TRACK_WEIGHTS.get(track, TRACK_WEIGHTS["browsing"])

    filter_result = attribute_filter.filter_with_relaxation(
        slot_queries, price_min, price_max, min_results=5
    )
    bm25_hits = bm25_index.search(query_text, top_n=ROUTE_TOP_N)
    dense_hits = vector_index.search(query_text, top_n=ROUTE_TOP_N)

    restrict = 0 < len(filter_result.candidate_ids) < len(catalog)

    def _restrict(hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
        if not restrict:
            return hits
        narrowed = [(pid, s) for pid, s in hits if pid in filter_result.candidate_ids]
        return narrowed or hits  # never let over-restriction erase a route entirely

    bm25_hits_r = _restrict(bm25_hits)
    dense_hits_r = _restrict(dense_hits)

    scores: dict[str, float] = {}

    def _add_route(hits: list[tuple[str, float]], weight: float) -> None:
        if weight <= 0:
            return
        for rank, (pid, _score) in enumerate(hits, start=1):
            scores[pid] = scores.get(pid, 0.0) + weight / (RRF_K + rank)

    _add_route(bm25_hits_r, weights["bm25"])
    _add_route(dense_hits_r, weights["dense"])

    if slot_queries:
        filter_universe = (
            {pid for pid, _ in bm25_hits_r}
            | {pid for pid, _ in dense_hits_r}
            | set(list(filter_result.candidate_ids)[:200])
        )
        filter_ranked = sorted(
            filter_universe,
            key=lambda pid: attribute_filter.soft_match_score(pid, slot_queries),
            reverse=True,
        )
        _add_route([(pid, 0.0) for pid in filter_ranked], weights["filter"])

    for pid in scores:
        scores[pid] += RATING_PRIOR_WEIGHT * _popularity(catalog, pid)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    if not ranked:
        # Absolute last resort: nothing matched BM25/dense at all (e.g. an
        # empty or gibberish turn). Fall back to the filtered pool (or the
        # whole catalog) ordered by a popularity prior, so downstream
        # stages always have *something* to rank/clarify against.
        fallback_pool = list(filter_result.candidate_ids) or list(catalog.ids)
        fallback_pool.sort(key=lambda pid: _popularity(catalog, pid), reverse=True)
        ranked = [(pid, 0.0) for pid in fallback_pool[:CANDIDATE_POOL_CAP]]

    pool_size = len(filter_result.candidate_ids) if restrict else len(ranked)

    if track == "buying":
        dynamic_top_k = K_TIGHT if n_confident_slots >= 3 else K_MEDIUM if n_confident_slots >= 1 else K_LOOSE
    else:
        dynamic_top_k = K_MEDIUM if n_confident_slots >= 2 else K_LOOSE

    threshold = OVER_GENERALITY_THRESHOLD.get(track, OVER_GENERALITY_THRESHOLD["browsing"])
    min_confident_required = 2 if track == "buying" else 1
    under_specified = n_confident_slots < min_confident_required
    over_generality = under_specified and pool_size > threshold

    if widen_pool:
        # Session profile signal: several Browsing turns without convergence —
        # favor returning a broader best-effort ranking over asking again.
        dynamic_top_k = max(dynamic_top_k, K_LOOSE)
        over_generality = under_specified and pool_size > threshold * 1.5

    return FusionResult(
        ranked=ranked[:CANDIDATE_POOL_CAP],
        dynamic_top_k=dynamic_top_k,
        over_generality=over_generality,
        relaxed_slots=filter_result.relaxed_slots,
        pool_size=pool_size,
        route_sizes={
            "bm25": len(bm25_hits),
            "dense": len(dense_hits),
            "filter_pool": len(filter_result.candidate_ids),
        },
    )
