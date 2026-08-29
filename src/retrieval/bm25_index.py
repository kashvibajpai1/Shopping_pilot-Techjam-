"""Route 1: BM25 keyword search over title + description + attributes.

Prefers `rank_bm25` (pinned in requirements.txt) but degrades to a small
pure-Python/NumPy BM25 implementation if the dependency is missing at
runtime, so a broken install never takes retrieval to zero — it only loses
a bit of speed/quality. This is the "fail-safe fallback" pattern applied to
a dependency, not just to a network call.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Sequence

from src.catalog.loader import Catalog, tokenize

logger = logging.getLogger("techjam.retrieval.bm25")

try:
    from rank_bm25 import BM25Okapi as _RankBM25Okapi
except ImportError:  # pragma: no cover - exercised only when dependency missing
    _RankBM25Okapi = None
    logger.warning("rank_bm25 not installed; using built-in fallback BM25 implementation.")


class _FallbackBM25:
    """Minimal BM25Okapi-compatible scorer (k1=1.5, b=0.75), numpy-free."""

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_freqs: list[Counter] = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if corpus else 0.0
        self.n_docs = len(corpus)
        df: Counter = Counter()
        for doc in corpus:
            for term in set(doc):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query: Sequence[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        for i, (freqs, dl) in enumerate(zip(self.doc_freqs, self.doc_lens)):
            score = 0.0
            for term in query:
                idf = self.idf.get(term)
                if not idf:
                    continue
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores[i] = score
        return scores


class BM25Index:
    """BM25 index built once from the frozen catalog corpus."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        impl = _RankBM25Okapi or _FallbackBM25
        self._bm25 = impl(catalog.corpus)

    def search(self, query_text: str, top_n: int = 100) -> list[tuple[str, float]]:
        """Return up to `top_n` (parent_asin, score) pairs, best first."""
        terms = tokenize(query_text)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(
            zip(self.catalog.ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return [(pid, score) for pid, score in ranked[:top_n] if score > 0]
