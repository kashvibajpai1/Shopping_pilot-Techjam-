"""Route 3: dense vector similarity (plain NumPy, in-memory, 50k scale is fine).

Two embedder backends, selected by `get_default_embedder()`:

  * `LocalHashEmbedder` (default) — a deterministic signed feature-hashing
    embedding (hashing trick + log-tf weighting + L2 normalize). Pure
    NumPy + hashlib, no model download, no network call, ever. This is the
    backend that keeps the dense route alive even in a fully offline
    grading sandbox (see build-brief section 0, critical unknown #2).
  * `SentenceTransformerEmbedder` (optional upgrade) — real
    `sentence-transformers/all-MiniLM-L6-v2` embeddings. Only used if
    explicitly enabled via `TECHJAM_USE_SENTENCE_TRANSFORMERS=1` *and* the
    package + model are available; any failure (not installed, no network
    to fetch model weights, OOM) is caught and we fall back to the hash
    embedder silently (logged), never raising.

Embeddings for the catalog are meant to be precomputed once by
`scripts/build_index.py` and cached to `data/embeddings.npy` (+ a sidecar
`data/embeddings_meta.json` recording the embedder name and the exact
catalog id order used) so a fresh agent process just loads the cache
instead of recomputing 50k embeddings on every run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from src.catalog.loader import Catalog, tokenize

logger = logging.getLogger("techjam.retrieval.vector")

HASH_EMBEDDER_NAME = "local_hash_v1"
SENTENCE_TRANSFORMER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str
    dim: int

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_one(self, text: str) -> np.ndarray: ...


class LocalHashEmbedder:
    """Deterministic, dependency-light (numpy only) sentence embedding."""

    name = HASH_EMBEDDER_NAME

    def __init__(self, dim: int = 256, salt: str = "techjam-hash-embed-v1"):
        self.dim = dim
        self.salt = salt.encode("utf-8")

    def _token_hash(self, token: str) -> int:
        digest = hashlib.blake2b(self.salt + token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little")

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            h = self._token_hash(token)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


class SentenceTransformerEmbedder:
    """Optional real-model backend. Only constructed when explicitly enabled."""

    name = SENTENCE_TRANSFORMER_NAME

    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_NAME):
        from sentence_transformers import SentenceTransformer  # deferred import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        embeddings = self._model.encode(
            list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


def get_default_embedder() -> Embedder:
    """Pick an embedder backend based on env config, always falling back safely."""
    if os.environ.get("TECHJAM_USE_SENTENCE_TRANSFORMERS", "").strip() in ("1", "true", "yes"):
        try:
            return SentenceTransformerEmbedder()
        except Exception:  # noqa: BLE001 - any failure degrades to the offline embedder
            logger.warning(
                "sentence-transformers requested but unavailable (not installed, no "
                "network to fetch weights, or failed to load) — falling back to the "
                "offline hash embedder.",
                exc_info=True,
            )
    return LocalHashEmbedder()


class VectorIndex:
    """Cosine-similarity search over precomputed (or lazily computed) product embeddings."""

    def __init__(self, catalog: Catalog, embedder: Optional[Embedder] = None, embeddings: Optional[np.ndarray] = None):
        self.catalog = catalog
        self.embedder = embedder or get_default_embedder()
        if embeddings is not None and embeddings.shape[0] == len(catalog):
            self.embeddings = embeddings
        else:
            self.embeddings = self.embedder.embed_batch([p.text for p in catalog.products])

    def search(self, query_text: str, top_n: int = 100) -> list[tuple[str, float]]:
        if not query_text.strip() or self.embeddings.shape[0] == 0:
            return []
        query_vec = self.embedder.embed_one(query_text)
        sims = self.embeddings @ query_vec
        top_n = min(top_n, sims.shape[0])
        top_idx = np.argpartition(-sims, top_n - 1)[:top_n] if top_n < sims.shape[0] else np.arange(sims.shape[0])
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self.catalog.ids[i], float(sims[i])) for i in top_idx if sims[i] > 0]

    def save(self, embeddings_path: str | Path, meta_path: str | Path) -> None:
        np.save(embeddings_path, self.embeddings)
        Path(meta_path).write_text(
            json.dumps({"embedder": self.embedder.name, "ids": list(self.catalog.ids)}),
            encoding="utf-8",
        )

    @classmethod
    def load_cached(cls, catalog: Catalog, embeddings_path: str | Path, meta_path: str | Path) -> "VectorIndex":
        """Load a cached embeddings matrix if it matches the current catalog exactly.

        Falls back to recomputing (offline hash embedder) if the cache is
        missing, unreadable, or built from a different catalog/order —
        never raises.
        """
        embeddings_path, meta_path = Path(embeddings_path), Path(meta_path)
        try:
            if embeddings_path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if tuple(meta.get("ids", [])) == catalog.ids:
                    embeddings = np.load(embeddings_path)
                    embedder = get_default_embedder()
                    logger.info("Loaded cached embeddings from %s (%s)", embeddings_path, meta.get("embedder"))
                    return cls(catalog, embedder=embedder, embeddings=embeddings)
                logger.warning("Cached embeddings id order does not match catalog — recomputing.")
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load cached embeddings — recomputing.", exc_info=True)
        return cls(catalog)
