"""Official Agent interface implementation (docs/agent_api_contract.json).

This is the class the participant kit's evaluator imports and instantiates
as `Agent(catalog_path)`. It owns catalog loading (with optional checksum
verification and cached embeddings) and delegates all turn logic to
`Orchestrator`, whose central error boundary guarantees `respond()` never
raises — every internal failure degrades to a safe BM25-only response.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.catalog.loader import load_catalog
from src.orchestrator import Orchestrator
from src.retrieval.vector_index import VectorIndex

logger = logging.getLogger("techjam.agent")

DEFAULT_CATALOG_PATH = "data/catalog.jsonl"
DEFAULT_EMBEDDINGS_PATH = "data/embeddings.npy"
DEFAULT_EMBEDDINGS_META_PATH = "data/embeddings_meta.json"


class Agent:
    def __init__(self, catalog_path: str | Path = DEFAULT_CATALOG_PATH) -> None:
        self.catalog = load_catalog(catalog_path)
        vector_index = VectorIndex.load_cached(
            self.catalog, DEFAULT_EMBEDDINGS_PATH, DEFAULT_EMBEDDINGS_META_PATH
        )
        self._orchestrator = Orchestrator(self.catalog, vector_index=vector_index)

    def reset(self, session_id: str, user_profile: dict) -> None:
        try:
            self._orchestrator.reset(session_id, user_profile)
        except Exception:  # noqa: BLE001 - reset() must never raise either
            logger.exception("reset() failed for session %s — continuing with a blank profile.", session_id)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._orchestrator.respond(session_id, user_message, turn, top_k)
