from __future__ import annotations

from pathlib import Path

import pytest

from src.catalog.loader import Catalog, load_catalog

SAMPLE_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_catalog.jsonl"
REAL_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.jsonl"


@pytest.fixture(scope="session")
def sample_catalog() -> Catalog:
    """The small synthetic catalog checked into data/sample_catalog.jsonl.

    Used by unit/integration tests so the suite runs without the real
    50,000-item organizer catalog. See tests/test_full_pipeline_eval.py for
    the (skipped-if-absent) real-catalog smoke test.
    """
    return load_catalog(SAMPLE_CATALOG_PATH)


@pytest.fixture(scope="session")
def real_catalog_path() -> Path:
    return REAL_CATALOG_PATH
