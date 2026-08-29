#!/usr/bin/env python3
"""One-time offline preprocessing: precompute + cache product embeddings.

BM25 and the attribute indices are cheap enough to rebuild at every process
start (see src/catalog/loader.py, src/retrieval/bm25_index.py) and are not
cached. Dense embeddings are the expensive part, so this script computes
them once for the whole catalog and writes them to
`data/embeddings.npy` (+ `data/embeddings_meta.json` recording the exact
catalog id order + embedder used), so `Agent.__init__` just loads the cache
instead of recomputing 50k embeddings on every run — see
`src/retrieval/vector_index.py:VectorIndex.load_cached`.

Usage:
    python3 scripts/build_index.py [--catalog data/catalog.jsonl]

By default this uses the offline hash embedder (no network, no model
download). Pass --sentence-transformers to use the real
sentence-transformers/all-MiniLM-L6-v2 model instead (requires the
`sentence-transformers` extra and, the first time, network access to fetch
model weights).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.catalog.loader import load_catalog  # noqa: E402
from src.retrieval.vector_index import (  # noqa: E402
    LocalHashEmbedder, SentenceTransformerEmbedder, VectorIndex,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DATA_DIR / "catalog.jsonl"))
    parser.add_argument("--embeddings-out", default=str(DATA_DIR / "embeddings.npy"))
    parser.add_argument("--meta-out", default=str(DATA_DIR / "embeddings_meta.json"))
    parser.add_argument(
        "--sentence-transformers", action="store_true",
        help="Use the real sentence-transformers model instead of the offline hash embedder.",
    )
    args = parser.parse_args()

    print(f"Loading catalog from {args.catalog} ...")
    catalog = load_catalog(args.catalog)
    print(f"Loaded {len(catalog)} products.")

    if args.sentence_transformers:
        print("Building sentence-transformers embeddings (requires network on first run) ...")
        embedder = SentenceTransformerEmbedder()
    else:
        print("Building offline hash embeddings (no network required) ...")
        embedder = LocalHashEmbedder()

    start = time.time()
    vector_index = VectorIndex(catalog, embedder=embedder)
    elapsed = time.time() - start
    print(f"Computed {vector_index.embeddings.shape} embeddings in {elapsed:.1f}s.")

    vector_index.save(args.embeddings_out, args.meta_out)
    print(f"Saved embeddings to {args.embeddings_out} and metadata to {args.meta_out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
