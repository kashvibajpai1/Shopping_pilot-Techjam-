"""Load the frozen catalog into read-only, in-memory structures.

Design notes (see build brief section A):
  * Catalog is loaded once per process and never mutated afterwards — every
    structure returned here is wrapped in an immutable container
    (tuple / MappingProxyType) so no downstream code path can write back
    into catalog data.
  * SHA256 verification is opt-in: pass `expected_sha256` (or set the
    TECHJAM_CATALOG_SHA256 env var) once you have the organizer-published
    checksum. When no checksum is configured we skip verification rather
    than fail the whole pipeline, but we log loudly that we did so, per the
    "fail loudly, don't fail silently" requirement — silence here would
    only ever be silent about a *missing check*, never about a *mismatch*.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

logger = logging.getLogger("techjam.catalog")

SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")

MATERIAL_VOCAB = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "suede", "linen", "cashmere", "canvas",
    "fleece", "mesh", "faux leather", "synthetic", "elastane", "acrylic",
)
COLOR_VOCAB = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "navy", "beige", "tan", "gold",
    "silver", "multicolor", "khaki", "maroon", "ivory", "cream",
)
SIZE_TOKEN_RE = re.compile(
    r"\b(xx-?small|xx-?large|x-?small|x-?large|small|medium|large|"
    r"xxs|xs|xxl|xl|\d{1,2}(?:\.\d)?(?:w|wide|narrow)?)\b",
    re.IGNORECASE,
)
MATERIAL_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in MATERIAL_VOCAB) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in COLOR_VOCAB) + r")\b", re.I)
TOKEN_RE = re.compile(r"[a-z0-9]+")

EXCLUDED_CATEGORY_TOKENS = {
    # Only the umbrella category node itself is excluded — "Shoes" or
    # "Jewelry" as a standalone coarse category (one level below the
    # umbrella) is a legitimate, useful filter value. Matches the kit's own
    # coarse_category() exclusion set in evaluator/local_evaluator.py.
    "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
}

PRICE_BUCKETS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, float("inf")))


class CatalogChecksumError(RuntimeError):
    """Raised when a configured catalog checksum does not match the file on disk."""


def _flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def searchable_text(product: dict) -> str:
    parts = [_flatten_text(product.get(field)) for field in SEARCH_FIELDS]
    return " ".join(p for p in parts if p).strip()


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]


def price_bucket(price: Optional[float]) -> str:
    if price is None:
        return "unknown"
    for low, high in PRICE_BUCKETS:
        if low <= price < high:
            return f"{low}-{high if high != float('inf') else '+'}"
    return "unknown"


def category_tokens(categories: object) -> list[str]:
    tokens: list[str] = []
    for value in (categories if isinstance(categories, list) else [categories]):
        if not value:
            continue
        for part in str(value).split(","):
            part = part.strip().lower()
            if part and part not in EXCLUDED_CATEGORY_TOKENS:
                tokens.append(part)
    return tokens


class Product:
    """Immutable view over one catalog row."""

    __slots__ = (
        "parent_asin", "title", "price", "average_rating", "rating_number",
        "store", "categories", "raw", "text", "tokens",
        "materials", "colors", "sizes", "category_tokens", "price_bucket",
    )

    def __init__(self, raw: dict):
        self.parent_asin = str(raw["parent_asin"])
        self.title = str(raw.get("title") or "")
        price = raw.get("price")
        self.price = float(price) if isinstance(price, (int, float)) else None
        rating = raw.get("average_rating")
        self.average_rating = float(rating) if isinstance(rating, (int, float)) else None
        rating_number = raw.get("rating_number")
        self.rating_number = int(rating_number) if isinstance(rating_number, (int, float)) else 0
        self.store = str(raw.get("store") or "")
        self.categories = raw.get("categories") or []
        self.raw = MappingProxyType(dict(raw))
        self.text = searchable_text(raw)
        self.tokens = tuple(tokenize(self.text))
        self.materials = frozenset(m.lower() for m in MATERIAL_RE.findall(self.text))
        self.colors = frozenset(c.lower() for c in COLOR_RE.findall(self.text))
        self.sizes = frozenset(s.lower() for s in SIZE_TOKEN_RE.findall(self.text))
        self.category_tokens = tuple(category_tokens(self.categories))
        self.price_bucket = price_bucket(self.price)


class Catalog:
    """Frozen, read-only catalog: products + attribute inverted indices."""

    def __init__(self, products: list[Product]):
        self.products: tuple[Product, ...] = tuple(products)
        self.by_id: Mapping[str, Product] = MappingProxyType(
            {p.parent_asin: p for p in self.products}
        )
        self.ids: tuple[str, ...] = tuple(p.parent_asin for p in self.products)
        self.corpus: tuple[tuple[str, ...], ...] = tuple(p.tokens for p in self.products)

        category_idx: dict[str, set[str]] = {}
        brand_idx: dict[str, set[str]] = {}
        material_idx: dict[str, set[str]] = {}
        color_idx: dict[str, set[str]] = {}
        size_idx: dict[str, set[str]] = {}
        price_idx: dict[str, set[str]] = {}

        for p in self.products:
            for tok in p.category_tokens:
                category_idx.setdefault(tok, set()).add(p.parent_asin)
            if p.store:
                brand_idx.setdefault(p.store.strip().lower(), set()).add(p.parent_asin)
            for m in p.materials:
                material_idx.setdefault(m, set()).add(p.parent_asin)
            for c in p.colors:
                color_idx.setdefault(c, set()).add(p.parent_asin)
            for s in p.sizes:
                size_idx.setdefault(s, set()).add(p.parent_asin)
            price_idx.setdefault(p.price_bucket, set()).add(p.parent_asin)

        def _freeze(d: dict[str, set[str]]) -> Mapping[str, frozenset]:
            return MappingProxyType({k: frozenset(v) for k, v in d.items()})

        self.category_index = _freeze(category_idx)
        self.brand_index = _freeze(brand_idx)
        self.material_index = _freeze(material_idx)
        self.color_index = _freeze(color_idx)
        self.size_index = _freeze(size_idx)
        self.price_index = _freeze(price_idx)

        # populated lazily by scripts/build_index.py or vector_index.py
        self.embeddings = None  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self.products)

    def __contains__(self, parent_asin: str) -> bool:
        return parent_asin in self.by_id


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_catalog(
    catalog_path: str | Path,
    expected_sha256: Optional[str] = None,
) -> Catalog:
    """Load `catalog_path` (JSONL, one product object per line) into a Catalog.

    `expected_sha256` overrides the `TECHJAM_CATALOG_SHA256` env var. If
    neither is set, checksum verification is skipped (logged, not silent).
    """
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog file not found at {path}. Download catalog.jsonl.gz from the "
            "participant-kit GitHub Release and decompress it to this path — see "
            "scripts/download_catalog.py and data/README.md."
        )

    checksum = expected_sha256 or os.environ.get("TECHJAM_CATALOG_SHA256")
    if checksum:
        actual = _compute_sha256(path)
        if actual.lower() != checksum.lower():
            raise CatalogChecksumError(
                f"Catalog checksum mismatch for {path}: expected {checksum}, got {actual}. "
                "Refusing to load a catalog that does not match the published checksum."
            )
        logger.info("Catalog checksum verified for %s", path)
    else:
        logger.warning(
            "No catalog checksum configured (TECHJAM_CATALOG_SHA256 unset) — "
            "skipping integrity verification for %s. Set the env var once the "
            "organizer-published SHA256SUMS value is known.",
            path,
        )

    products = [Product(raw) for raw in _iter_jsonl(path)]
    if not products:
        raise ValueError(f"Catalog at {path} is empty — refusing to build an empty index.")
    return Catalog(products)
