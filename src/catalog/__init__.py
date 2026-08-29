"""Catalog & index layer: frozen, read-only product data structures."""
from src.catalog.loader import Catalog, CatalogChecksumError, load_catalog

__all__ = ["Catalog", "CatalogChecksumError", "load_catalog"]
