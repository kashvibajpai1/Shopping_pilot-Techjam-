#!/usr/bin/env python3
"""Download and verify the real 50k-item frozen catalog.

The organizer distributes `catalog.jsonl.gz` as a GitHub Release asset on
https://github.com/TechJam2026/techjam-conversational-search (tag
`participant-kit`) rather than in the repository tree — see
DATA_ATTRIBUTION.md for why we don't redistribute it here.

Usage:
    python3 scripts/download_catalog.py --url <release-asset-url> [--sha256 <hex>]

If `--sha256` (or the TECHJAM_CATALOG_SHA256 env var) is provided, the
downloaded file is verified against it before being decompressed —
verification fails loudly (raises) on a mismatch, per the build-brief's
"fail loudly, not silently" requirement.

This script has no dependency on `src/` and only touches the network when
explicitly invoked — nothing in the pipeline itself calls it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True,
        help="Direct URL to catalog.jsonl.gz from the participant-kit GitHub Release.",
    )
    parser.add_argument(
        "--sha256", default=os.environ.get("TECHJAM_CATALOG_SHA256"),
        help="Expected SHA256 of the .gz file (falls back to TECHJAM_CATALOG_SHA256 env var).",
    )
    parser.add_argument("--out", default=str(DATA_DIR / "catalog.jsonl"))
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = DATA_DIR / "catalog.jsonl.gz"

    print(f"Downloading {args.url} -> {gz_path}")
    urllib.request.urlretrieve(args.url, gz_path)  # noqa: S310 - explicit, user-provided URL

    if args.sha256:
        actual = sha256_of(gz_path)
        if actual.lower() != args.sha256.lower():
            gz_path.unlink(missing_ok=True)
            print(f"CHECKSUM MISMATCH: expected {args.sha256}, got {actual}. Deleted the download.", file=sys.stderr)
            return 1
        print("Checksum verified.")
    else:
        print("WARNING: no --sha256 / TECHJAM_CATALOG_SHA256 provided — skipping integrity check.", file=sys.stderr)

    out_path = Path(args.out)
    with gzip.open(gz_path, "rb") as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"Decompressed to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
