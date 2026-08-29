# `data/`

| File | Committed? | What it is |
|---|---|---|
| `public_set.jsonl` | Yes | The official 200-session labeled public dev set, copied unmodified from the participant kit. |
| `sample_catalog.jsonl` | Yes | A **synthetic** ~40-item catalog we generated (see `scripts/` history) for unit/integration tests and quick smoke runs. Not real Amazon data, not the graded catalog — `parent_asin` values look like `SYN00001` specifically so they can never be mistaken for real ASINs. |
| `catalog.jsonl` | **No** (gitignored) | The real, frozen 50,000-item organizer catalog. Download it yourself — see below. |
| `catalog.jsonl.gz` | **No** (gitignored) | The compressed download, before decompression. |
| `embeddings.npy` / `embeddings_meta.json` | **No** (gitignored) | Cached dense embeddings for the real catalog, built by `scripts/build_index.py`. Regenerate locally; not redistributed. |
| `README_KIT.md` | Yes | The participant kit's own `data/README.md`, copied unmodified for reference. |

## Getting the real catalog

```bash
python3 scripts/download_catalog.py --url <catalog.jsonl.gz release asset URL> [--sha256 <published SHA256>]
python3 scripts/build_index.py --catalog data/catalog.jsonl
```

The release asset URL and its SHA256SUMS entry are published on the
[participant-kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit).
We do not mirror `catalog.jsonl` in this repository — see
`DATA_ATTRIBUTION.md` for why, and `src/catalog/loader.py` for how
checksum verification is applied on load once you have a published hash.

Never place API keys, private evaluation data, or participant outputs in
this directory.
