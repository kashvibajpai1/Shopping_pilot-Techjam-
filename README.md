# TechJam 2026 — Shopping Copilot (Track 4)

A multi-turn conversational shopping agent for the TechJam Conversational
E-Commerce Search Challenge: it reads a customer's chat turns, retrieves
candidates from a frozen 50,000-item Amazon `Clothing_Shoes_and_Jewelry`
catalog, and converges on the customer's hidden target product within a
hard 10-turn budget — entirely in-memory, no UI, no training.

This repository implements the official `Agent` interface
(`docs/agent_api_contract.json`) against the real participant kit at
[`TechJam2026/techjam-conversational-search`](https://github.com/TechJam2026/techjam-conversational-search)
(`participant-kit` release). `evaluator/local_evaluator.py`,
`docs/agent_api_contract.json`, `docs/evaluation_config.json`,
`docs/competition_specification.md`, `docs/submission_rules.md`,
`docs/baseline_results.json`, `DATA_ATTRIBUTION.md`, and
`data/public_set.jsonl` are copied **unmodified** from that kit; everything
under `src/`, `starter/agent.py`, `scripts/`, and `tests/` (excluding
`tests/test_kit_evaluator.py`, also copied unmodified) is this submission.

## Resolved critical unknowns

Before writing pipeline code we read the actual kit rather than guessing.
Findings, with the kit source that backs each one:

1. **Does the simulator adapt to clarifying questions?** Yes.
   `evaluator/local_evaluator.py:customer_reply()` reads the agent's
   `ask_attribute` from the previous turn and reveals a matching
   constraint from the (organizer-only) intent card if one hasn't been
   disclosed yet. Investing in the clarification generator genuinely pays
   off — it isn't a fixed transcript. This is why we built
   `src/dialog/clarification.py` to ask the single most *discriminative*
   missing slot rather than a generic question, and why the
   `ask_attribute` enum in `src/schemas.py` mirrors the contract's enum
   exactly (the simulator matches on that field, not on prose).
2. **Outbound network access for an LLM reranker in the grading sandbox?**
   Unknown and unverifiable from here, and `docs/submission_rules.md` says
   plainly: *"For official final scoring, organizer policy may disable
   network access."* We treat this as "assume no." The default reranker
   (`HeuristicReranker` in `src/ranking/llm_reranker.py`) and the default
   dense-embedding backend (`LocalHashEmbedder` in
   `src/retrieval/vector_index.py`) are both pure NumPy/stdlib, make zero
   network calls, and are what every number in this README was measured
   with. An LLM reranker and a real sentence-transformers embedding model
   are both wired in as optional upgrades (see "Model & network policy"
   below) that transparently no-op back to the offline path on any
   failure — never a crash, never a silent quality cliff across the whole
   private eval.
3. **How is `TechnicalScore` combined?** Read directly from
   `docs/evaluation_config.json`'s `recommended_composite` and confirmed in
   `evaluator/local_evaluator.py`:
   `TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency`,
   `Efficiency = clip((11 − MTTC) / 10, 0, 1)`. Hit rate dominates; MTTC
   matters but far less than getting the target into the top 10 at all —
   so the orchestrator is willing to spend a turn on a genuinely
   discriminative clarification, but never on turns 9–10 (see
   `src/orchestrator.py:FORCE_ANSWER_TURN`), and never on a question that
   won't shrink the candidate pool.
4. **Is Hit Rate measured every turn, or only at the end?** Every turn —
   `evaluator/local_evaluator.py:evaluate()` checks
   `target in ranked` and records the hit turn/rank on the *first* turn it
   appears, for every turn from 1 to 10. Early-turn retrieval quality
   directly moves MTTC, not just a final-turn snapshot.

## Architecture

```
User turn
   |
   v
[Intent Router]  rule-based Buying/Browsing + confidence, re-evaluated every turn
   |                         (src/router/intent_router.py)
   v
[Dialog State Tracker]  slot extraction, override detection, confidence decay
   |                         (src/dialog/state_tracker.py)
   v
[Multi-Route Retrieval + Fusion]
   BM25 (src/retrieval/bm25_index.py) + attribute filter with progressive
   relaxation (src/retrieval/attribute_filter.py) + dense cosine similarity
   (src/retrieval/vector_index.py) -> track-weighted RRF fusion, dynamic
   top-K, over-generality flag (src/retrieval/fusion.py)
   |
   +---- over-generality / no signal? ----+
   |  yes                                 |  no
   v                                      v
[Clarification Generator]          [Semantic Reranker]
 most-discriminative missing slot   HeuristicReranker (default, offline) or
 (src/dialog/clarification.py)      LLMReranker (opt-in, prompt-injection-
   |                                 safe, schema+membership validated)
   |                                 (src/ranking/llm_reranker.py)
   +------------------+-------------------+
                       |
                       v
        [Session Profile]  confirmed-slot boosts, track override, widen-pool
         (src/context/session_profile.py) — feeds back into fusion + rerank
                       |
                       v
        [Turn Orchestrator]  owns the 10-turn cap, central error boundary,
         per-session isolation (src/orchestrator.py)
                       |
                       v
        [Agent]  docs/agent_api_contract.json interface (src/agent.py)
```

## Repository layout

```text
data/
  public_set.jsonl        200 official labeled dev sessions (kit, unmodified)
  sample_catalog.jsonl    ~40-item SYNTHETIC catalog for tests/smoke runs
  catalog.jsonl            [gitignored] real 50k catalog — download it, see data/README.md
  embeddings.npy/.json     [gitignored] cached dense embeddings — build it
docs/                      kit docs (unmodified) + docs/demo_script.md (ours)
evaluator/local_evaluator.py   kit evaluator, unmodified
starter/agent.py           thin shim: `from src.agent import Agent`
src/
  agent.py                 official Agent interface implementation
  orchestrator.py           turn/session orchestrator, 10-turn cap, error boundary
  schemas.py                 pydantic input/output validation
  catalog/loader.py          load + checksum-verify + index the frozen catalog
  router/intent_router.py    Buying/Browsing classification
  dialog/state_tracker.py    slot manager: extraction, override, decay
  dialog/clarification.py    most-discriminative-slot question generator
  retrieval/{bm25_index,vector_index,attribute_filter,fusion}.py
  ranking/{llm_client,llm_reranker}.py   heuristic + optional LLM reranker
  context/session_profile.py  self-evolution: strategy log + behavior changes
scripts/
  download_catalog.py       fetch + verify the real catalog release asset
  build_index.py             precompute + cache dense embeddings
  run_eval.py                 thin wrapper around the official evaluator
tests/                        see "Testing" below
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ recommended (developed and tested on 3.11.15).

## Reproducing the reported numbers

1. Download the real catalog (see `data/README.md` for the release URL /
   checksum flow):
   ```bash
   python3 scripts/download_catalog.py --url <catalog.jsonl.gz release URL> --sha256 <published SHA256>
   ```
2. (Optional but recommended) precompute embeddings so `Agent.__init__`
   doesn't recompute them every run:
   ```bash
   python3 scripts/build_index.py
   ```
3. Run the official local evaluator against the 200-session public set:
   ```bash
   python3 -m evaluator.local_evaluator
   # or: python3 scripts/run_eval.py
   ```
   This writes `results.json` (per-session results + aggregate + per-scenario
   metrics) and prints the summary, exactly as the kit's own README
   documents. We have **not** run this against the real catalog ourselves in
   this environment — the grading sandbox we built in had no route to the
   kit's GitHub Release assets (git clone/fetch of the public repo worked;
   authenticated release-asset download did not). Every number we can state
   with confidence comes from `tests/` running the identical pipeline
   against the small synthetic catalog in `data/sample_catalog.jsonl`
   (42 items) — see "Testing" below. Anyone with access to the real
   `catalog.jsonl.gz` release asset can run steps 1–3 above to get the real
   `hit_rate_at_10` / `mrr` / `mttc` / `recommended_technical_score`
   against the actual 200 dev sessions; `tests/test_full_pipeline_eval.py`
   auto-detects `data/catalog.jsonl` and runs that same 200-session
   comparison as part of the test suite once it's present.

## Testing

```bash
pytest -q
```

48 tests (47 passing, 1 skipped pending the real catalog — see below), 0
network calls, in well under a second — covering
router accuracy on a hand-labeled set, slot extraction/override/decay,
retrieval fusion sanity (known-relevant items surface for a given slot
combination; a hard filter with zero matches progressively relaxes; a
gibberish query still returns a non-empty fallback pool), clarification
targeting (never re-asks an unresolved slot, never asks about an
already-confident one), LLM-reranker validation (hallucinated IDs dropped,
malformed JSON and client timeouts fall back to the deterministic order,
untrusted text never leaves the `<data>` tags), and orchestrator edge
cases: a session run out to exactly turn 10, an empty/one-word first turn,
an Intent Override on turn 1 before any state exists, a prompt-injection
attempt, a lookup that only ever returns catalog-valid IDs, back-to-back
sessions proving state isolation, `respond()` called before `reset()`, a
malformed user profile, a `None` message, out-of-range `turn`/`top_k`
values, and a reranker that raises mid-call.

`tests/test_full_pipeline_eval.py::test_real_public_set_end_to_end` is
`skipif`'d until `data/catalog.jsonl` exists locally; it runs the exact
`evaluator.evaluate()` call against all 200 real public sessions once it
does.

## Model & network policy

**Default configuration makes zero network calls and requires no API key.**
This was a deliberate response to critical unknown #2 above, not a
fallback bolted on afterward:

* Dense retrieval uses `LocalHashEmbedder` (deterministic signed
  feature-hashing over NumPy) by default. Set
  `TECHJAM_USE_SENTENCE_TRANSFORMERS=1` to use real
  `sentence-transformers/all-MiniLM-L6-v2` embeddings instead (requires the
  `sentence-transformers` extra and network access to fetch model weights
  the first time — do this offline via `scripts/build_index.py` ahead of
  grading, never at agent-init time in the eval loop).
* Semantic reranking uses `HeuristicReranker` (deterministic slot-match +
  price-fit scoring on top of the fusion order) by default. Set
  `TECHJAM_ENABLE_LLM=1` plus `ANTHROPIC_API_KEY` to enable `LLMReranker`
  (also used for the intent router's ambiguous-case escalation). Every
  external call is wrapped in a hard timeout (`TECHJAM_LLM_TIMEOUT_SECONDS`,
  default 6s) and a broad exception handler; on any failure — missing key,
  network error, timeout, malformed JSON, a hallucinated product ID — it
  falls back to the heuristic order for that turn. See
  `src/ranking/llm_client.py` and `src/ranking/llm_reranker.py`.
* See `.env.example` for the full list of toggles.

If you do enable the LLM path for your own local tuning against the 200
dev sessions, disclose the model, approximate cost, and latency in your
write-up per `docs/submission_rules.md` — we have not done so here because
we did not have a funded API key in the environment this was built in, and
because the default path is what every claim in this README is based on.

## Limitations (honest)

* **Never run against the real 50k catalog or the real public set in this
  environment.** All test coverage and manual verification used the
  42-item synthetic catalog in `data/sample_catalog.jsonl`. The retrieval
  weights, over-generality thresholds, and dynamic-K constants in
  `src/retrieval/fusion.py` are the build brief's own placeholder defaults,
  explicitly *not* tuned against the real 200 dev sessions — that
  calibration pass (holding out a slice of the 200 to avoid overfitting the
  public set, per the build brief) still needs to happen once the real
  catalog is available.
* **Attribute extraction is regex/vocabulary-based, not learned.** Material,
  color, size, style, use-case, and feature detection use fixed word lists
  (`src/dialog/state_tracker.py`). This is fast and fully offline but will
  miss synonyms and phrasing outside the vocabulary — a reasonable v1 given
  the "prompt engineering and lightweight local scoring, no fine-tuning"
  constraint, but a clear improvement target. Brand extraction is even
  cruder — a `store` field lookup plus a "brand is X" / "by X" regex — and
  will under-cover multi-brand or aliased products.
* **The LLM reranker path is implemented and unit-tested against a stub
  client, but never exercised against a real model in this environment**
  (no funded API key here). The fallback behavior (malformed JSON, dropped
  hallucinated IDs, timeout) is directly tested in
  `tests/test_llm_reranker.py`; the actual ranking quality of a real model
  is unverified.
* **Scope is Clothing/Shoes/Jewelry only**, per the frozen catalog. The
  architecture (route → filter → fuse → clarify-or-rank, with a
  self-adjusting session profile) generalizes to other retail verticals
  without a redesign, but the vocabulary lists in `state_tracker.py` are
  specific to this category and would need re-authoring for, say,
  electronics or groceries.
* **No real-money, real-inventory, or concurrency concerns** — out of
  scope per the challenge rules, and this codebase does not attempt them
  (single-user, in-memory, read-only catalog, as specified).

## Data attribution

See `DATA_ATTRIBUTION.md`. Catalog and sessions derive from Amazon Reviews
2023 (McAuley Lab, UCSD), `Clothing_Shoes_and_Jewelry` category, joined via
`parent_asin`. This repository does not redistribute the real catalog or
any private evaluation data — see `data/README.md`.

## Team contributions

_Fill in for your team's submission._
