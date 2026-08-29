"""Integration test: full pipeline through the official local evaluator.

Runs `evaluator.local_evaluator.evaluate` (an unmodified copy of the
official kit's evaluator) against constructed sessions over the small
synthetic catalog, covering all four scenario types. This is the
"zero unhandled exceptions" check from build-brief section 6 — it does not
require the real 50,000-item organizer catalog.

A second test (skipped unless `data/catalog.jsonl` is present) runs the
real 200-session public dev set end to end, which is what
`scripts/run_eval.py` / `python -m evaluator.local_evaluator` does for a
full local reproduction — see README.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src.agent import Agent

SAMPLE_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_catalog.jsonl"
REAL_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.jsonl"
PUBLIC_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "public_set.jsonl"

BASE_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "Prior purchases emphasize fit and comfort.",
}


def _build_sessions(sample_catalog_ids: list[str]) -> list[dict]:
    scenario_cycle = ["buying", "browsing", "intent_override", "boundary"]
    sessions = []
    for i, parent_asin in enumerate(sample_catalog_ids[:12]):
        sessions.append({
            "sample_id": f"synthetic_{i:04d}",
            "scenario_type": scenario_cycle[i % len(scenario_cycle)],
            "user_profile": BASE_PROFILE,
            "ground_truth": {"parent_asin": parent_asin},
        })
    return sessions


def test_full_pipeline_runs_without_exceptions_on_synthetic_sessions() -> None:
    catalog_ids, categories, products = catalog_index(SAMPLE_CATALOG_PATH)
    sessions = _build_sessions(sorted(catalog_ids))
    agent = Agent(str(SAMPLE_CATALOG_PATH))

    result = evaluate(agent, sessions, catalog_ids, categories, products)

    assert result["sample_count"] == len(sessions)
    for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score"):
        assert key in result
    assert 0.0 <= result["hit_rate_at_10"] <= 1.0
    assert set(result["scenario_metrics"].keys()) <= {"buying", "browsing", "intent_override", "boundary"}


def test_full_pipeline_finds_the_target_at_least_sometimes() -> None:
    """Sanity check: an agent that never converges would suggest a real bug.

    Not a strict accuracy bar (the synthetic catalog + intent cards are not
    representative of the real dev set) — just guards against a silent
    "always returns nothing relevant" regression.
    """
    catalog_ids, categories, products = catalog_index(SAMPLE_CATALOG_PATH)
    sessions = _build_sessions(sorted(catalog_ids))
    agent = Agent(str(SAMPLE_CATALOG_PATH))

    result = evaluate(agent, sessions, catalog_ids, categories, products)
    hits = sum(1 for s in result["sessions"] if s["hit"])
    assert hits >= 1, "expected the agent to converge on at least one synthetic session"


@pytest.mark.skipif(not REAL_CATALOG_PATH.exists(), reason="real catalog.jsonl not downloaded (see data/README.md)")
def test_real_public_set_end_to_end() -> None:
    catalog_ids, categories, products = catalog_index(REAL_CATALOG_PATH)
    samples = load_jsonl(PUBLIC_SET_PATH)
    agent = Agent(str(REAL_CATALOG_PATH))

    result = evaluate(agent, samples, catalog_ids, categories, products)

    assert result["sample_count"] == len(samples) == 200
    assert 0.0 <= result["hit_rate_at_10"] <= 1.0
