#!/usr/bin/env python3
"""Interactive, real-time terminal chat with the Agent.

There is no UI for this project by design (see docs/competition_specification.md
— "mandatory UI work" is explicitly out of scope). This script is the
practical stand-in: it drives the exact same `Agent.reset`/`Agent.respond`
interface the official evaluator uses, one turn at a time, so you can type
messages and watch the router/state/retrieval/reranker/orchestrator react
live in your terminal.

Usage:
    python3 scripts/chat.py                          # uses the small synthetic catalog
    python3 scripts/chat.py --catalog data/catalog.jsonl   # once you've downloaded the real one

Type your message and press Enter each turn. Type `quit` (or Ctrl-D) to end
early. The session ends automatically after turn 10, exactly like the real
evaluator enforces.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import Agent  # noqa: E402
from src.orchestrator import MAX_TURNS  # noqa: E402

DEFAULT_CATALOG = str(Path(__file__).resolve().parent.parent / "data" / "sample_catalog.jsonl")


def _print_response(agent: Agent, response: dict) -> None:
    print(f"\nagent: {response['message']}")
    if response.get("ask_attribute"):
        print(f"       (asking about: {response['ask_attribute']})")
    recs = response.get("recommendations") or []
    if recs:
        print(f"       top {len(recs)} recommendation(s):")
        for i, rec in enumerate(recs, start=1):
            pid = rec["parent_asin"]
            product = agent.catalog.by_id.get(pid)
            title = product.title if product else "(unknown product)"
            price = f"${product.price:.2f}" if product and product.price is not None else "?"
            print(f"         {i:>2}. {pid}  {title}  [{price}]")
    else:
        print("       (no recommendations this turn)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, help="Path to catalog.jsonl (default: synthetic sample catalog)")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading catalog from {args.catalog} ...")
    agent = Agent(args.catalog)
    print(f"Loaded {len(agent.catalog)} products.\n")

    session_id = f"chat_{uuid.uuid4().hex[:8]}"
    user_profile = {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 4.5,
        "rating_style": "usually positive",
        "preference_tags": [],
        "summary": "",
    }
    agent.reset(session_id, user_profile)

    print(f"Session {session_id} started. Up to {MAX_TURNS} turns. Type 'quit' to stop early.\n")

    for turn in range(1, MAX_TURNS + 1):
        try:
            user_message = input(f"[turn {turn}/{MAX_TURNS}] you: ")
        except EOFError:
            print()
            break
        if user_message.strip().lower() in ("quit", "exit"):
            break

        response = agent.respond(session_id, user_message, turn, args.top_k)
        _print_response(agent, response)
        print()

    print("Session ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
