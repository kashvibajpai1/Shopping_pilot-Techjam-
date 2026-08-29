"""Unit tests for the rule-based intent router (section B / section 6)."""
from __future__ import annotations

from src.dialog.state_tracker import DialogState
from src.router.intent_router import classify

# A small hand-labeled set, written by us (per build-brief section 6).
LABELED_TURNS = [
    ("I need black leather boots under $80 in size 9.", "buying"),
    ("Looking for something casual, not sure what exactly.", "browsing"),
    ("I want a red cotton dress, size medium, by TrailBlaze.", "buying"),
    ("Just browsing, any suggestions?", "browsing"),
    ("I'm still exploring, maybe something for a wedding.", "browsing"),
    ("I need a necklace, budget around $50.", "buying"),
]


def test_router_accuracy_on_labeled_set() -> None:
    correct = 0
    for text, expected in LABELED_TURNS:
        decision = classify(text, DialogState())
        if decision.track == expected:
            correct += 1
    accuracy = correct / len(LABELED_TURNS)
    assert accuracy >= 0.8, f"router accuracy {accuracy} below threshold on labeled set"


def test_router_never_raises_on_empty_text() -> None:
    decision = classify("", DialogState())
    assert decision.track in ("buying", "browsing")
    assert 0.0 <= decision.confidence <= 1.0


def test_router_reroutes_mid_conversation_as_slots_accumulate() -> None:
    state = DialogState()
    first = classify("I'm just looking around.", state)
    assert first.track == "browsing"

    state.merge_turn("I'm just looking around.", 1)
    state.merge_turn("Actually I want black leather boots, size 9, under $80.", 2)
    second = classify("Actually I want black leather boots, size 9, under $80.", state)
    assert second.track == "buying"


def test_ambiguous_case_defaults_to_browsing_without_llm_client() -> None:
    # A message that trips neither strong buying nor strong browsing signals.
    decision = classify("Something for the weekend.", DialogState(), llm_client=None)
    assert decision.track == "browsing"
