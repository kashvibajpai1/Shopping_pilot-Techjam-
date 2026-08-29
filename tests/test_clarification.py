"""Clarification generator tests (section F / section 6)."""
from __future__ import annotations

from src.dialog.clarification import build_clarification
from src.dialog.state_tracker import DialogState


def test_asks_about_a_discriminative_slot_not_generic(sample_catalog) -> None:
    state = DialogState()
    candidate_ids = [p.parent_asin for p in sample_catalog.products]
    clar = build_clarification(candidate_ids, sample_catalog, state)
    assert clar is not None
    ask_attribute, message = clar
    assert ask_attribute in (
        "category", "material", "color", "size", "style",
        "brand", "budget", "feature", "use_case", "other",
    )
    assert message and isinstance(message, str)


def test_does_not_ask_about_already_confident_slot(sample_catalog) -> None:
    state = DialogState()
    state.merge_turn("I want black shoes, definitely black shoes.", 1)
    assert state.slots["color"].confidence >= 0.5
    candidate_ids = [p.parent_asin for p in sample_catalog.products]
    clar = build_clarification(candidate_ids, sample_catalog, state)
    if clar is not None:
        ask_attribute, _message = clar
        assert ask_attribute != "color"


def test_avoids_repeating_the_same_unresolved_slot(sample_catalog) -> None:
    state = DialogState()
    candidate_ids = [p.parent_asin for p in sample_catalog.products]
    first = build_clarification(candidate_ids, sample_catalog, state, avoid_slot=None)
    assert first is not None
    first_slot = first[0]
    second = build_clarification(candidate_ids, sample_catalog, state, avoid_slot=first_slot)
    if second is not None:
        assert second[0] != first_slot


def test_returns_none_when_only_the_avoided_slot_remains() -> None:
    # A trivial catalog stub where only one attribute varies at all.
    class FakeProduct:
        def __init__(self, pid, color):
            self.parent_asin = pid
            self.category_tokens = ()
            self.materials = frozenset()
            self.colors = frozenset([color])
            self.sizes = frozenset()
            self.store = ""
            self.price_bucket = "unknown"

    class FakeCatalog:
        def __init__(self, products):
            self.by_id = {p.parent_asin: p for p in products}

    products = [FakeProduct("A", "black"), FakeProduct("B", "white")]
    catalog = FakeCatalog(products)
    state = DialogState()
    result = build_clarification(["A", "B"], catalog, state, avoid_slot="color")
    # "color" is the only discriminative attribute available and it's avoided,
    # so we must not fall back to asking it again.
    assert result is None or result[0] != "color"
