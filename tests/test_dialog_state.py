"""Unit tests for the dialog state tracker (section C / section 6)."""
from __future__ import annotations

from src.dialog.state_tracker import CONFIDENT_THRESHOLD, DialogState


def test_extracts_material_color_and_price() -> None:
    state = DialogState()
    state.merge_turn("I want black leather boots under $80.", 1)
    assert state.slots["color"].value == "black"
    assert state.slots["material"].value == "leather"
    assert state.price_max == 80.0


def test_price_range_extraction() -> None:
    state = DialogState()
    state.merge_turn("Something between $30 and $60 please.", 1)
    assert state.price_min == 30.0
    assert state.price_max == 60.0


def test_reinforcement_raises_confidence_and_confirmed_count() -> None:
    state = DialogState()
    state.merge_turn("I like black shoes.", 1)
    first_confidence = state.slots["color"].confidence
    state.merge_turn("Yes, black is the color I want.", 2)
    assert state.slots["color"].confidence >= first_confidence
    assert state.slots["color"].confirmed_count == 2


def test_override_language_replaces_slot_value() -> None:
    state = DialogState()
    state.merge_turn("I want black running shoes.", 1)
    assert state.slots["color"].value == "black"
    state.merge_turn("Actually, ignore my earlier preference. Make them white casual sneakers.", 2)
    assert state.slots["color"].value == "white"
    assert state.slots["style"].value == "casual"
    assert len(state.override_events) >= 1


def test_override_resets_confidence_for_replaced_slot() -> None:
    state = DialogState()
    state.merge_turn("I want black shoes.", 1)
    state.merge_turn("Black shoes, definitely black.", 2)  # reinforce to raise confidence
    high_confidence = state.slots["color"].confidence
    state.merge_turn("Actually, never mind, I want white shoes instead.", 3)
    assert state.slots["color"].value == "white"
    assert state.slots["color"].confidence < high_confidence


def test_slot_decay_when_not_reinforced() -> None:
    state = DialogState()
    state.merge_turn("I want black shoes.", 1)
    initial_confidence = state.slots["color"].confidence
    # Several turns pass mentioning something unrelated.
    for turn in range(2, 6):
        state.merge_turn("Tell me about materials.", turn)
    assert state.slots["color"].confidence < initial_confidence
    assert state.slots["color"].turns_since_reinforced >= 3


def test_decayed_slot_drops_out_of_confident_set() -> None:
    state = DialogState()
    state.merge_turn("black", 1)
    for turn in range(2, 10):
        state.merge_turn("nothing relevant here", turn)
    assert state.slots["color"].confidence < CONFIDENT_THRESHOLD
    assert "color" not in state.confident_slots()


def test_budget_reinforcement_vs_new_value() -> None:
    state = DialogState()
    state.merge_turn("Under $80 please.", 1)
    state.merge_turn("Yes, under $80 works.", 2)
    assert state.budget_slot.confirmed_count == 2
    state.merge_turn("Actually make it under $50.", 3)
    assert state.price_max == 50.0
