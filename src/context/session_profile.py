"""Session profile / context-programming layer — build-brief section G.

This is what makes "self-evolution" more than static prompt engineering:
`observe()` is called once per turn with the freshly updated dialog state
and actually changes downstream behavior for the *next* turn —

  * up-weights a slot in rerank scoring once it has been reconfirmed twice,
  * flags "widen the candidate pool" once Browsing has run several turns
    without convergence,
  * overrides the router's own Buying/Browsing call once enough concrete
    constraints have accumulated, tightening toward Buying-track retrieval
    even if the current turn's phrasing still reads as exploratory.

Every adjustment is appended to `strategy_log` with a short reason string,
so the log itself is the demonstrable artifact for the demo video / README
(see build-brief section G and the "Definition of done" for Innovation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.dialog.state_tracker import DialogState
from src.schemas import UserProfile

BROWSING_STREAK_THRESHOLD = 3
TIGHTEN_CONFIDENT_SLOT_THRESHOLD = 3
CONFIRM_BOOST_THRESHOLD = 2


@dataclass
class StrategyLogEntry:
    turn: int
    reason: str
    adjustment: str


@dataclass
class TurnAdjustments:
    track_override: str | None = None
    widen_pool: bool = False
    boosted_slots: frozenset[str] = field(default_factory=frozenset)


class SessionProfile:
    def __init__(self, session_id: str, user_profile: UserProfile):
        self.session_id = session_id
        self.summary = user_profile.summary
        self.preference_tags = list(user_profile.preference_tags)
        self.rating_style = user_profile.rating_style
        self.purchase_frequency = user_profile.purchase_frequency
        self.turn_count = 0
        self.browsing_streak = 0
        self.boosted_slots: set[str] = set()
        self.strategy_log: list[StrategyLogEntry] = []

    def record(self, turn: int, reason: str, adjustment: str) -> None:
        self.strategy_log.append(StrategyLogEntry(turn=turn, reason=reason, adjustment=adjustment))

    def observe(self, turn: int, track: str, state: DialogState) -> TurnAdjustments:
        self.turn_count = turn
        self.browsing_streak = self.browsing_streak + 1 if track == "browsing" else 0

        widen_pool = False
        if self.browsing_streak >= BROWSING_STREAK_THRESHOLD:
            widen_pool = True
            self.record(
                turn,
                f"{self.browsing_streak} consecutive Browsing turns without convergence",
                "widen candidate pool / relax the over-generality threshold for this turn",
            )

        track_override = None
        confident = state.n_confident_slots()
        if confident >= TIGHTEN_CONFIDENT_SLOT_THRESHOLD and track != "buying":
            track_override = "buying"
            self.record(
                turn,
                f"{confident} confident constraints accumulated",
                "override router: tighten toward Buying-track retrieval this turn",
            )

        for name, slot in state.slots.items():
            if slot.confirmed_count >= CONFIRM_BOOST_THRESHOLD and name not in self.boosted_slots:
                self.boosted_slots.add(name)
                self.record(
                    turn,
                    f"slot '{name}' reconfirmed {slot.confirmed_count}x",
                    f"up-weight '{name}' in rerank scoring going forward",
                )
        if (
            state.budget_slot is not None
            and state.budget_slot.confirmed_count >= CONFIRM_BOOST_THRESHOLD
            and "budget" not in self.boosted_slots
        ):
            self.boosted_slots.add("budget")
            self.record(
                turn,
                f"budget reconfirmed {state.budget_slot.confirmed_count}x",
                "up-weight price fit in rerank scoring going forward",
            )

        return TurnAdjustments(
            track_override=track_override,
            widen_pool=widen_pool,
            boosted_slots=frozenset(self.boosted_slots),
        )

    def strategy_log_as_dicts(self) -> list[dict]:
        return [
            {"turn": e.turn, "reason": e.reason, "adjustment": e.adjustment}
            for e in self.strategy_log
        ]
