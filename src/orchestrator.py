"""Turn / session orchestrator — build-brief section H.

The single entry point tying router → retrieval/fusion → (clarify or
rerank) → response together. Responsibilities that live here and nowhere
else:

  * owns the turn counter and the hard 10-turn cap in code (never "hoped"),
  * never lets the clarification generator fire with <=1 turn of margin
    left before the cap — forces a best-effort ranked answer instead,
  * is the central error boundary: every stage runs inside one try/except
    per turn, and any failure degrades to a plain BM25 response rather
    than propagating and zeroing out the session,
  * guarantees session-state isolation: `reset()` always installs a brand
    new `SessionRuntime`, never mutates a shared/global object.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.catalog.loader import Catalog
from src.context.session_profile import SessionProfile
from src.dialog.clarification import build_clarification
from src.dialog.state_tracker import CONFIDENT_THRESHOLD, DialogState
from src.ranking.llm_client import ModelClient, get_client
from src.ranking.llm_reranker import get_reranker
from src.retrieval.attribute_filter import INDEXED_SLOTS, AttributeFilter, SlotQuery
from src.retrieval.bm25_index import BM25Index
from src.retrieval.fusion import fuse
from src.retrieval.vector_index import VectorIndex
from src.router import intent_router
from src.schemas import (
    Recommendation, TurnResponse, UserProfile,
    safe_parse_user_profile, safe_top_k, safe_turn, safe_user_message,
)

logger = logging.getLogger("techjam.orchestrator")

MAX_TURNS = 10
FORCE_ANSWER_TURN = 9  # on turns 9-10, never start a *new* clarification


class SessionRuntime:
    """Everything scoped to one session_id. Always constructed fresh."""

    __slots__ = ("state", "profile")

    def __init__(self, session_id: str, user_profile: UserProfile):
        self.state = DialogState()
        self.profile = SessionProfile(session_id, user_profile)


class Orchestrator:
    def __init__(self, catalog: Catalog, vector_index: Optional[VectorIndex] = None):
        self.catalog = catalog
        self.bm25_index = BM25Index(catalog)
        self.vector_index = vector_index if vector_index is not None else VectorIndex(catalog)
        self.attribute_filter = AttributeFilter(catalog)
        self.llm_client: Optional[ModelClient] = get_client()
        self.reranker = get_reranker(self.llm_client)
        # Session state lives only here, keyed by session_id, and is only
        # ever replaced wholesale in reset() — never mutated across sessions.
        self._sessions: dict[str, SessionRuntime] = {}

    # -- public contract-facing API -----------------------------------
    def reset(self, session_id: str, user_profile_raw: Any) -> None:
        user_profile = safe_parse_user_profile(user_profile_raw)
        self._sessions[str(session_id)] = SessionRuntime(str(session_id), user_profile)

    def respond(self, session_id: str, user_message_raw: Any, turn_raw: Any, top_k_raw: Any) -> dict:
        turn = safe_turn(turn_raw)
        top_k = safe_top_k(top_k_raw)
        user_message = safe_user_message(user_message_raw)
        session_id = str(session_id)
        try:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                # Contract guarantees reset() precedes respond(), but never
                # raise over a missing/out-of-order reset — degrade instead.
                logger.warning("respond() called before reset() for session %s — initializing blank state.", session_id)
                runtime = SessionRuntime(session_id, UserProfile())
                self._sessions[session_id] = runtime
            response = self._respond_inner(runtime, user_message, turn, top_k)
            return response.as_contract_dict()
        except Exception:  # noqa: BLE001 - central error boundary, see module docstring
            logger.exception(
                "Pipeline failure for session=%s turn=%s — degrading to plain BM25 fallback.",
                session_id, turn,
            )
            return self._safe_fallback(user_message, top_k)

    # -- pipeline -------------------------------------------------------
    def _respond_inner(self, runtime: SessionRuntime, user_message: str, turn: int, top_k: int) -> TurnResponse:
        state = runtime.state
        profile = runtime.profile

        state.merge_turn(user_message, turn)

        unresolved_previous_ask = None
        if state.last_ask_attribute is not None and not self._slot_confident(state, state.last_ask_attribute):
            unresolved_previous_ask = state.last_ask_attribute

        route = intent_router.classify(user_message, state, llm_client=self.llm_client)
        adjustments = profile.observe(turn, route.track, state)
        effective_track = adjustments.track_override or route.track

        slot_queries = [
            SlotQuery(name=name, value=slot.value, confidence=slot.confidence)
            for name, slot in state.slots.items()
            if name in INDEXED_SLOTS
        ]

        fusion_result = fuse(
            self.catalog, self.bm25_index, self.vector_index, self.attribute_filter,
            query_text=state.query_text(),
            slot_queries=slot_queries,
            price_min=state.price_min,
            price_max=state.price_max,
            track=effective_track,
            n_confident_slots=state.n_confident_slots(),
            widen_pool=adjustments.widen_pool,
        )

        near_cap = turn >= FORCE_ANSWER_TURN
        no_signal = not state.has_any_hard_constraint() and len(user_message.split()) <= 3
        should_try_clarify = (fusion_result.over_generality or no_signal) and not near_cap

        prompt_tokens, completion_tokens = route.prompt_tokens, route.completion_tokens
        candidate_ids = [pid for pid, _ in fusion_result.ranked]

        if should_try_clarify:
            clar = build_clarification(candidate_ids, self.catalog, state, avoid_slot=unresolved_previous_ask)
            if clar is not None:
                ask_attribute, message = clar
                ranked, usage = self.reranker.rerank(
                    fusion_result.ranked, self.catalog, state, profile.summary,
                    top_k=min(top_k, fusion_result.dynamic_top_k),
                    boosted_slots=adjustments.boosted_slots,
                )
                self._update_ask_tracking(state, ask_attribute)
                return TurnResponse(
                    message=message,
                    ask_attribute=ask_attribute,
                    recommendations=[Recommendation(parent_asin=rc.parent_asin) for rc in ranked],
                    usage={
                        "prompt_tokens": prompt_tokens + usage["prompt_tokens"],
                        "completion_tokens": completion_tokens + usage["completion_tokens"],
                    },
                )
            # No good clarification available (e.g. only the just-asked,
            # still-unresolved slot remains) — fall through to a best-effort answer.

        ranked, usage = self.reranker.rerank(
            fusion_result.ranked, self.catalog, state, profile.summary,
            top_k=top_k, boosted_slots=adjustments.boosted_slots,
        )
        state.last_ask_attribute = None
        state.consecutive_same_ask = 0
        return TurnResponse(
            message=self._answer_message(state, len(ranked)),
            ask_attribute=None,
            recommendations=[Recommendation(parent_asin=rc.parent_asin) for rc in ranked],
            usage={
                "prompt_tokens": prompt_tokens + usage["prompt_tokens"],
                "completion_tokens": completion_tokens + usage["completion_tokens"],
            },
        )

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _slot_confident(state: DialogState, name: str) -> bool:
        if name == "budget":
            return state.budget_slot is not None and state.budget_slot.confidence >= CONFIDENT_THRESHOLD
        slot = state.slots.get(name)
        return slot is not None and slot.confidence >= CONFIDENT_THRESHOLD

    @staticmethod
    def _update_ask_tracking(state: DialogState, ask_attribute: str) -> None:
        if state.last_ask_attribute == ask_attribute:
            state.consecutive_same_ask += 1
        else:
            state.last_ask_attribute = ask_attribute
            state.consecutive_same_ask = 1

    @staticmethod
    def _answer_message(state: DialogState, n_results: int) -> str:
        if n_results == 0:
            return "I couldn't find a strong match yet — could you tell me a bit more about what you're looking for?"
        confident = state.confident_slots()
        if confident:
            described = ", ".join(f"{name} {slot.value}" for name, slot in list(confident.items())[:3])
            return f"Here are the closest matches for {described}."
        return "Here are the closest matches I found so far."

    def _safe_fallback(self, user_message: str, top_k: int) -> dict:
        """Last-resort path: touches only the BM25 index, nothing else."""
        try:
            hits = self.bm25_index.search(user_message, top_n=top_k)
            recommendations = [{"parent_asin": pid} for pid, _ in hits[:top_k]]
        except Exception:  # noqa: BLE001 - even the fallback must never raise
            logger.exception("Safe fallback itself failed — returning an empty recommendation list.")
            recommendations = []
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
