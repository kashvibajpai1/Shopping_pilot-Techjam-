"""Thin, swappable LLM client abstraction.

Every call site (the reranker, the ambiguous-case intent router escalation)
goes through `get_client()` / `ModelClient.complete(...)`, never through a
provider SDK directly — swapping models or providers later touches only
this file, per build-brief section 8.

Fully optional: with no API key / package / `TECHJAM_ENABLE_LLM=1`,
`get_client()` returns `None` and every caller is required to treat that as
"use the offline fallback", not as an error. This is what keeps the
pipeline reliable when the grading sandbox has no outbound network access
(build-brief section 0, critical unknown #2) — nothing here is on the
critical path to producing a response.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger("techjam.ranking.llm_client")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TIMEOUT_SECONDS = 6.0
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="techjam-llm")


@dataclass
class CompletionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class ModelClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 512) -> CompletionResult: ...


class AnthropicClient:
    """Reference implementation using the `anthropic` SDK."""

    def __init__(self, model: str, api_key: str):
        import anthropic  # deferred import: only required if this backend is used

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 512) -> CompletionResult:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0,
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        return CompletionResult(
            text=text,
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


class _TimeoutGuardedClient:
    """Wraps any ModelClient with a hard wall-clock timeout."""

    def __init__(self, inner: ModelClient, timeout_seconds: float):
        self._inner = inner
        self._timeout = timeout_seconds

    def complete(self, system: str, user: str, max_tokens: int = 512) -> CompletionResult:
        future = _EXECUTOR.submit(self._inner.complete, system, user, max_tokens)
        return future.result(timeout=self._timeout)


def get_client() -> Optional[ModelClient]:
    """Returns a configured, timeout-guarded ModelClient, or None.

    Enabled only when TECHJAM_ENABLE_LLM is truthy AND a matching API key
    is present AND the provider SDK imports cleanly. Any failure along the
    way is logged and treated as "no client available" — never raised.
    """
    if os.environ.get("TECHJAM_ENABLE_LLM", "").strip().lower() not in ("1", "true", "yes"):
        return None
    provider = os.environ.get("TECHJAM_LLM_PROVIDER", "anthropic").strip().lower()
    model = os.environ.get("TECHJAM_LLM_MODEL", DEFAULT_MODEL)
    timeout = float(os.environ.get("TECHJAM_LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    try:
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("TECHJAM_ENABLE_LLM set but ANTHROPIC_API_KEY is missing.")
                return None
            client: ModelClient = AnthropicClient(model=model, api_key=api_key)
        else:
            logger.warning("Unknown TECHJAM_LLM_PROVIDER=%r — no LLM client available.", provider)
            return None
    except Exception:  # noqa: BLE001 - any import/config failure disables the LLM path
        logger.warning("Failed to construct LLM client for provider=%r", provider, exc_info=True)
        return None
    return _TimeoutGuardedClient(client, timeout)
