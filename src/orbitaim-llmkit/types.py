# src/my_llm_router/types.py

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, field_validator


# ─────────────────────────────────────────────────────────────
# PROVIDER LITERAL
# ─────────────────────────────────────────────────────────────

ProviderName = Literal[
    "openai",
    "anthropic",
    "gemini",
    "bedrock",
]
# Add new providers here as you build them.
# This gives you autocomplete + type safety on llm_name.


# ─────────────────────────────────────────────────────────────
# LLM RESPONSE
# ─────────────────────────────────────────────────────────────

class LLMResponse(BaseModel):
    """
    Unified response object returned by every provider.

    Every provider's handler MUST return this — no matter which
    LLM is called underneath, the Router always gets back the
    same shape.

    Fields
    ------
    text            : The raw text output from the LLM
    provider        : Which LLM handled this call
    model           : Exact model used e.g. "claude-3-5-sonnet-20241022"
    input_tokens    : Tokens consumed by the prompt
    output_tokens   : Tokens consumed by the response
    llm_identifier  : Echo of the request's llm_identifier — for tracing
    latency_ms      : How long the LLM call took (set by Router, not provider)
    """

    text: str
    provider: str                   # not ProviderName — allows custom providers
    model: str
    input_tokens: int
    output_tokens: int
    llm_identifier: str
    latency_ms: int = 0             # default 0 — Router fills this after the call

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("LLM returned an empty response text")
        return v

    @field_validator("input_tokens", "output_tokens", "latency_ms")
    @classmethod
    def must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Token counts and latency must be non-negative")
        return v

    # ── Convenience properties ────────────────────────────────

    @property
    def total_tokens(self) -> int:
        """Total tokens used = input + output."""
        return self.input_tokens + self.output_tokens

    @property
    def is_truncated(self) -> bool:
        """
        Rough heuristic — if output_tokens is suspiciously round
        it likely hit max_tokens. Providers should ideally set
        this explicitly but this is a safe fallback check.
        """
        return self.output_tokens % 100 == 0 and self.output_tokens > 0

    def __repr__(self) -> str:
        return (
            f"LLMResponse("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"tokens={self.total_tokens}, "
            f"latency_ms={self.latency_ms}, "
            f"identifier={self.llm_identifier!r})"
        )