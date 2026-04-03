# src/my_llm_router/registry.py

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .exceptions import ProviderNotFoundError

# ── Provider function imports ──────────────────────────────────
from .providers.bedrock.claude_haiku_45 import claude_haiku_45_function
# from .providers.bedrock.claude_sonnet_35 import claude_sonnet_35_function
# from .providers.openai.gpt4o import gpt4o_function
# from .providers.gemini.gemini_pro import gemini_pro_function

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# REGISTRY MAP
# llm_name (what user passes) → function (what handles the call)
# ─────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable[..., Any]] = {
    "claude_haiku_4.5"  : claude_haiku_45_function
    # "claude_sonnet_3.5" : claude_sonnet_35_function,
    # "gpt4o"             : gpt4o_function,
    # "gemini_pro"        : gemini_pro_function,
}


# ─────────────────────────────────────────────────────────────
# GET HANDLER — called by Router
# ─────────────────────────────────────────────────────────────

def get_handler(llm_name: str) -> Callable[..., Any]:
    """
    Match user's llm_name to a registered function.

    Raises
    ------
    ProviderNotFoundError
        If llm_name has no match in _REGISTRY.
    """
    name = llm_name.strip()

    if name not in _REGISTRY:
        available = list_providers()
        raise ProviderNotFoundError(
            f"We don't have any LLM registered with the name {name!r}. "
            f"Available: {available}"
        )

    return _REGISTRY[name]


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def list_providers() -> list[str]:
    """Return all registered llm_names."""
    return sorted(_REGISTRY.keys())


def is_registered(llm_name: str) -> bool:
    """Check if a name exists in the registry."""
    return llm_name.strip() in _REGISTRY