# src/orbitaim_llmkit/router.py

from __future__ import annotations

import logging
from typing import TypeVar, Type

from pydantic import BaseModel

from .models import LLMRequest
from .registry import get_handler
from .types import LLMResponse
from .exceptions import (
    RouterValidationError,
    ProviderNotFoundError,
    LLMCallError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Router:

    def __init__(self, usage_repo=None):
        self._usage_repo = usage_repo

    async def get_response(
        self,
        llm_name: str,
        region_name: str,
        system_prompt: str,
        user_prompt: str,
        context: str | None,
        temperature: float,
        pydantic_model: Type[T] | None,
        max_tokens: int,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        repo_name: str,
        llm_identifier: str,
    ) -> LLMResponse | T:

        # ── 1. Validate inputs ────────────────────────────────
        try:
            request = LLMRequest(
                llm_name=llm_name,
                region_name=region_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context=context,
                temperature=temperature,
                pydantic_model=pydantic_model,
                max_tokens=max_tokens,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                repo_name=repo_name,
                llm_identifier=llm_identifier,
            )
        except Exception as exc:
            raise RouterValidationError(f"Invalid input: {exc}") from exc

        # ── 2. Find the relevant function from registry ───────
        handler = get_handler(request.llm_name)

        # ── 3. Transfer user input to that function ───────────
        try:
            response: LLMResponse = await handler(request)
        except ProviderNotFoundError:
            raise
        except Exception as exc:
            logger.error(
                "LLM call failed | llm=%s | identifier=%s | error=%s",
                request.llm_name, request.llm_identifier, exc,
            )
            raise LLMCallError(
                f"'{request.llm_name}' failed for '{request.llm_identifier}': {exc}"
            ) from exc

        # ── 4. Return the LLM output to the user ────────────
        return response