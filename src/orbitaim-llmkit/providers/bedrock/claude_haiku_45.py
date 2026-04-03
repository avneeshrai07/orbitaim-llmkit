# src/orbitaim_llmkit/providers/bedrock/claude_haiku_45.py

from __future__ import annotations

import logging
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

from ...registry import register
from ...models import LLMRequest
from ...types import LLMResponse

logger = logging.getLogger(__name__)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@register("claude_haiku_4.5")
async def claude_haiku_45_function(request: LLMRequest) -> LLMResponse:

    # ── 1. Build client from request creds ───────────────────
    client = ChatBedrockConverse(
        model_id=MODEL_ID,
        region_name=request.region_name,
        aws_access_key_id=request.aws_access_key_id,
        aws_secret_access_key=request.aws_secret_access_key,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # ── 2. Build messages ─────────────────────────────────────
    if request.context:
        user_message = f"{request.user_prompt}\n\nContext:\n{request.context}"
    else:
        user_message = request.user_prompt

    messages = [
        SystemMessage(content=request.system_prompt),
        HumanMessage(content=user_message),
    ]

    # ── 3a. Structured output — pydantic_model provided ───────
    if request.pydantic_model is not None:
        structured_client = client.with_structured_output(
            request.pydantic_model,
            include_raw=True,
        )
        response = await structured_client.ainvoke(messages)

        raw     = response["raw"]
        parsed  = response["parsed"]
        usage   = raw.usage_metadata

        logger.info(
            "claude_haiku_4.5 [structured] | identifier=%s | input=%d | output=%d",
            request.llm_identifier,
            usage["input_tokens"],
            usage["output_tokens"],
        )

        return LLMResponse(
            text=parsed.model_dump_json(),   # serialise parsed model → JSON string
            provider="bedrock",
            model=MODEL_ID,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            llm_identifier=request.llm_identifier,
        )

    # ── 3b. Normal output — no pydantic_model ─────────────────
    response = await client.ainvoke(messages)
    usage    = response.usage_metadata

    logger.info(
        "claude_haiku_4.5 [text] | identifier=%s | input=%d | output=%d",
        request.llm_identifier,
        usage["input_tokens"],
        usage["output_tokens"],
    )

    return LLMResponse(
        text=response.content,
        provider="bedrock",
        model=MODEL_ID,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        llm_identifier=request.llm_identifier,
    )