# src/orbitaim_llmkit/models.py

from __future__ import annotations

from typing import Type, Any, Literal
from pydantic import BaseModel, model_validator, field_validator


# ─────────────────────────────────────────────────────────────
# VALID AWS REGIONS
# ─────────────────────────────────────────────────────────────

AWSRegion = Literal[
    # United States
    "us-east-1",       # N. Virginia
    "us-east-2",       # Ohio
    "us-west-1",       # N. California
    "us-west-2",       # Oregon
    # Asia Pacific
    "ap-south-1",      # Mumbai
    "ap-northeast-1",  # Tokyo
    "ap-northeast-2",  # Seoul
    "ap-northeast-3",  # Osaka
    "ap-southeast-1",  # Singapore
    "ap-southeast-2",  # Sydney
    # Canada
    "ca-central-1",    # Central
    # Europe
    "eu-central-1",    # Frankfurt
    "eu-west-1",       # Ireland
    "eu-west-2",       # London
    "eu-west-3",       # Paris
    "eu-north-1",      # Stockholm
    # South America
    "sa-east-1",       # São Paulo
]


# ─────────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────────

class LLMRequest(BaseModel):
    llm_name: str
    region_name: AWSRegion          
    system_prompt: str
    user_prompt: str
    context: str | None = None
    temperature: float
    pydantic_model: Type[BaseModel] | None = None
    max_tokens: int
    aws_access_key_id: str
    aws_secret_access_key: str
    repo_name: str
    llm_identifier: str

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("llm_name", "system_prompt", "user_prompt",
                     "aws_access_key_id", "aws_secret_access_key",
                     "repo_name", "llm_identifier")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name!r} must be a non-empty string")
        return v.strip()

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"temperature must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {v}")
        if v > 128_000:
            raise ValueError(f"max_tokens exceeds upper limit of 128,000, got {v}")
        return v

    @field_validator("context")
    @classmethod
    def validate_context(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None
        return v

    @model_validator(mode="after")
    def validate_pydantic_model_is_basemodel(self) -> "LLMRequest":
        if self.pydantic_model is not None:
            if not (isinstance(self.pydantic_model, type) and
                    issubclass(self.pydantic_model, BaseModel)):
                raise ValueError(
                    f"pydantic_model must be a BaseModel class, "
                    f"got {type(self.pydantic_model)}"
                )
        return self