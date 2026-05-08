"""Top-level LLM entry points.

Two public async functions:
- `generate_struct` — validated Pydantic output (for LLM-as-judge, analysis, etc.)
- `generate_agentic` — tool-use loop (for agents)
"""

from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import Field, TypeAdapter

from src.privacy.logger import PrivacyLogger

from .base import (
    AgenticResponse,
    AllowedChatCompletionMessageParams,
    TResponseModel,
    Usage,
)
from .clients.anthropic import AnthropicClient, AnthropicConfig
from .clients.gemini import GeminiClient, GeminiConfig
from .clients.openai import OpenAIClient, OpenAIConfig
from .config import EXCLUDE_FIELDS, LLM_PROVIDER, BaseLLMConfig

ConcreteLLMConfigs = Annotated[
    AnthropicConfig | GeminiConfig | OpenAIConfig,
    Field(discriminator="provider"),
]
ConcreteConfigAdapter: TypeAdapter[ConcreteLLMConfigs] = TypeAdapter(ConcreteLLMConfigs)


def _build_client(
    provider: LLM_PROVIDER | None,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    reasoning_effort: str | int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve config + construct cached client. Returns (client, config_kwargs)."""
    # Route through BaseLLMConfig so EnvField defaults fire for any field the
    # caller omitted. The discriminated-union TypeAdapter picks the variant
    # from the discriminator value but does NOT run variant defaults during
    # discrimination, so we must hand it a dict that already has `provider`.
    overrides = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    base = BaseLLMConfig(**overrides)
    config = ConcreteConfigAdapter.validate_python(base.model_dump())

    match config.provider:
        case "anthropic":
            client = AnthropicClient.from_cache(config)
        case "openai":
            client = OpenAIClient.from_cache(config)
        case "gemini":
            client = GeminiClient.from_cache(config)
        case _:
            raise ValueError(f"Unsupported provider: {config.provider}")

    return client, config.model_dump(exclude=EXCLUDE_FIELDS)


async def generate_struct(
    messages: Sequence[AllowedChatCompletionMessageParams] | str,
    *,
    response_format: type[TResponseModel],
    provider: LLM_PROVIDER | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | int | None = None,
    logger: PrivacyLogger | None = None,
    log_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[TResponseModel, Usage]:
    """Generate a validated Pydantic response from the LLM.

    Use this for LLM-as-judge, analysis scoring, classification — any case where
    you want a typed, validated result rather than free-form text or tool calls.
    """
    client, config_kwargs = _build_client(provider, model, temperature, max_tokens, reasoning_effort)
    return await client.generate_struct(
        messages=messages,
        response_format=response_format,
        logger=logger,
        log_metadata=log_metadata,
        **{**config_kwargs, **kwargs},
    )


async def generate_agentic(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    provider: LLM_PROVIDER | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system: str | None = None,
    logger: PrivacyLogger | None = None,
    log_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> AgenticResponse:
    """Generate a response with tool-use support. Used by the agent framework."""
    client, config_kwargs = _build_client(provider, model, temperature, max_tokens)
    return await client.generate_agentic(
        messages=messages,
        tools=tools,
        system=system,
        logger=logger,
        log_metadata=log_metadata,
        **{**config_kwargs, **kwargs},
    )


def clear_client_caches() -> None:
    """Clear the client caches in all client classes."""
    AnthropicClient._client_cache.clear()
    OpenAIClient._client_cache.clear()
    GeminiClient._client_cache.clear()
