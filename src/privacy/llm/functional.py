"""Abstract base class for LLM model clients."""

from collections.abc import Sequence
from typing import Annotated, Any, overload

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
from .config import EXCLUDE_FIELDS, LLM_PROVIDER

ConcreteLLMConfigs = Annotated[
    AnthropicConfig | GeminiConfig | OpenAIConfig,
    Field(discriminator="provider"),
]
ConcreteConfigAdapter: TypeAdapter[ConcreteLLMConfigs] = TypeAdapter(ConcreteLLMConfigs)


@overload
async def generate(
    messages: Sequence[AllowedChatCompletionMessageParams],
    *,
    provider: LLM_PROVIDER | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | int | None = None,
    response_format: None = None,
    logger: PrivacyLogger | None = None,
    log_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[str, Usage]: ...
@overload
async def generate(
    messages: str,
    *,
    provider: LLM_PROVIDER | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | int | None = None,
    response_format: None = None,
    logger: PrivacyLogger | None = None,
    log_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[str, Usage]: ...


@overload
async def generate(
    messages: Sequence[AllowedChatCompletionMessageParams],
    *,
    provider: LLM_PROVIDER | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | int | None = None,
    response_format: type[TResponseModel],
    logger: PrivacyLogger | None = None,
    log_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[TResponseModel, Usage]: ...
@overload
async def generate(
    messages: str,
    *,
    provider: LLM_PROVIDER | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | int | None = None,
    response_format: type[TResponseModel],
    logger: PrivacyLogger | None = None,
    log_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[TResponseModel, Usage]: ...


async def generate(
    messages: Sequence[AllowedChatCompletionMessageParams] | str,
    *,
    provider: LLM_PROVIDER | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | int | None = None,
    response_format: type[TResponseModel] | None = None,
    logger: PrivacyLogger | None = None,
    log_metadata: dict[str, Any] | None = None,
    max_concurrency: int | None = None,
    **kwargs: Any,
) -> tuple[str, Usage] | tuple[TResponseModel, Usage]:
    """Generate a completion using OpenAI SDK arguments.

    Args:
        messages: List of chat messages in OpenAI format
        provider: The LLM provider.
        model: The model to use (or default if not provided)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        reasoning_effort: Reasoning effort level for capable models
        response_format: Optional structured output schema
        logger: Optional PrivacyLogger for logging LLM calls
        log_metadata: Optional metadata to include with LLM logs
        max_concurrency: Optional, the maximum number of concurrent requests the returned client supports.
        **kwargs: Additional provider-specific arguments

    Returns:
        String response when response_format is None, otherwise structured BaseModel

    """
    config_kwargs = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
    config = ConcreteConfigAdapter.validate_python(config_kwargs)

    # Get or create client from cache using the from_cache method
    match config.provider:
        case "anthropic":
            client = AnthropicClient.from_cache(config)
        case "openai":
            client = OpenAIClient.from_cache(config)
        case "gemini":
            client = GeminiClient.from_cache(config)
        case _:
            raise ValueError(f"Unsupported provider: {config.provider}")

    kwargs = {**config.model_dump(exclude=EXCLUDE_FIELDS), **kwargs}

    return await client.generate(
        messages=messages,
        response_format=response_format,
        logger=logger,
        log_metadata=log_metadata,
        **kwargs,
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
    """Generate a completion with tool-use support."""
    config_kwargs = {"provider": provider, "model": model, "temperature": temperature, "max_tokens": max_tokens}
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
    config = ConcreteConfigAdapter.validate_python(config_kwargs)

    match config.provider:
        case "anthropic":
            client = AnthropicClient.from_cache(config)
        case "openai":
            client = OpenAIClient.from_cache(config)
        case "gemini":
            client = GeminiClient.from_cache(config)
        case _:
            raise ValueError(f"Unsupported provider: {config.provider}")

    return await client.generate_agentic(
        messages=messages,
        tools=tools,
        system=system,
        logger=logger,
        log_metadata=log_metadata,
        **{**config.model_dump(exclude=EXCLUDE_FIELDS), **kwargs},
    )


def clear_client_caches() -> None:
    """Clear the client caches in all client classes. Useful for testing or changing configurations."""
    AnthropicClient._client_cache.clear()
    OpenAIClient._client_cache.clear()
    GeminiClient._client_cache.clear()
