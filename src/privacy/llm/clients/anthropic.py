"""Anthropic model client implementation."""

import threading
from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Literal, cast

import anthropic
import anthropic.types

from ..base import (
    AgenticResponse,
    AllowedChatCompletionMessageParams,
    ProviderClient,
    ToolCallInfo,
    TResponseModel,
    Usage,
)
from ..config import BaseLLMConfig, EnvField


class AnthropicConfig(BaseLLMConfig):
    """Configuration for Anthropic provider."""

    provider: Literal["anthropic"] = EnvField("LLM_PROVIDER", default="anthropic")  # pyright: ignore[reportIncompatibleVariableOverride]
    api_key: str = EnvField("ANTHROPIC_API_KEY", exclude=True)


class AnthropicClient(ProviderClient[AnthropicConfig]):
    """Anthropic model client that accepts OpenAI SDK arguments."""

    _client_cache: dict[str, "AnthropicClient"] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: AnthropicConfig | None = None):
        """Initialize Anthropic client.

        Args:
            config: Anthropic configuration. If None, creates from environment.

        """
        if config is None:
            config = AnthropicConfig()
        else:
            config = AnthropicConfig.model_validate(config)

        super().__init__(config)

        self.config = config
        if not self.config.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key in config."
            )
        self.client = anthropic.AsyncAnthropic(api_key=self.config.api_key)

    @staticmethod
    def _get_cache_key(config: AnthropicConfig) -> str:
        """Generate cache key for a config.

        Includes `base_url` so a config that switches endpoints (e.g. Bedrock
        proxy) doesn't silently share a cached client with the default endpoint.
        Pydantic's `include` skips fields that don't exist on the model, so this
        is a no-op until `AnthropicConfig` gains a `base_url` field.
        """
        config_json = config.model_dump_json(include={"api_key", "provider", "base_url"})
        return sha256(config_json.encode()).hexdigest()

    @staticmethod
    def from_cache(config: AnthropicConfig) -> "AnthropicClient":
        """Get or create client from cache."""
        cache_key = AnthropicClient._get_cache_key(config)
        with AnthropicClient._cache_lock:
            if cache_key not in AnthropicClient._client_cache:
                AnthropicClient._client_cache[cache_key] = AnthropicClient(config)
            return AnthropicClient._client_cache[cache_key]

    async def _generate_struct(
        self,
        *,
        model: str,
        messages: Sequence[AllowedChatCompletionMessageParams],
        response_format: type[TResponseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | int | None = None,
        **kwargs: Any,
    ) -> tuple[TResponseModel, Usage]:
        """Generate a validated Pydantic response using Anthropic tool-calling."""
        anthropic_messages, system_prompt = self._convert_messages(messages)

        args: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 2000,
        }
        if system_prompt:
            args["system"] = system_prompt

        if reasoning_effort is not None:
            # Any string value (including "minimal", "low", "high", etc.) maps
            # to "no thinking budget" — Anthropic's thinking takes an int budget
            # in tokens, not an effort label.
            if isinstance(reasoning_effort, str):
                reasoning_effort = 0
            if reasoning_effort == 0:
                args["thinking"] = anthropic.types.ThinkingConfigDisabledParam(type="disabled")
            else:
                temperature = None
                args["thinking"] = anthropic.types.ThinkingConfigEnabledParam(
                    type="enabled", budget_tokens=reasoning_effort,
                )

        if temperature is not None:
            args["temperature"] = temperature
        args.update(kwargs)

        return await self._generate_structured(args, response_format, model)

    def _convert_messages(
        self, messages: Sequence[AllowedChatCompletionMessageParams]
    ) -> tuple[list[anthropic.types.MessageParam], str | None]:
        """Convert OpenAI messages to Anthropic format.

        Returns:
            A tuple of (messages, system_prompt) where messages is the list of
            MessageParam objects and system_prompt is the concatenated system messages
            or None if there are no system messages.

        """
        anthropic_messages: list[anthropic.types.MessageParam] = []
        system_messages: list[str] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if not content:
                continue

            # Collect system messages separately
            if role == "system":
                if isinstance(content, str):
                    system_messages.append(content)
                elif isinstance(content, list):
                    # Handle multi-part content (text only for now)
                    text_parts: list[str] = []
                    for part in content:
                        if part.get("type", None) == "text":
                            text = part.get("text", "")
                            text_parts.append(text)
                    if text_parts:
                        system_messages.append("\n".join(text_parts))
            elif role in ("user", "assistant"):
                # Handle different content formats
                if isinstance(content, str):
                    anthropic_messages.append(
                        anthropic.types.MessageParam(role=role, content=content)
                    )
                elif isinstance(content, list):
                    # Handle multi-part content (text only for now)
                    text_content: list[str] = []
                    for part in content:
                        if part.get("type", None) == "text":
                            text = part.get("text", "")
                            text_content.append(text)

                    if text_content:
                        anthropic_messages.append(
                            anthropic.types.MessageParam(
                                role=role,
                                content="\n".join(text_content),
                            )
                        )

        # Concatenate system messages or return None
        system_prompt = "\n\n".join(system_messages) if system_messages else None

        return anthropic_messages, system_prompt

    async def _generate_structured(
        self,
        base_args: dict[str, Any],
        response_format: type[TResponseModel],
        model: str,
    ) -> tuple[TResponseModel, Usage]:
        """Generate structured output using tool calling."""
        # Create tool for the response format
        tool = anthropic.types.ToolParam(
            name=f"generate{response_format.__name__}",
            description=f"Generate a {response_format.__name__} object.",
            input_schema=response_format.model_json_schema(),
        )

        tool_choice = anthropic.types.ToolChoiceToolParam(
            name=tool["name"], type="tool", disable_parallel_tool_use=True
        )

        # Update args for tool use
        args = base_args.copy()
        args["tools"] = [tool]
        args["tool_choice"] = tool_choice

        # Thinking not allowed when forcing tool use
        if "thinking" in args:
            args["thinking"] = anthropic.types.ThinkingConfigDisabledParam(
                type="disabled"
            )

        # Track exception messages for final error
        exceptions: list[str] = []

        # Make up to 3 attempts
        for attempt in range(3):
            try:
                response = await self.client.messages.create(**args, stream=False)

                # Find tool use block
                for block in response.content:
                    if block.type == "tool_use":
                        usage = Usage(
                            token_count=response.usage.input_tokens
                            + response.usage.output_tokens
                            if response.usage
                            else 0,
                            provider="anthropic",
                            model=model,
                        )

                        try:
                            validated_response = response_format.model_validate(
                                block.input
                            )
                            return validated_response, usage
                        except Exception as e:
                            import traceback

                            tb_str = traceback.format_exc()
                            error_message = f"Error parsing tool response: {e}\nTraceback:\n{tb_str}"
                            exceptions.append(error_message)

                            # Add error to conversation for retry
                            if attempt < 2:
                                current_messages = cast(
                                    list[anthropic.types.MessageParam], args["messages"]
                                )
                                current_messages.append(
                                    anthropic.types.MessageParam(
                                        role="assistant",
                                        content=response.content,
                                    )
                                )
                                current_messages.append(
                                    anthropic.types.MessageParam(
                                        role="user",
                                        content=error_message,
                                    )
                                )
                            break

                # No tool use found, add to conversation for retry
                else:
                    error_message = (
                        f"No tool use found in response on attempt {attempt + 1}"
                    )
                    exceptions.append(error_message)

                    if attempt < 2:
                        # Get current messages and add assistant response
                        current_messages = cast(
                            list[anthropic.types.MessageParam], args["messages"]
                        )
                        current_messages.append(
                            anthropic.types.MessageParam(
                                role="assistant",
                                content=response.content,
                            )
                        )
                        current_messages.append(
                            anthropic.types.MessageParam(
                                role="user",
                                content="Please use the required tool to format your response.",
                            )
                        )

            except Exception as e:
                import traceback

                tb_str = traceback.format_exc()
                error_message = f"API call failed on attempt {attempt + 1}: {e}\nTraceback:\n{tb_str}"
                exceptions.append(error_message)

                if attempt < 2:
                    # Add error to conversation for retry
                    current_messages = cast(
                        list[anthropic.types.MessageParam], args["messages"]
                    )
                    current_messages.append(
                        anthropic.types.MessageParam(
                            role="user",
                            content=error_message,
                        )
                    )
                else:
                    break

        raise Exception(
            "Failed to _generate_structured: Exceeded maximum retries. Inner exceptions: "
            + " -> ".join(exceptions)
        )

    async def _generate_agentic(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AgenticResponse:
        """Generate with tool-use support via Anthropic API."""
        # Anthropic agentic doesn't yet honor reasoning_effort (would need
        # `thinking` config plumbed in here, same as _generate_struct does);
        # strip it so it doesn't reach the SDK as an unknown kwarg.
        kwargs.pop("reasoning_effort", None)
        anthropic_messages: list[anthropic.types.MessageParam] = []
        system_prompt: str | None = None

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                anthropic_messages.append(
                    anthropic.types.MessageParam(role="user", content=content)
                )
            elif role == "assistant":
                # Reconstruct content blocks (text + tool_use)
                blocks: list[Any] = []
                if msg.get("text"):
                    blocks.append({"type": "text", "text": msg["text"]})
                for tc in msg.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                if not blocks and isinstance(content, str) and content:
                    blocks = content
                anthropic_messages.append(
                    anthropic.types.MessageParam(role="assistant", content=blocks or content)
                )
            elif role == "tool":
                # Tool results are user messages with tool_result blocks in Anthropic
                anthropic_messages.append(
                    anthropic.types.MessageParam(
                        role="user",
                        content=[{
                            "type": "tool_result",
                            "tool_use_id": msg["tool_use_id"],
                            "content": msg.get("content", ""),
                        }],
                    )
                )

        anthropic_tools = [
            anthropic.types.ToolParam(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t["parameters"],
            )
            for t in tools
        ]

        args: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "tools": anthropic_tools,
            "max_tokens": max_tokens or 4096,
        }
        if system_prompt:
            args["system"] = system_prompt
        if temperature is not None:
            args["temperature"] = temperature
        args.update(kwargs)

        response = await self.client.messages.create(**args, stream=False)

        usage = Usage(
            token_count=(response.usage.input_tokens + response.usage.output_tokens)
            if response.usage else 0,
            provider="anthropic",
            model=model,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCallInfo] = []
        raw_blocks: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                raw_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCallInfo(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))
                raw_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return AgenticResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "",
            usage=usage,
            raw_assistant_message={
                "role": "assistant",
                "content": raw_blocks,
                "text": "\n".join(text_parts) if text_parts else None,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            },
        )
