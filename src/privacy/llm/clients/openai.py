"""OpenAI model client implementation."""

import json
import threading
from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Literal

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.shared_params import FunctionDefinition

from ..base import (
    AgenticResponse,
    AllowedChatCompletionMessageParams,
    ProviderClient,
    ToolCallInfo,
    TResponseModel,
    Usage,
)
from ..config import BaseLLMConfig, EnvField


class OpenAIConfig(BaseLLMConfig):
    """Configuration for OpenAI provider."""

    provider: Literal["openai"] = EnvField("LLM_PROVIDER", default="openai")  # pyright: ignore[reportIncompatibleVariableOverride]
    api_key: str = EnvField("OPENAI_API_KEY", exclude=True)
    base_url: str | None = EnvField("OPENAI_BASE_URL", default=None)


class OpenAIClient(ProviderClient[OpenAIConfig]):
    """OpenAI model client that accepts OpenAI SDK arguments."""

    _client_cache: dict[str, "OpenAIClient"] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: OpenAIConfig | None = None):
        """Initialize OpenAI client.

        Args:
            config: OpenAI configuration. If None, creates from environment.

        """
        if config is None:
            config = OpenAIConfig()
        else:
            config = OpenAIConfig.model_validate(config)

        super().__init__(config)

        self.config = config
        if not self.config.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )
        self.client = AsyncOpenAI(
            api_key=self.config.api_key, base_url=self.config.base_url
        )

    @staticmethod
    def _get_cache_key(config: OpenAIConfig) -> str:
        """Generate cache key for a config."""
        config_json = config.model_dump_json(include={"api_key", "provider", "base_url"})
        return sha256(config_json.encode()).hexdigest()

    @staticmethod
    def from_cache(config: OpenAIConfig) -> "OpenAIClient":
        """Get or create client from cache."""
        cache_key = OpenAIClient._get_cache_key(config)
        with OpenAIClient._cache_lock:
            if cache_key not in OpenAIClient._client_cache:
                OpenAIClient._client_cache[cache_key] = OpenAIClient(config)
            return OpenAIClient._client_cache[cache_key]

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
        """Generate a validated Pydantic response using OpenAI's native parse()."""
        args: dict[str, Any] = {"model": model, "messages": list(messages)}

        is_reasoning_model = any(
            m in model for m in ("gpt-5", "o4", "o3", "o1")
        )
        if is_reasoning_model:
            if max_tokens:
                args["max_completion_tokens"] = max_tokens
            if "gpt-5-chat" not in model and temperature and temperature < 1.0:
                temperature = None
            if reasoning_effort is not None and "o1" not in model:
                if reasoning_effort == "minimal":
                    reasoning_effort = "low"
                args["reasoning_effort"] = reasoning_effort
        else:
            if temperature is not None:
                args["temperature"] = temperature
            if max_tokens is not None:
                args["max_tokens"] = max_tokens
        args.update(kwargs)

        exceptions: list[Exception] = []
        for _ in range(3):
            try:
                response = await self.client.chat.completions.parse(
                    response_format=response_format, **args
                )
                parsed = response.choices[0].message.parsed
                if parsed is not None:
                    usage = Usage(
                        token_count=response.usage.total_tokens if response.usage else 0,
                        provider="openai",
                        model=model,
                    )
                    return parsed, usage
                elif response.choices[0].message.refusal:
                    raise ValueError(response.choices[0].message.refusal)
                else:
                    break
            except Exception as e:
                exceptions.append(e)
                args["messages"].append({"role": "user", "content": str(e)})

        msg = "Exceeded attempts to parse response_format."
        if exceptions:
            msg += " Inner exceptions: " + " ".join(map(str, exceptions))
        raise RuntimeError(msg)

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
        """Generate with tool-use support via OpenAI API."""
        openai_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            role = msg["role"]
            if role == "tool":
                openai_messages.append(ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=msg["tool_use_id"],
                    content=msg.get("content", ""),
                ))
            elif role == "assistant" and msg.get("tool_calls"):
                openai_messages.append(ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=msg.get("text") or msg.get("content") or "",
                    tool_calls=[
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ],
                ))
            else:
                openai_messages.append({"role": role, "content": msg.get("content", "")})

        openai_tools = [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t["parameters"],
                ),
            )
            for t in tools
        ]

        response = await self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=openai_tools,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            stream=False,
            **kwargs,
        )

        usage = Usage(
            token_count=response.usage.total_tokens if response.usage else 0,
            provider="openai",
            model=model,
        )

        message = response.choices[0].message if response.choices else None
        text = message.content if message else None
        tool_calls: list[ToolCallInfo] = []

        if message and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCallInfo(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
                ))

        raw_assistant: dict[str, Any] = {
            "role": "assistant",
            "content": text or "",
            "text": text,
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        }

        return AgenticResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason or "" if response.choices else "",
            usage=usage,
            raw_assistant_message=raw_assistant,
        )
