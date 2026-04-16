"""Gemini model client implementation."""

import json
import threading
from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Literal

import google.genai as genai
import google.genai.types
import pydantic

from ..base import (
    AgenticResponse,
    AllowedChatCompletionMessageParams,
    ProviderClient,
    ToolCallInfo,
    TResponseModel,
    Usage,
)
from ..config import BaseLLMConfig, EnvField


class GeminiConfig(BaseLLMConfig):
    """Configuration for Gemini provider."""

    provider: Literal["gemini"] = EnvField("LLM_PROVIDER", default="gemini")  # pyright: ignore[reportIncompatibleVariableOverride]
    api_key: str = EnvField("GEMINI_API_KEY", exclude=True)


class GeminiClient(ProviderClient[GeminiConfig]):
    """Gemini model client that accepts OpenAI SDK arguments."""

    _client_cache: dict[str, "GeminiClient"] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: GeminiConfig | None = None):
        """Initialize Gemini client.

        Args:
            config: Gemini configuration. If None, creates from environment.

        """
        if config is None:
            config = GeminiConfig()
        else:
            config = GeminiConfig.model_validate(config)

        super().__init__(config)

        self.config = config
        if not self.config.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key in config."
            )
        self.client = genai.Client(api_key=self.config.api_key)

    @staticmethod
    def _get_cache_key(config: GeminiConfig) -> str:
        """Generate cache key for a config."""
        config_json = config.model_dump_json(include={"api_key", "provider"})
        return sha256(config_json.encode()).hexdigest()

    @staticmethod
    def from_cache(config: GeminiConfig) -> "GeminiClient":
        """Get or create client from cache."""
        cache_key = GeminiClient._get_cache_key(config)
        with GeminiClient._cache_lock:
            if cache_key not in GeminiClient._client_cache:
                GeminiClient._client_cache[cache_key] = GeminiClient(config)
            return GeminiClient._client_cache[cache_key]

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
        """Generate structured output using Gemini API with retry logic."""
        # Convert messages to Gemini format
        contents, system_prompt = self._convert_messages(messages)

        # Track exception messages for final error
        exceptions: list[str] = []

        # Build config
        config = google.genai.types.GenerateContentConfig()

        if temperature is not None:
            config.temperature = temperature

        if max_tokens is not None:
            config.max_output_tokens = max_tokens

        # Handle reasoning effort -> thinking config
        if reasoning_effort is not None:
            if reasoning_effort == "minimal":
                reasoning_effort = 0
            elif isinstance(reasoning_effort, str):
                reasoning_effort = 0  # Fallback for unsupported string values

            if isinstance(reasoning_effort, int) and reasoning_effort >= -1:  # type: ignore[misc]
                config.thinking_config = google.genai.types.ThinkingConfig(
                    thinking_budget=reasoning_effort
                )

        # Configure for structured output
        config.response_schema = response_format.model_json_schema()
        config.response_mime_type = "application/json"

        # Add system instruction if we have system messages
        if system_prompt:
            config.system_instruction = system_prompt

        args: dict[str, Any] = {
            "model": model,
            "contents": contents,
            "config": config,
        }

        # Make 3 attempts while recovering from errors
        for attempt in range(3):
            try:
                # Add any additional kwargs
                args.update(kwargs)

                response = await self.client.aio.models.generate_content(**args)

                token_count = 0
                if (
                    response.usage_metadata
                    and response.usage_metadata.total_token_count
                ):
                    token_count = response.usage_metadata.total_token_count

                usage = Usage(
                    token_count=token_count,
                    provider="gemini",
                    model=model,
                )

                # Parse structured response
                if response.parsed:
                    return response_format.model_validate(response.parsed), usage
                elif response.text:
                    try:
                        return response_format.model_validate_json(response.text), usage
                    except (json.JSONDecodeError, pydantic.ValidationError) as e:
                        import traceback

                        tb_str = traceback.format_exc()
                        error_message = (
                            f"Error parsing response: {e}\nTraceback:\n{tb_str}"
                        )
                        exceptions.append(error_message)

                        # Add error context to conversation for retry
                        if attempt < 2:
                            contents.append(
                                google.genai.types.Content(
                                    role="user",
                                    parts=[google.genai.types.Part(text=error_message)],
                                )
                            )
                        continue
                else:
                    error_message = f"No response content available from Gemini on attempt {attempt + 1}"
                    exceptions.append(error_message)

                    # Add context for retry
                    if attempt < 2:
                        contents.append(
                            google.genai.types.Content(
                                role="user",
                                parts=[
                                    google.genai.types.Part(
                                        text="No response received. Please provide a JSON response that matches the required schema."
                                    )
                                ],
                            )
                        )
                    continue

            except Exception as e:
                error_message = (
                    f"Gemini API call failed on attempt {attempt + 1}: {str(e)}"
                )
                exceptions.append(error_message)

                # Add error context to conversation for retry
                if attempt < 2:
                    contents.append(
                        google.genai.types.Content(
                            role="user",
                            parts=[
                                google.genai.types.Part(
                                    text=f"Error: {str(e)}. Please try again with a valid JSON response."
                                )
                            ],
                        )
                    )
                continue

        raise Exception(
            "Failed to _generate_struct: Exceeded maximum retries. Inner exceptions: "
            + " -> ".join(exceptions)
        )

    def _convert_messages(
        self, messages: Sequence[AllowedChatCompletionMessageParams]
    ) -> tuple[list[google.genai.types.Content], str | None]:
        """Convert OpenAI messages to Gemini Content format.

        Returns:
            A tuple of (contents, system_prompt) where contents is the list of
            Content objects and system_prompt is the concatenated system messages
            or None if there are no system messages.

        """
        gemini_contents: list[google.genai.types.Content] = []
        system_messages: list[str] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if not content:
                continue

            # Handle different content types
            if isinstance(content, str):
                text_content = content
            elif isinstance(content, list):
                # Handle multi-part content
                text_parts: list[str] = []
                for part in content:
                    if part["type"] == "text":
                        text_parts.append(part.get("text", ""))
                text_content = "\n".join(text_parts)
            else:
                text_content = str(content)

            if not text_content:
                continue

            # Convert roles: OpenAI "assistant" -> Gemini "model", "user" stays "user"
            # Collect system messages separately
            if role == "system":
                system_messages.append(text_content)
            elif role == "assistant":
                gemini_contents.append(
                    google.genai.types.Content(
                        role="model", parts=[google.genai.types.Part(text=text_content)]
                    )
                )
            elif role == "user":
                gemini_contents.append(
                    google.genai.types.Content(
                        role="user", parts=[google.genai.types.Part(text=text_content)]
                    )
                )

        # Concatenate system messages or return None
        system_prompt = "\n\n".join(system_messages) if system_messages else None

        return gemini_contents, system_prompt

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
        """Generate with tool-use support via Gemini API.

        Gemini's function-calling API maps cleanly: tools become
        FunctionDeclarations, tool_use responses become FunctionCall parts,
        and tool results become FunctionResponse parts.
        """
        contents: list[google.genai.types.Content] = []
        system_prompt: str | None = None

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                contents.append(google.genai.types.Content(
                    role="user", parts=[google.genai.types.Part(text=content)],
                ))
            elif role == "assistant":
                parts: list[google.genai.types.Part] = []
                if msg.get("text"):
                    parts.append(google.genai.types.Part(text=msg["text"]))
                for tc in msg.get("tool_calls", []):
                    parts.append(google.genai.types.Part(
                        function_call=google.genai.types.FunctionCall(
                            name=tc["name"], args=tc["arguments"],
                        )
                    ))
                if not parts and content:
                    parts.append(google.genai.types.Part(text=content))
                contents.append(google.genai.types.Content(role="model", parts=parts))
            elif role == "tool":
                contents.append(google.genai.types.Content(
                    role="user",
                    parts=[google.genai.types.Part(
                        function_response=google.genai.types.FunctionResponse(
                            name=msg.get("name", "tool"),
                            response={"result": msg.get("content", "")},
                        )
                    )],
                ))

        gemini_tools = [google.genai.types.Tool(
            function_declarations=[
                google.genai.types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("parameters"),
                )
                for t in tools
            ]
        )]

        config = google.genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 4096,
            tools=gemini_tools,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        client = genai.Client(api_key=self.config.api_key)
        response = await client.aio.models.generate_content(
            model=model, contents=contents, config=config, **kwargs,
        )

        token_count = 0
        if response.usage_metadata:
            token_count = (
                (response.usage_metadata.prompt_token_count or 0)
                + (response.usage_metadata.candidates_token_count or 0)
            )
        usage = Usage(token_count=token_count, provider="gemini", model=model)

        text_parts: list[str] = []
        tool_calls: list[ToolCallInfo] = []

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    tool_calls.append(ToolCallInfo(
                        id=f"gemini-{part.function_call.name}",
                        name=part.function_call.name,
                        arguments=dict(part.function_call.args) if part.function_call.args else {},
                    ))

        return AgenticResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=str(response.candidates[0].finish_reason) if response.candidates else "",
            usage=usage,
            raw_assistant_message={
                "role": "assistant",
                "text": "\n".join(text_parts) if text_parts else None,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            },
        )
