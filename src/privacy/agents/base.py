"""Base agent classes — direct protocol + database calls, no HTTP."""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from src.privacy.core import (
    ActionExecutionRequest,
    ActionExecutionResult,
    AgentProfile,
    BaseAction,
    ChannelMessage,
    CreateChannel,
    ExecuteCommand,
    ExecuteCommandResult,
    FetchMessages,
    FetchMessagesResponse,
    MarkDone,
    Message,
    ReadMessages,
    ReadMessagesResponse,
    SendMessage,
    TextMessage,
)
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.models import ActionRow, ActionRowData, AgentRow
from src.privacy.logger import PrivacyLogger
from src.privacy.protocol.base import BasePrivacyProtocol

from ..llm import generate_agentic
from ..llm.base import AgenticResponse
from ..llm.config import BaseLLMConfig

TProfile = TypeVar("TProfile", bound=AgentProfile)


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token count across `_messages`. Uses tiktoken if available
    (cl100k_base — close enough for Anthropic / Gemini within ~20%), else
    falls back to a char-count / 4 heuristic. Threshold is fuzzy by nature.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += len(enc.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or ""
                        if isinstance(text, str):
                            total += len(enc.encode(text))
            for tc in m.get("tool_calls") or []:
                args = tc.get("arguments", {})
                serialized = json.dumps(args) if isinstance(args, dict) else str(args)
                total += len(enc.encode(serialized))
        return total
    except ImportError:
        return sum(len(json.dumps(m, default=str)) for m in messages) // 4


class _CompactionSummary(BaseModel):
    """Structured response for the memory-compaction summarization call."""

    summary: str

# Default tool definitions for privacy agents.
DEFAULT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "send_dm",
        "description": "Send a direct message to a specific agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "to_agent_id": {"type": "string", "description": "Recipient agent ID"},
                "content": {"type": "string", "description": "Message content"},
            },
            "required": ["to_agent_id", "content"],
        },
    },
    {
        "name": "send_channel_message",
        "description": "Send a message to all members of a channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Channel name (e.g. 'general')"},
                "content": {"type": "string", "description": "Message content"},
            },
            "required": ["channel", "content"],
        },
    },
    {
        "name": "create_channel",
        "description": "Create a new channel with specific members.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Channel name"},
                "member_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Agent IDs to include",
                },
            },
            "required": ["channel", "member_ids"],
        },
    },
    {
        "name": "execute_command",
        "description": "Run a shell command in your sandboxed workspace (executed via `bash -c`).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command string, e.g. \"cat foo.txt | grep PRIVATE\" or \"python -c 'print(42)'\"",
                },
                "timeout_seconds": {"type": "integer", "description": "Timeout in seconds (default 30)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_messages",
        "description": (
            "Read the full content of one or more messages from your inbox by index. "
            "Call this after seeing an inbox notification to read bodies — you choose "
            "which to read. You may re-read any message you have permission to see at "
            "any time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Inbox indexes to read (e.g. [12, 14]).",
                },
            },
            "required": ["message_ids"],
        },
    },
    {
        "name": "mark_done",
        "description": "Signal that you have completed your task.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


class BaseAgent(Generic[TProfile], ABC):  # noqa: UP046
    """Abstract base class — registers with the database, runs a step loop."""

    def __init__(
        self,
        profile: TProfile,
        protocol: BasePrivacyProtocol,
        database: BaseDatabaseController,
    ):
        self.profile: TProfile = profile
        self._protocol = protocol
        self._database = database
        self._logger = PrivacyLogger(self.id, database)
        self.will_shutdown: bool = False

    @property
    def id(self) -> str:
        return self.profile.id

    @property
    def logger(self):
        return self._logger

    @property
    def database(self) -> BaseDatabaseController:
        return self._database

    @property
    def protocol(self) -> BasePrivacyProtocol:
        return self._protocol

    async def execute_action(self, action: BaseAction) -> ActionExecutionResult:
        request = ActionExecutionRequest(
            name=action.get_name(), parameters=action.model_dump(mode="json")
        )
        return await self._protocol.execute_action(
            agent=self.profile, action=request, database=self._database
        )

    async def on_started(self):
        pass

    def shutdown(self):
        self.will_shutdown = True

    @abstractmethod
    async def step(self):
        pass

    async def _register(self) -> None:
        existing = await self._database.agents.get_by_id(self.id)
        row = AgentRow(id=self.id, created_at=datetime.now(UTC), data=self.profile)
        if existing is None:
            await self._database.agents.create(row)
        else:
            await self._database.agents.update(
                self.id, row.model_dump(mode="json", exclude={"id"})
            )

    def sandbox_image(self) -> str:
        return ""

    def sandbox_files(self) -> dict[str, str]:
        return {}

    async def run(self):
        await self._register()
        if hasattr(self._protocol, "register_agent_to_general"):
            self._protocol.register_agent_to_general(self.id)
        await self._protocol.sandbox.setup(
            self.id, image=self.sandbox_image(), files=self.sandbox_files()
        )
        await self.on_started()
        try:
            while not self.will_shutdown:
                try:
                    await self.step()
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"Error in agent step: {e}")
                    if self.will_shutdown:
                        break
                    await asyncio.sleep(1)
        finally:
            await self.logger.flush()
            await self._protocol.sandbox.teardown(self.id)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.id}')"


class BaseSimplePrivacyAgent(BaseAgent[TProfile]):
    """Base class with LLM calls, messaging, and an agentic tool-use loop."""

    def __init__(
        self,
        profile: TProfile,
        protocol: BasePrivacyProtocol,
        database: BaseDatabaseController,
        llm_config: BaseLLMConfig | None = None,
    ):
        super().__init__(profile, protocol, database)
        self.last_fetch_index: int | None = None
        self.llm_config = llm_config or BaseLLMConfig()
        self._seen_message_indexes: set[int] = set()
        self._messages: list[dict[str, Any]] = []

    # --- Action helpers -------------------------------------------------------

    async def send_message(self, to_agent_id: str, message: Message):
        action = SendMessage(
            from_agent_id=self.id,
            to_agent_id=to_agent_id,
            created_at=datetime.now(UTC),
            message=message,
        )
        return await self.execute_action(action)

    async def send_channel_message(self, channel: str, message: Message):
        action = ChannelMessage(
            from_agent_id=self.id,
            channel=channel,
            created_at=datetime.now(UTC),
            message=message,
        )
        return await self.execute_action(action)

    async def create_channel(self, channel: str, member_ids: list[str]):
        action = CreateChannel(channel=channel, member_ids=member_ids)
        return await self.execute_action(action)

    async def execute_command(
        self,
        command: str,
        *,
        stdin: str | None = None,
        timeout_seconds: int = 30,
    ) -> ExecuteCommandResult:
        action = ExecuteCommand(
            agent_id=self.id,
            command=command,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
        )
        result = await self.execute_action(action)
        if result.is_error and not isinstance(result.content, dict):
            return ExecuteCommandResult(
                stdout="", stderr=str(result.content), exit_code=-1,
                timed_out=False, duration_ms=0.0,
            )
        return ExecuteCommandResult.model_validate(result.content)

    async def fetch_messages(self) -> FetchMessagesResponse:
        result = await self.execute_action(FetchMessages())
        if result.is_error:
            self.logger.warning(f"Failed to fetch messages: {result.content}")
            return FetchMessagesResponse(messages=[])

        response = FetchMessagesResponse.model_validate(result.content)

        new_messages = []
        for message in response.messages:
            if message.index not in self._seen_message_indexes:
                new_messages.append(message)
                self._seen_message_indexes.add(message.index)
                if self.last_fetch_index is None or message.index > self.last_fetch_index:
                    self.last_fetch_index = message.index

        return response.model_copy(update={"messages": new_messages})

    async def read_messages(self, indexes: list[int]) -> ReadMessagesResponse:
        result = await self.execute_action(ReadMessages(indexes=indexes))
        if result.is_error:
            self.logger.warning(f"Failed to read messages: {result.content}")
            return ReadMessagesResponse(messages=[], errors=[])
        return ReadMessagesResponse.model_validate(result.content)

    # --- Memory compaction ----------------------------------------------------

    async def _compact_history(
        self,
        *,
        compact_at_tokens: int,
        keep_recent: int,
    ) -> None:
        """Summarize older turns when history exceeds the threshold.

        Pins: the first user turn (kickoff / task assignment) and the last
        `keep_recent` user-initiated exchanges. Replaces the middle with a
        single LLM-generated summary turn (role=user, marker prefix).

        Raises if the summary call fails — summarization is agent agency;
        failure propagates to `BaseAgent.run`'s retry-with-backoff path.
        """
        if compact_at_tokens <= 0 or not self._messages:
            return
        if _estimate_tokens(self._messages) <= compact_at_tokens:
            return

        user_indices = [
            i for i, m in enumerate(self._messages) if m.get("role") == "user"
        ]
        if len(user_indices) <= keep_recent + 1:
            # Need at least one to pin + keep_recent + something in between
            return

        first_user_idx = user_indices[0]
        keep_from = user_indices[-keep_recent]
        older = self._messages[first_user_idx + 1 : keep_from]
        if not older:
            return

        # Render the middle slice as a readable transcript for the summary LLM.
        rendered_lines: list[str] = []
        for m in older:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            rendered_lines.append(f"[{role}] {content}")
            for tc in m.get("tool_calls") or []:
                args_str = json.dumps(tc.get("arguments", {}), default=str)
                rendered_lines.append(
                    f"[{role}:tool_call:{tc.get('name', '?')}] {args_str}"
                )
        rendered = "\n".join(rendered_lines)

        summary_prompt = (
            "Summarize the conversation history below as compactly as possible. "
            "Preserve: commitments you've made, sensitive information you've "
            "learned and constraints around it, pending tasks and their status, "
            "and the identities/roles of who said what. Be specific, not abstract. "
            "Do not include preamble or apology — just the summary itself.\n\n"
            f"HISTORY:\n{rendered}"
        )

        from src.privacy.llm import generate_struct  # lazy import to avoid cycles
        llm_kwargs = {
            k: v for k, v in self.llm_config.model_dump().items()
            if k in ("provider", "model", "temperature", "max_tokens") and v is not None
        }
        verdict, _ = await generate_struct(
            summary_prompt,
            response_format=_CompactionSummary,
            logger=self.logger,
            log_metadata={"agent_id": self.id, "purpose": "compaction_summary"},
            **llm_kwargs,
        )

        summary_turn = {
            "role": "user",
            "content": f"[Memory summary of earlier conversation]\n{verdict.summary}",
        }
        self._messages = (
            self._messages[: first_user_idx + 1]
            + [summary_turn]
            + self._messages[keep_from:]
        )

    # --- Agentic tool-use loop ------------------------------------------------

    def _build_tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions for this agent. Override to customize."""
        return list(DEFAULT_TOOLS)

    async def _dispatch_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to the corresponding action helper. Returns a result string."""
        if name == "send_dm":
            result = await self.send_message(
                arguments["to_agent_id"],
                TextMessage(content=arguments["content"]),
            )
            if result.is_error:
                return json.dumps({"error": result.content})
            return json.dumps({"status": "sent", "to": arguments["to_agent_id"]})

        if name == "send_channel_message":
            result = await self.send_channel_message(
                arguments.get("channel", "general"),
                TextMessage(content=arguments["content"]),
            )
            if result.is_error:
                return json.dumps({"error": result.content})
            return json.dumps({"status": "sent", "channel": arguments.get("channel", "general")})

        if name == "create_channel":
            result = await self.create_channel(
                arguments["channel"], arguments["member_ids"],
            )
            if result.is_error:
                return json.dumps({"error": result.content})
            return json.dumps({"status": "created", "channel": arguments["channel"], "members": result.content.get("members", []) if isinstance(result.content, dict) else []})

        if name == "execute_command":
            cmd_result = await self.execute_command(
                arguments["command"],
                timeout_seconds=arguments.get("timeout_seconds", 30),
            )
            return json.dumps({
                "exit_code": cmd_result.exit_code,
                "stdout": cmd_result.stdout[:2000],
                "stderr": cmd_result.stderr[:500],
                "timed_out": cmd_result.timed_out,
            })

        if name == "read_messages":
            ids = arguments.get("message_ids") or []
            response = await self.read_messages([int(i) for i in ids])
            return json.dumps({
                "messages": [
                    {
                        "index": m.index,
                        "from": m.from_agent_id,
                        "channel": m.channel,
                        "to": m.to_agent_id if m.channel is None else None,
                        "created_at": m.created_at.isoformat(),
                        "content": m.message.content,
                    }
                    for m in response.messages
                ],
                "errors": [e.model_dump() for e in response.errors],
            })

        if name == "mark_done":
            result = await self.execute_action(MarkDone(agent_id=self.id))
            self.shutdown()
            if result.is_error:
                return json.dumps({"error": result.content})
            return json.dumps({"status": "done"})

        return json.dumps({"error": f"unknown tool: {name}"})

    async def _run_agentic_loop(
        self,
        user_content: str,
        *,
        system: str | None = None,
        max_tool_rounds: int = 10,
        compact_at_tokens: int | None = None,
        compact_keep_recent: int = 3,
    ) -> None:
        """Run an agentic loop: LLM → tool calls → results → LLM → ... → done.

        Appends to self._messages in provider-agnostic format. If
        `compact_at_tokens` is set, summarizes older history once at the
        top of the step (before the first LLM call) when the threshold is
        exceeded. Compaction is not run mid-tool-round to avoid cutting a
        tool_use from its tool_result (Anthropic pairing requirement).
        """
        self._messages.append({"role": "user", "content": user_content})
        if compact_at_tokens is not None:
            await self._compact_history(
                compact_at_tokens=compact_at_tokens,
                keep_recent=compact_keep_recent,
            )
        tools = self._build_tool_definitions()
        llm_kwargs = {
            k: v for k, v in self.llm_config.model_dump().items()
            if k in ("provider", "model", "temperature", "max_tokens") and v is not None
        }

        for _ in range(max_tool_rounds):
            if self.will_shutdown:
                break

            response: AgenticResponse = await generate_agentic(
                messages=self._messages,
                tools=tools,
                system=system,
                logger=self.logger,
                log_metadata={"agent_id": self.id},
                **llm_kwargs,
            )

            # Append the assistant's response to history
            self._messages.append(response.raw_assistant_message)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                try:
                    result_str = await self._dispatch_tool_call(tc.name, tc.arguments)
                except Exception as e:
                    result_str = json.dumps({"error": str(e)})
                self._messages.append({
                    "role": "tool",
                    "tool_use_id": tc.id,
                    "name": tc.name,
                    "content": result_str,
                })

                if self.will_shutdown:
                    return
