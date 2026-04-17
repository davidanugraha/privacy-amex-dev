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
    ActionProtocol,
    AgentProfile,
    BaseAction,
    ChannelMessage,
    CreateChannel,
    ExecuteCommand,
    ExecuteCommandResult,
    FetchMessages,
    FetchMessagesResponse,
    Message,
    SendMessage,
    TextMessage,
)
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.models import AgentRow
from src.privacy.logger import PrivacyLogger
from src.privacy.protocol.base import BasePrivacyProtocol

from ..llm import generate_agentic
from ..llm.base import AgenticResponse
from ..llm.config import BaseLLMConfig

TProfile = TypeVar("TProfile", bound=AgentProfile)

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
        "description": "Run a command in your sandboxed workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argv-style command (e.g. ['python', 'script.py'])",
                },
                "stdin": {"type": "string", "description": "Optional stdin input"},
                "timeout_seconds": {"type": "integer", "description": "Timeout (default 30)"},
            },
            "required": ["command"],
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

    async def get_protocol(self) -> list[ActionProtocol]:
        return [
            a if isinstance(a, ActionProtocol) else a.to_protocol()
            for a in self._protocol.get_actions()
        ]

    async def execute_action(self, action: BaseAction) -> ActionExecutionResult:
        request = ActionExecutionRequest(
            name=action.get_name(), parameters=action.model_dump(mode="json")
        )
        return await self._protocol.execute_action(
            agent=self.profile, action=request, database=self._database
        )

    async def on_started(self):
        pass

    async def on_will_stop(self):
        pass

    async def on_stopped(self):
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
            await self.on_will_stop()
            await self.on_stopped()
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
        command: list[str],
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
                stdin=arguments.get("stdin"),
                timeout_seconds=arguments.get("timeout_seconds", 30),
            )
            return json.dumps({
                "exit_code": cmd_result.exit_code,
                "stdout": cmd_result.stdout[:2000],
                "stderr": cmd_result.stderr[:500],
                "timed_out": cmd_result.timed_out,
            })

        if name == "mark_done":
            self.shutdown()
            return json.dumps({"status": "done"})

        return json.dumps({"error": f"unknown tool: {name}"})

    async def _run_agentic_loop(
        self,
        user_content: str,
        *,
        system: str | None = None,
        max_tool_rounds: int = 10,
    ) -> None:
        """Run an agentic loop: LLM → tool calls → results → LLM → ... → done.

        Appends to self._messages in provider-agnostic format.
        """
        self._messages.append({"role": "user", "content": user_content})
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
