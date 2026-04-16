"""Privacy framework protocol — DM, channels, sandbox execution."""

from datetime import UTC, datetime

from src.privacy.core import (
    ActionAdapter,
    ActionExecutionRequest,
    ActionExecutionResult,
    AgentProfile,
    ChannelMessage,
    CreateChannel,
    ExecuteCommand,
    FetchMessages,
    SendMessage,
)
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.models import ActionRow, ActionRowData
from src.privacy.protocol.base import BasePrivacyProtocol
from src.privacy.sandbox import Sandbox, build_sandbox

from .execute_command import execute_execute_command
from .messaging import (
    execute_channel_message,
    execute_create_channel,
    execute_fetch_messages,
    execute_send_message,
)


class PrivacyProtocol(BasePrivacyProtocol):
    """Protocol with DM, channel messaging, and sandbox execution.

    No provenance dependency — provenance analysis is done post-hoc by reading
    the action log in the database.
    """

    def __init__(self, sandbox: Sandbox | None = None):
        self._sandbox = sandbox or build_sandbox()
        self._channels: dict[str, set[str]] = {}

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    @property
    def channels(self) -> dict[str, set[str]]:
        return self._channels

    def register_agent_to_general(self, agent_id: str) -> None:
        self._channels.setdefault("general", set()).add(agent_id)

    def get_actions(self):
        return [SendMessage, ChannelMessage, CreateChannel, FetchMessages, ExecuteCommand]

    async def initialize(self, database: BaseDatabaseController) -> None:
        """No-op for the in-memory backend."""

    async def execute_action(
        self,
        *,
        agent: AgentProfile,
        action: ActionExecutionRequest,
        database: BaseDatabaseController,
    ) -> ActionExecutionResult:
        """Dispatch an action and persist the action row."""
        parsed_action = ActionAdapter.validate_python(action.parameters)
        result = await self._dispatch(parsed_action, agent, database)

        await database.actions.create(
            ActionRow(
                id="",
                created_at=datetime.now(UTC),
                data=ActionRowData(agent_id=agent.id, request=action, result=result),
            )
        )
        return result

    async def _dispatch(
        self,
        parsed_action,
        agent: AgentProfile,
        database: BaseDatabaseController,
    ) -> ActionExecutionResult:
        if isinstance(parsed_action, SendMessage):
            return await execute_send_message(parsed_action, database)

        if isinstance(parsed_action, ChannelMessage):
            return await execute_channel_message(parsed_action, self._channels)

        if isinstance(parsed_action, CreateChannel):
            return await execute_create_channel(parsed_action, self._channels)

        if isinstance(parsed_action, FetchMessages):
            return await execute_fetch_messages(parsed_action, agent, database, self._channels)

        if isinstance(parsed_action, ExecuteCommand):
            return await execute_execute_command(
                parsed_action, agent, database, sandbox=self._sandbox,
            )

        raise ValueError(f"Unknown action type: {parsed_action.type}")
