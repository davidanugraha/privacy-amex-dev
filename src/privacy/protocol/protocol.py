"""Privacy framework protocol implementation with provenance monitoring."""

import json
from datetime import UTC, datetime

from src.privacy.core import (
    ActionAdapter,
    ActionExecutionRequest,
    ActionExecutionResult,
    AgentProfile,
    ExecuteCommand,
    FetchMessages,
    SendMessage,
)
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.models import ActionRow, ActionRowData
from src.privacy.protocol.base import BasePrivacyProtocol
from src.privacy.provenance import ProvenanceStore
from src.privacy.sandbox import Sandbox, build_sandbox
from .execute_command import execute_execute_command
from .messaging import execute_fetch_messages, execute_send_message


class PrivacyProtocol(BasePrivacyProtocol):
    """Privacy framework protocol — SendMessage and FetchMessages with provenance hooks."""

    def __init__(
        self,
        provenance_store: ProvenanceStore | None = None,
        sandbox: Sandbox | None = None,
    ):
        """Initialize the protocol.

        Args:
            provenance_store: Optional provenance store for recording transfers.
                              If None, provenance recording is disabled.
            sandbox: Optional sandbox for ExecuteCommand. If None, a sandbox is
                     built from the SANDBOX_BACKEND env var (default: local).
        """
        self._provenance = provenance_store
        self._sandbox = sandbox or build_sandbox()

    def get_actions(self):
        """Define available actions."""
        return [SendMessage, FetchMessages, ExecuteCommand]

    async def initialize(self, database: BaseDatabaseController) -> None:
        """Initialize protocol — no-op for the in-memory backend."""

    async def execute_action(
        self,
        *,
        agent: AgentProfile,
        action: ActionExecutionRequest,
        database: BaseDatabaseController,
    ) -> ActionExecutionResult:
        """Dispatch an action, record provenance, and persist the action row.

        In the HTTP-free architecture this method is the single entry point for
        all agent actions — it dispatches to the per-action handler, records
        provenance side-effects, and persists the request/result pair so
        FetchMessages can read prior SendMessages back.
        """
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
            result = await execute_send_message(parsed_action, database)
            if not result.is_error and self._provenance is not None:
                await self._record_transfer(parsed_action, agent, database)
            return result

        if isinstance(parsed_action, FetchMessages):
            return await execute_fetch_messages(parsed_action, agent, database)

        if isinstance(parsed_action, ExecuteCommand):
            image = agent.metadata.get("sandbox_image", "")
            return await execute_execute_command(
                parsed_action,
                agent,
                database,
                sandbox=self._sandbox,
                provenance=self._provenance,
                image=image,
            )

        raise ValueError(f"Unknown action type: {parsed_action.type}")

    async def _record_transfer(
        self,
        action: SendMessage,
        sender_profile: AgentProfile,
        database: BaseDatabaseController,
    ) -> None:
        """Record a message transfer in the provenance store.

        Extracts principal info from agent metadata (stored at registration time).
        """
        if self._provenance is None:
            return

        # Get sender principal from metadata
        sender_principal = sender_profile.metadata.get("principal_name", "unknown")
        allowed_receivers: list[str] = sender_profile.metadata.get("allowed_receivers", [])

        # Get receiver principal from database
        receiver_profile = await database.agents.get_by_id(action.to_agent_id)
        if receiver_profile is None:
            return
        receiver_principal = receiver_profile.data.metadata.get("principal_name", "unknown")

        # Extract message content for hashing/preview
        content = ""
        if hasattr(action.message, "content"):
            content = action.message.content
        else:
            try:
                content = json.dumps(action.message.model_dump())
            except Exception:
                content = str(action.message)

        self._provenance.record(
            sender_agent_id=action.from_agent_id,
            sender_principal=sender_principal,
            allowed_receivers=allowed_receivers,
            receiver_agent_id=action.to_agent_id,
            receiver_principal=receiver_principal,
            content=content,
        )
