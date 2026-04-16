"""Handlers for SendMessage / FetchMessages — the messaging feature.

SendMessage validates the receiver exists and lets the protocol persist the
action row (fetch_messages then reads it back). FetchMessages scans the action
table for SendMessage rows addressed to the caller.
"""

from src.privacy.core import (
    ActionExecutionResult,
    AgentProfile,
    FetchMessages,
    FetchMessagesResponse,
    MessageAdapter,
    ReceivedMessage,
    SendMessage,
)
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.models import ActionRow


_SEND_MESSAGE = SendMessage.get_name()


# --- SendMessage --------------------------------------------------------------


async def execute_send_message(
    action: SendMessage,
    database: BaseDatabaseController,
) -> ActionExecutionResult:
    """Validate the target agent exists; the action row itself is persisted by the protocol."""
    if await database.agents.get_by_id(action.to_agent_id) is None:
        return ActionExecutionResult(
            content={"error": f"to_agent_id {action.to_agent_id} not found"},
            is_error=True,
        )

    return ActionExecutionResult(
        content=action.model_dump(mode="json"),
        is_error=False,
        metadata={"status": "sent"},
    )


# --- FetchMessages ------------------------------------------------------------


async def execute_fetch_messages(
    action: FetchMessages,
    agent: AgentProfile,
    database: BaseDatabaseController,
) -> ActionExecutionResult:
    """Return all SendMessage actions addressed to `agent` (optionally from a specific sender)."""
    sender_filter = action.from_agent_id

    def predicate(row: ActionRow) -> bool:
        if row.data.request.name != _SEND_MESSAGE:
            return False
        params = row.data.request.parameters
        if params.get("to_agent_id") != agent.id:
            return False
        if sender_filter is not None and params.get("from_agent_id") != sender_filter:
            return False
        return True

    rows = await database.actions.find(predicate)
    messages = [_to_received_message(row) for row in rows]

    response = FetchMessagesResponse(messages=messages)
    return ActionExecutionResult(content=response.model_dump(mode="json"))


def _to_received_message(action_row: ActionRow) -> ReceivedMessage:
    params = action_row.data.request.parameters
    if action_row.index is None:
        raise ValueError("action_row.index must be populated by the database backend")
    return ReceivedMessage(
        from_agent_id=params["from_agent_id"],
        to_agent_id=params["to_agent_id"],
        created_at=action_row.created_at,
        message=MessageAdapter.validate_python(params.get("message", {})),
        index=action_row.index,
    )
