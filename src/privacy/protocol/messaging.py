"""Handlers for DM, channel messaging, and channel creation.

SendMessage — point-to-point DM.
ChannelMessage — fan-out to all members of a named channel.
CreateChannel — register a new channel with a fixed member set.
FetchMessages — returns both DMs and channel messages addressed to the caller.
"""

from src.privacy.core import (
    ActionExecutionResult,
    AgentProfile,
    ChannelMessage,
    CreateChannel,
    FetchMessages,
    FetchMessagesResponse,
    MessageAdapter,
    ReadMessageError,
    ReadMessages,
    ReadMessagesResponse,
    ReceivedMessage,
    SendMessage,
)
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.models import ActionRow


_SEND_MESSAGE = SendMessage.get_name()
_CHANNEL_MESSAGE = ChannelMessage.get_name()


# --- SendMessage (DM) --------------------------------------------------------


async def execute_send_message(
    action: SendMessage,
    database: BaseDatabaseController,
) -> ActionExecutionResult:
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


# --- ChannelMessage -----------------------------------------------------------


async def execute_channel_message(
    action: ChannelMessage,
    channels: dict[str, set[str]],
) -> ActionExecutionResult:
    members = channels.get(action.channel)
    if members is None:
        return ActionExecutionResult(
            content={"error": f"channel {action.channel!r} does not exist"},
            is_error=True,
        )
    if action.from_agent_id not in members:
        return ActionExecutionResult(
            content={"error": f"agent {action.from_agent_id!r} is not a member of {action.channel!r}"},
            is_error=True,
        )
    return ActionExecutionResult(
        content=action.model_dump(mode="json"),
        is_error=False,
        metadata={"status": "sent", "recipient_count": len(members) - 1},
    )


# --- CreateChannel ------------------------------------------------------------


async def execute_create_channel(
    action: CreateChannel,
    channels: dict[str, set[str]],
    agent_id: str,
) -> ActionExecutionResult:
    if action.channel in channels:
        return ActionExecutionResult(
            content={"error": f"channel {action.channel!r} already exists"},
            is_error=True,
        )
    members = set(action.member_ids) | {agent_id}
    channels[action.channel] = members
    return ActionExecutionResult(
        content={"channel": action.channel, "members": sorted(members)},
        is_error=False,
        metadata={"status": "created"},
    )


# --- FetchMessages (DMs + channel messages) -----------------------------------


async def execute_fetch_messages(
    action: FetchMessages,
    agent: AgentProfile,
    database: BaseDatabaseController,
    channels: dict[str, set[str]],
) -> ActionExecutionResult:
    agent_channels = {
        name for name, members in channels.items() if agent.id in members
    }

    def predicate(row: ActionRow) -> bool:
        name = row.data.request.name
        params = row.data.request.parameters

        if name == _SEND_MESSAGE:
            if params.get("to_agent_id") != agent.id:
                return False
        elif name == _CHANNEL_MESSAGE:
            ch = params.get("channel")
            if ch not in agent_channels:
                return False
            if params.get("from_agent_id") == agent.id:
                return False
        else:
            return False

        return True

    rows = await database.actions.find(predicate)
    messages = [_to_received_message(row) for row in rows]
    return ActionExecutionResult(
        content=FetchMessagesResponse(messages=messages).model_dump(mode="json"),
    )


# --- ReadMessages (pull bodies by index) --------------------------------------


async def execute_read_messages(
    action: ReadMessages,
    agent: AgentProfile,
    database: BaseDatabaseController,
    channels: dict[str, set[str]],
) -> ActionExecutionResult:
    agent_channels = {
        name for name, members in channels.items() if agent.id in members
    }
    requested = list(action.indexes)
    requested_set = set(requested)

    def predicate(row: ActionRow) -> bool:
        if row.index is None or row.index not in requested_set:
            return False
        return row.data.request.name in (_SEND_MESSAGE, _CHANNEL_MESSAGE)

    rows = await database.actions.find(predicate)
    by_index: dict[int, ActionRow] = {row.index: row for row in rows if row.index is not None}

    def is_authorized(row: ActionRow) -> bool:
        name = row.data.request.name
        params = row.data.request.parameters
        if name == _SEND_MESSAGE:
            return params.get("to_agent_id") == agent.id
        if name == _CHANNEL_MESSAGE:
            return params.get("channel") in agent_channels
        return False

    messages: list[ReceivedMessage] = []
    errors: list[ReadMessageError] = []
    for idx in requested:
        row = by_index.get(idx)
        if row is None:
            errors.append(ReadMessageError(index=idx, error="not_found"))
            continue
        if not is_authorized(row):
            errors.append(ReadMessageError(index=idx, error="not_authorized"))
            continue
        messages.append(_to_received_message(row))

    response = ReadMessagesResponse(messages=messages, errors=errors)
    return ActionExecutionResult(content=response.model_dump(mode="json"))


def _to_received_message(action_row: ActionRow) -> ReceivedMessage:
    params = action_row.data.request.parameters
    if action_row.index is None:
        raise ValueError("action_row.index must be populated by the database backend")

    name = action_row.data.request.name
    channel = params.get("channel") if name == _CHANNEL_MESSAGE else None
    to_agent_id = params.get("to_agent_id", "*")

    return ReceivedMessage(
        from_agent_id=params["from_agent_id"],
        to_agent_id=to_agent_id,
        channel=channel,
        created_at=action_row.created_at,
        message=MessageAdapter.validate_python(params.get("message", {})),
        index=action_row.index,
    )
