"""Concrete action and message types.

Organized in sections:
  1. Message payload types
  2. DM + channel messaging actions
  3. Channel management actions
  4. Sandbox execution action
  5. Action union + TypeAdapter
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, Field
from pydantic.type_adapter import TypeAdapter

from .types import BaseAction


# --- Message payloads ---------------------------------------------------------


class TextMessage(BaseModel):
    """A text message."""

    type: Literal["text"] = "text"
    content: str = Field(description="Text content of the message")


Message = TextMessage

MessageAdapter: TypeAdapter[Message] = TypeAdapter(Message)


# --- Messaging actions --------------------------------------------------------


class SendMessage(BaseAction):
    """DM — send a message directly to one agent."""

    type: Literal["send_message"] = "send_message"
    from_agent_id: str = Field(description="ID of the sender")
    to_agent_id: str = Field(description="ID of the recipient")
    created_at: AwareDatetime = Field(description="When the message was created")
    message: Message = Field(description="The message to send")


class ChannelMessage(BaseAction):
    """Send a message to all members of a channel."""

    type: Literal["channel_message"] = "channel_message"
    from_agent_id: str = Field(description="ID of the sender")
    channel: str = Field(description="Channel name (e.g. 'general')")
    created_at: AwareDatetime = Field(description="When the message was created")
    message: Message = Field(description="The message to send")


class CreateChannel(BaseAction):
    """Create a new channel with a fixed member set."""

    type: Literal["create_channel"] = "create_channel"
    channel: str = Field(description="Channel name")
    member_ids: list[str] = Field(description="Agent IDs to include")


class FetchMessages(BaseAction):
    """Fetch all messages addressed to this agent — DMs + channel messages.

    Returns every matching message; agent-side code dedupes by index.
    """

    type: Literal["fetch_messages"] = "fetch_messages"


class ReceivedMessage(BaseModel):
    """A message as received by an agent with metadata."""

    from_agent_id: str = Field(description="ID of the sender")
    to_agent_id: str = Field(description="Recipient agent ID, or '*' for channel messages")
    channel: str | None = Field(default=None, description="Channel name if this was a channel message")
    created_at: AwareDatetime = Field(description="When the message was created")
    message: Message = Field(description="The actual message content")
    index: int = Field(description="The row index of the message")


class FetchMessagesResponse(BaseModel):
    """Response from fetching messages."""

    messages: list[ReceivedMessage] = Field(description="List of received messages")


class ReadMessages(BaseAction):
    """Read full content of specific messages by their inbox indexes.

    Authorization mirrors FetchMessages: the requesting agent must be a
    direct recipient (DM) or a member of the channel the message was
    posted to. Unknown or unauthorized indexes come back as error
    entries rather than failing the whole call.
    """

    type: Literal["read_messages"] = "read_messages"
    indexes: list[int] = Field(description="Inbox indexes to read")


class ReadMessageError(BaseModel):
    """Per-index failure entry returned by ReadMessages."""

    index: int = Field(description="The requested index that failed")
    error: Literal["not_found", "not_authorized"] = Field(description="Failure reason")


class ReadMessagesResponse(BaseModel):
    """Response from reading messages by index."""

    messages: list[ReceivedMessage] = Field(default_factory=list)
    errors: list[ReadMessageError] = Field(default_factory=list)


# --- Sandbox execution --------------------------------------------------------


class ExecuteCommand(BaseAction):
    """Execute a shell command inside the invoking agent's persistent sandbox workspace.

    The substrate wraps the command in `bash -c`, so pipes / redirects /
    heredoc / `&&` / `||` all work. Matches the surface of Claude Code's
    Bash tool and SWE-bench's per-step exec.
    """

    type: Literal["execute_command"] = "execute_command"
    agent_id: str = Field(description="ID of the agent running the command")
    command: str = Field(description="Shell command string (executed via `bash -c`)")
    stdin: str | None = Field(default=None, description="Optional stdin (not LLM-exposed; substrate-internal)")
    timeout_seconds: int = Field(default=30, description="Hard wall-clock timeout")


class ExecuteCommandResult(BaseModel):
    """Result of a sandboxed command execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: float


# --- Lifecycle actions --------------------------------------------------------


class MarkDone(BaseAction):
    """Signal that the agent has completed its task and will shut down.

    Goes through `protocol._dispatch` like every other action so the
    action row is recorded in the trajectory and any future lifecycle
    policy hook can gate it. The agent's `_dispatch_tool_call` for
    `mark_done` calls this and then sets `will_shutdown=True`.
    """

    type: Literal["mark_done"] = "mark_done"
    agent_id: str = Field(description="ID of the agent marking itself done")


# --- Action union -------------------------------------------------------------


Action = Annotated[
    SendMessage | ChannelMessage | CreateChannel | FetchMessages | ReadMessages | ExecuteCommand | MarkDone,
    Field(discriminator="type"),
]

ActionAdapter: TypeAdapter[Action] = TypeAdapter(Action)
