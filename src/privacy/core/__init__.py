"""Core framework types and concrete action types.

Re-exports everything from `types` and `actions` so callers can simply do:

    from src.privacy.core import SendMessage, AgentProfile, BaseAction
"""

from .actions import (
    Action,
    ActionAdapter,
    ChannelMessage,
    CreateChannel,
    ExecuteCommand,
    ExecuteCommandResult,
    FetchMessages,
    FetchMessagesResponse,
    MarkDone,
    Message,
    MessageAdapter,
    ReadMessageError,
    ReadMessages,
    ReadMessagesResponse,
    ReceivedMessage,
    SendMessage,
    TextMessage,
)
from .types import (
    ActionExecutionRequest,
    ActionExecutionResult,
    ActionProtocol,
    AgentProfile,
    BaseAction,
    Log,
    LogLevel,
)

__all__ = [
    "Action",
    "ActionAdapter",
    "ActionExecutionRequest",
    "ActionExecutionResult",
    "ActionProtocol",
    "AgentProfile",
    "BaseAction",
    "ChannelMessage",
    "CreateChannel",
    "ExecuteCommand",
    "ExecuteCommandResult",
    "FetchMessages",
    "FetchMessagesResponse",
    "Log",
    "LogLevel",
    "MarkDone",
    "Message",
    "MessageAdapter",
    "ReadMessageError",
    "ReadMessages",
    "ReadMessagesResponse",
    "ReceivedMessage",
    "SendMessage",
    "TextMessage",
]
