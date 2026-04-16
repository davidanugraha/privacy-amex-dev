"""Framework-level Pydantic types."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny


class AgentProfile(BaseModel):
    """Base agent profile. Subclass with `extra="allow"` lets callers add fields."""

    model_config = ConfigDict(extra="allow")
    id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionProtocol(BaseModel):
    """Descriptor of an action type (name, description, JSON Schema)."""

    name: str
    description: str
    parameters: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseAction(BaseModel):
    """Base class for all actions an agent can invoke via the protocol."""

    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    @classmethod
    def get_description(cls) -> str:
        return cls.__doc__ or ""

    @classmethod
    def get_parameters(cls) -> dict[str, Any]:
        return cls.model_json_schema()

    @classmethod
    def to_protocol(cls) -> ActionProtocol:
        return ActionProtocol(
            name=cls.get_name(),
            description=cls.get_description(),
            parameters=cls.get_parameters(),
            metadata={"source": "BaseAction"},
        )


class ActionExecutionRequest(BaseModel):
    """Envelope a caller submits to `protocol.execute_action`."""

    name: str
    parameters: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionExecutionResult(BaseModel):
    """Envelope the protocol returns from `execute_action`."""

    content: Any
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


LogLevel = Literal["debug", "info", "warning", "error"]


class Log(BaseModel):
    """A single log record."""

    model_config = ConfigDict(extra="allow")
    level: LogLevel
    name: str
    message: str | None = None
    data: dict[str, Any] | SerializeAsAny[BaseModel] | None = None
    metadata: dict[str, Any] | None = None
