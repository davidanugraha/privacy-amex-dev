"""Behavior protocol interface for agent actions."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..core import (
    ActionExecutionRequest,
    ActionExecutionResult,
    ActionProtocol,
    AgentProfile,
    BaseAction,
)
from ..database.base import BaseDatabaseController


class BasePrivacyProtocol(ABC):
    """Abstract interface for defining agent behavior protocols."""

    @abstractmethod
    def get_actions(self) -> Sequence[ActionProtocol | type[BaseAction]]:
        """Get available actions for this protocol."""
        ...

    @abstractmethod
    async def execute_action(
        self,
        *,
        agent: AgentProfile,
        action: ActionExecutionRequest,
        database: BaseDatabaseController,
    ) -> ActionExecutionResult:
        """Execute a specific action with the given name and parameters."""
        ...

    async def initialize(self, database: BaseDatabaseController) -> None:  # noqa: B027
        """Initialize protocol-specific resources (e.g., database indexes).

        This method is called during server startup after the database is initialized.
        Override this method to perform protocol-specific setup like creating indexes.

        Args:
            database: The database controller instance

        """
        # Default implementation does nothing
