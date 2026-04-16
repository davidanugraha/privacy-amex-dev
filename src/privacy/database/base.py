"""Base database classes and interfaces."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import AwareDatetime, BaseModel

from .models import ActionRow, AgentRow, LogRow

TableEntryType = TypeVar("TableEntryType")


class RangeQueryParams(BaseModel):
    """Pagination + time-window parameters shared by list/find calls."""

    offset: int = 0
    limit: int | None = None
    after: AwareDatetime | None = None
    before: AwareDatetime | None = None
    after_index: int | None = None
    before_index: int | None = None


class TableController(ABC, Generic[TableEntryType]):  # noqa: UP046
    """CRUD interface for a single table, parameterized by row type."""

    @abstractmethod
    async def create(self, item: TableEntryType) -> TableEntryType:
        """Insert an item; returns the stored row with id/index populated."""

    @abstractmethod
    async def get_by_id(self, item_id: str) -> TableEntryType | None:
        """Fetch by id, or None if missing."""

    @abstractmethod
    async def get_all(
        self, params: RangeQueryParams | None = None
    ) -> list[TableEntryType]:
        """List rows with optional pagination and time-window filtering."""

    @abstractmethod
    async def find(
        self,
        predicate: Callable[[TableEntryType], bool],
        params: RangeQueryParams | None = None,
    ) -> list[TableEntryType]:
        """Return rows matching `predicate`, with optional pagination."""

    @abstractmethod
    async def update(
        self, item_id: str, updates: dict[str, Any]
    ) -> TableEntryType | None:
        """Merge `updates` into the row with `item_id`; None if missing."""

    @abstractmethod
    async def count(self) -> int:
        """Total row count."""


class AgentTableController(TableController[AgentRow]):
    """Abstract controller for Agent operations."""


class ActionTableController(TableController[ActionRow]):
    """Abstract controller for Action operations."""


class LogTableController(TableController[LogRow]):
    """Abstract controller for Log operations."""


class BaseDatabaseController(ABC):
    """Database controller that owns all entity controllers."""

    @property
    @abstractmethod
    def agents(self) -> AgentTableController: ...

    @property
    @abstractmethod
    def actions(self) -> ActionTableController: ...

    @property
    @abstractmethod
    def logs(self) -> LogTableController: ...
