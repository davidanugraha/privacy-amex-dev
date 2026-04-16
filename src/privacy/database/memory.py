"""In-memory database backend."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from .base import (
    ActionTableController,
    AgentTableController,
    BaseDatabaseController,
    LogTableController,
    RangeQueryParams,
)
from .models import ActionRow, AgentRow, LogRow, Row

T = TypeVar("T")
RowT = TypeVar("RowT", bound=Row)


class _MemoryTable(Generic[RowT]):
    """Shared storage + query machinery for any Row subclass."""

    def __init__(self) -> None:
        self._rows: list[RowT] = []
        self._by_id: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._next_index = 0

    async def create(self, item: RowT) -> RowT:
        async with self._lock:
            if not item.id:
                item.id = str(uuid.uuid4())
            if item.id in self._by_id:
                raise ValueError(f"duplicate id: {item.id}")
            item.index = self._next_index
            self._next_index += 1
            self._by_id[item.id] = len(self._rows)
            self._rows.append(item)
            return item

    async def get_by_id(self, item_id: str) -> RowT | None:
        idx = self._by_id.get(item_id)
        return self._rows[idx] if idx is not None else None

    async def get_all(
        self, params: RangeQueryParams | None = None
    ) -> list[RowT]:
        return self._apply(list(self._rows), params)

    async def find(
        self,
        predicate: Callable[[RowT], bool],
        params: RangeQueryParams | None = None,
    ) -> list[RowT]:
        filtered = [r for r in self._rows if predicate(r)]
        return self._apply(filtered, params)

    async def update(self, item_id: str, updates: dict[str, Any]) -> RowT | None:
        async with self._lock:
            idx = self._by_id.get(item_id)
            if idx is None:
                return None
            existing = self._rows[idx]
            merged = existing.model_dump()
            merged.update(updates)
            # Preserve id and index across updates.
            merged["id"] = existing.id
            merged["index"] = existing.index
            new_row = type(existing).model_validate(merged)
            self._rows[idx] = new_row
            return new_row

    async def count(self) -> int:
        return len(self._rows)

    @staticmethod
    def _apply(rows: list[RowT], params: RangeQueryParams | None) -> list[RowT]:
        if params is None:
            return rows
        if params.after is not None:
            rows = [r for r in rows if r.created_at > params.after]
        if params.before is not None:
            rows = [r for r in rows if r.created_at < params.before]
        if params.after_index is not None:
            rows = [r for r in rows if r.index is not None and r.index > params.after_index]
        if params.before_index is not None:
            rows = [r for r in rows if r.index is not None and r.index < params.before_index]
        start = params.offset
        end = start + params.limit if params.limit is not None else None
        return rows[start:end]


class MemoryAgentController(AgentTableController):
    def __init__(self) -> None:
        self._t: _MemoryTable[AgentRow] = _MemoryTable()

    async def create(self, item: AgentRow) -> AgentRow:
        if not item.created_at:
            item.created_at = datetime.now(UTC)
        return await self._t.create(item)

    async def get_by_id(self, item_id: str) -> AgentRow | None:
        return await self._t.get_by_id(item_id)

    async def get_all(self, params: RangeQueryParams | None = None) -> list[AgentRow]:
        return await self._t.get_all(params)

    async def find(
        self,
        predicate: Callable[[AgentRow], bool],
        params: RangeQueryParams | None = None,
    ) -> list[AgentRow]:
        return await self._t.find(predicate, params)

    async def update(self, item_id: str, updates: dict[str, Any]) -> AgentRow | None:
        return await self._t.update(item_id, updates)

    async def count(self) -> int:
        return await self._t.count()


class MemoryActionController(ActionTableController):
    def __init__(self) -> None:
        self._t: _MemoryTable[ActionRow] = _MemoryTable()

    async def create(self, item: ActionRow) -> ActionRow:
        if not item.created_at:
            item.created_at = datetime.now(UTC)
        return await self._t.create(item)

    async def get_by_id(self, item_id: str) -> ActionRow | None:
        return await self._t.get_by_id(item_id)

    async def get_all(self, params: RangeQueryParams | None = None) -> list[ActionRow]:
        return await self._t.get_all(params)

    async def find(
        self,
        predicate: Callable[[ActionRow], bool],
        params: RangeQueryParams | None = None,
    ) -> list[ActionRow]:
        return await self._t.find(predicate, params)

    async def update(self, item_id: str, updates: dict[str, Any]) -> ActionRow | None:
        return await self._t.update(item_id, updates)

    async def count(self) -> int:
        return await self._t.count()


class MemoryLogController(LogTableController):
    def __init__(self) -> None:
        self._t: _MemoryTable[LogRow] = _MemoryTable()

    async def create(self, item: LogRow) -> LogRow:
        if not item.created_at:
            item.created_at = datetime.now(UTC)
        return await self._t.create(item)

    async def get_by_id(self, item_id: str) -> LogRow | None:
        return await self._t.get_by_id(item_id)

    async def get_all(self, params: RangeQueryParams | None = None) -> list[LogRow]:
        return await self._t.get_all(params)

    async def find(
        self,
        predicate: Callable[[LogRow], bool],
        params: RangeQueryParams | None = None,
    ) -> list[LogRow]:
        return await self._t.find(predicate, params)

    async def update(self, item_id: str, updates: dict[str, Any]) -> LogRow | None:
        return await self._t.update(item_id, updates)

    async def count(self) -> int:
        return await self._t.count()


class MemoryDatabase(BaseDatabaseController):
    """In-memory implementation of BaseDatabaseController."""

    def __init__(self) -> None:
        self._agents = MemoryAgentController()
        self._actions = MemoryActionController()
        self._logs = MemoryLogController()

    @property
    def agents(self) -> AgentTableController:
        return self._agents

    @property
    def actions(self) -> ActionTableController:
        return self._actions

    @property
    def logs(self) -> LogTableController:
        return self._logs


@asynccontextmanager
async def connect_to_memory_database() -> AsyncIterator[BaseDatabaseController]:
    """Factory that yields a fresh MemoryDatabase as an async context manager."""
    yield MemoryDatabase()
