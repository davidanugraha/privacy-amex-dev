"""Privacy logger — writes to Python logging + the in-process database."""

import asyncio
import logging
import traceback
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from .database.base import BaseDatabaseController
from .database.models import LogRow
from .core import Log, LogLevel


class PrivacyLogger:
    """Dual logger: Python `logging` for stdout + database for post-hoc analysis."""

    def __init__(self, name: str, database: BaseDatabaseController):
        self.name = name
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
            )
        self.python_logger = logging.getLogger(name)
        self._database = database
        self._tasks: list[asyncio.Task] = []

    def _log(
        self,
        level: LogLevel,
        message: str | None = None,
        *,
        data: dict[str, Any] | BaseModel | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if message is None and data is None:
            raise ValueError("Must provide at least one of message or data.")

        self.python_logger.log(getattr(logging, level.upper()), message)

        log = Log(
            level=level, name=self.name, message=message, data=data, metadata=metadata
        )
        task = asyncio.create_task(self._log_to_db(log))
        self._tasks.append(task)
        task.add_done_callback(self._remove_task)
        return task

    async def _log_to_db(self, log: Log) -> None:
        try:
            await self._database.logs.create(
                LogRow(id="", created_at=datetime.now(UTC), data=log)
            )
        except Exception:
            self.python_logger.error(
                f"Failed to log to database: {traceback.format_exc()}"
            )

    def debug(self, message=None, *, data=None, metadata=None):
        return self._log("debug", message, data=data, metadata=metadata)

    def info(self, message=None, *, data=None, metadata=None):
        return self._log("info", message, data=data, metadata=metadata)

    def warning(self, message=None, *, data=None, metadata=None):
        return self._log("warning", message, data=data, metadata=metadata)

    def error(self, message=None, *, data=None, metadata=None):
        return self._log("error", message, data=data, metadata=metadata)

    def exception(self, message=None, *, data=None, metadata=None):
        message = ((message or "") + "\n" + traceback.format_exc(2)).strip()
        return self.error(message, data=data, metadata=metadata)

    def _remove_task(self, task: asyncio.Task):
        try:
            self._tasks.remove(task)
        except ValueError:
            self.python_logger.debug("Failed to remove task: task is not in list.")

    async def flush(self):
        tasks = list(self._tasks)
        self._tasks.clear()
        return await asyncio.gather(*tasks, return_exceptions=True)
