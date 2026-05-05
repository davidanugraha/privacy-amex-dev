"""Sandbox protocol — per-agent persistent workspace.

Each agent's workspace is created at setup, commands execute into it and share
state across calls, and the workspace is released at teardown. Files can be
pre-provisioned at setup for scenario data.
"""

from typing import Protocol

from src.privacy.core import ExecuteCommandResult


class Sandbox(Protocol):
    """A per-agent workspace executor."""

    async def setup(
        self,
        agent_id: str,
        *,
        image: str = "",
        files: dict[str, str] | None = None,
    ) -> None:
        """Create a persistent workspace for `agent_id`.

        Args:
            agent_id: Unique identifier; keys the workspace across calls.
            image: Container image (ignored by non-container backends).
            files: Initial workspace contents as path → text content.
                   Paths are relative to the workspace root.
        """
        ...

    async def run(
        self,
        *,
        agent_id: str,
        command: list[str],
        stdin: str | None,
        timeout: int,
    ) -> ExecuteCommandResult:
        """Run `command` inside the agent's workspace.

        Args:
            agent_id: Must have been set up via `setup()`.
            command: Argv-style command.
            stdin: Optional stdin payload.
            timeout: Wall-clock timeout; on expiry `timed_out=True`.
        """
        ...

    async def teardown(self, agent_id: str) -> None:
        """Release the workspace for `agent_id`.

        Implementations may keep the workspace contents on disk for post-hoc
        analysis; this call just releases any live resources (running containers,
        in-memory handles).
        """
        ...

    async def read_file(self, agent_id: str, path: str) -> str | None:
        """Read a file from the agent's workspace. Returns None if missing.

        Used by post-run evaluators (`FileExists` criterion, etc). Path is
        relative to the workspace root.
        """
        ...
