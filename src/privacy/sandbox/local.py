"""Local subprocess sandbox — persistent per-agent workspace directory.

Each agent gets a subdirectory under a shared root; commands run with that
subdirectory as cwd. No isolation; use Docker backend for that.
"""

import asyncio
import shutil
import tempfile
import time
from pathlib import Path

from src.privacy.core import ExecuteCommandResult


class LocalSubprocessSandbox:
    """Run commands as local subprocesses with per-agent persistent workspaces."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        cleanup_on_teardown: bool = False,
    ) -> None:
        self._root = Path(root) if root else Path(tempfile.mkdtemp(prefix="privacy-sandbox-"))
        self._root.mkdir(parents=True, exist_ok=True)
        self._cleanup_on_teardown = cleanup_on_teardown
        self._workspaces: dict[str, Path] = {}

    async def setup(
        self,
        agent_id: str,
        *,
        image: str = "",  # noqa: ARG002 — accepted for interface parity
        files: dict[str, str] | None = None,
    ) -> None:
        ws = self._root / f"agent-{agent_id}"
        ws.mkdir(parents=True, exist_ok=True)
        self._workspaces[agent_id] = ws

        for rel_path, content in (files or {}).items():
            target = ws / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

    async def run(
        self,
        *,
        agent_id: str,
        command: list[str],
        stdin: str | None,
        timeout: int,
    ) -> ExecuteCommandResult:
        ws = self._workspaces.get(agent_id)
        if ws is None:
            raise RuntimeError(
                f"Sandbox.run called for agent {agent_id!r} without prior setup()"
            )

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ws),
        )

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=stdin.encode() if stdin is not None else None),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            proc.kill()
            stdout_b, stderr_b = await proc.communicate()

        duration_ms = (time.monotonic() - started) * 1000.0
        return ExecuteCommandResult(
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )

    async def teardown(self, agent_id: str) -> None:
        ws = self._workspaces.pop(agent_id, None)
        if ws is not None and self._cleanup_on_teardown:
            shutil.rmtree(ws, ignore_errors=True)
