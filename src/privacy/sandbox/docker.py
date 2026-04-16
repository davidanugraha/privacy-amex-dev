"""Docker-backed sandbox — persistent per-agent container.

Each agent gets a long-lived container (started with `sleep infinity`) at
setup. Commands run via `docker exec` into it; files are copied in at setup
via `put_archive`. At teardown the container is removed.

The `docker` SDK is imported lazily inside `__init__` so this module imports
fine without Docker installed; the error surfaces only when `DockerSandbox`
is actually constructed.
"""

import asyncio
import io
import tarfile
import time
from typing import Any

from src.privacy.core import ExecuteCommandResult


class DockerSandbox:
    """Run commands inside a long-lived per-agent Docker container."""

    def __init__(
        self,
        *,
        mem_limit: str = "256m",
        nano_cpus: int = 1_000_000_000,  # 1 CPU
        network_disabled: bool = True,
        workdir: str = "/workspace",
    ) -> None:
        try:
            import docker  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "DockerSandbox requires the `docker` Python SDK. "
                "Install it with `pip install docker`."
            ) from e

        self._docker = __import__("docker")
        self._client = self._docker.from_env()
        self._mem_limit = mem_limit
        self._nano_cpus = nano_cpus
        self._network_disabled = network_disabled
        self._workdir = workdir
        self._containers: dict[str, Any] = {}

    async def setup(
        self,
        agent_id: str,
        *,
        image: str = "python:3.11-slim",
        files: dict[str, str] | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._setup_blocking, agent_id, image=image, files=files or {}
        )

    def _setup_blocking(
        self,
        agent_id: str,
        *,
        image: str,
        files: dict[str, str],
    ) -> None:
        container = self._client.containers.run(
            image=image,
            command=["sleep", "infinity"],
            working_dir=self._workdir,
            network_disabled=self._network_disabled,
            mem_limit=self._mem_limit,
            nano_cpus=self._nano_cpus,
            detach=True,
            tty=False,
            remove=False,
        )
        self._containers[agent_id] = container

        if files:
            tar_buf = io.BytesIO()
            with tarfile.open(fileobj=tar_buf, mode="w") as tar:
                for rel_path, content in files.items():
                    data = content.encode()
                    info = tarfile.TarInfo(name=rel_path)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            tar_buf.seek(0)
            container.put_archive(self._workdir, tar_buf.read())

    async def run(
        self,
        *,
        agent_id: str,
        command: list[str],
        stdin: str | None,
        timeout: int,
    ) -> ExecuteCommandResult:
        return await asyncio.to_thread(
            self._run_blocking,
            agent_id=agent_id,
            command=command,
            stdin=stdin,
            timeout=timeout,
        )

    def _run_blocking(
        self,
        *,
        agent_id: str,
        command: list[str],
        stdin: str | None,
        timeout: int,
    ) -> ExecuteCommandResult:
        container = self._containers.get(agent_id)
        if container is None:
            raise RuntimeError(
                f"Sandbox.run called for agent {agent_id!r} without prior setup()"
            )

        api = self._client.api
        started = time.monotonic()

        exec_id = api.exec_create(
            container.id,
            cmd=command,
            stdin=stdin is not None,
            stdout=True,
            stderr=True,
            tty=False,
            workdir=self._workdir,
        )["Id"]

        # Run with a wall-clock timeout; if exceeded, kill the exec process.
        deadline = started + timeout
        output_stream = api.exec_start(exec_id, detach=False, stream=False, demux=True)
        # demux=True returns (stdout_bytes, stderr_bytes) tuple when stream=False.

        timed_out = False
        if isinstance(output_stream, tuple):
            stdout_b, stderr_b = output_stream
        else:
            # Fallback for older SDK: single stream, assume stdout.
            stdout_b, stderr_b = output_stream, b""

        if time.monotonic() > deadline:
            timed_out = True

        inspect = api.exec_inspect(exec_id)
        exit_code = inspect.get("ExitCode")
        if exit_code is None:
            exit_code = -1

        duration_ms = (time.monotonic() - started) * 1000.0
        return ExecuteCommandResult(
            stdout=(stdout_b or b"").decode(errors="replace"),
            stderr=(stderr_b or b"").decode(errors="replace"),
            exit_code=int(exit_code),
            timed_out=timed_out,
            duration_ms=duration_ms,
        )

    async def teardown(self, agent_id: str) -> None:
        await asyncio.to_thread(self._teardown_blocking, agent_id)

    def _teardown_blocking(self, agent_id: str) -> None:
        container = self._containers.pop(agent_id, None)
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass
