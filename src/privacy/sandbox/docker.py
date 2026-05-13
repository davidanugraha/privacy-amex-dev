"""Docker-backed sandbox — persistent per-agent container.

Each agent gets a long-lived container (started with `sleep infinity`) at
setup. Commands run via `docker exec` into it; files are copied in at setup
via `put_archive`. At teardown the container is removed.

The `docker` SDK is imported lazily inside `__init__` so this module imports
fine without Docker installed; the error surfaces only when `DockerSandbox`
is actually constructed.

Wall-clock timeout enforcement: every exec is wrapped with coreutils
`timeout`, so the container image must ship `/usr/bin/timeout` (true for
`python:3.11-slim` and any debian/ubuntu-based image; busybox `timeout`
uses different flags and is not supported here).
"""

import asyncio
import io
import logging
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from src.privacy.core import ExecuteCommandResult

log = logging.getLogger(__name__)


class DockerSandbox:
    """Run commands inside a long-lived per-agent Docker container."""

    def __init__(
        self,
        *,
        mem_limit: str = "4g",
        nano_cpus: int | None = None,
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
        self._snapshot_root = Path(tempfile.mkdtemp(prefix="privacy-docker-snap-"))
        self._snapshots: dict[str, Path] = {}

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
        run_kwargs: dict[str, Any] = dict(
            image=image,
            command=["sleep", "infinity"],
            working_dir=self._workdir,
            network_disabled=self._network_disabled,
            mem_limit=self._mem_limit,
            detach=True,
            tty=False,
            remove=False,
        )
        if self._nano_cpus is not None:
            run_kwargs["nano_cpus"] = self._nano_cpus
        container = self._client.containers.run(**run_kwargs)
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
        command: str,
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
        command: str,
        stdin: str | None,
        timeout: int,
    ) -> ExecuteCommandResult:
        if stdin is not None:
            raise NotImplementedError(
                "DockerSandbox does not support stdin (silently dropped today). "
                "Have the caller write input to a file in the workspace and pass "
                "the filename as argv instead. See REVIEW.md B2."
            )

        container = self._containers.get(agent_id)
        if container is None:
            raise RuntimeError(
                f"Sandbox.run called for agent {agent_id!r} without prior setup()"
            )

        api = self._client.api
        started = time.monotonic()

        # Wrap with coreutils `timeout` (wall-clock enforcement) and `bash -c`
        # (shell features — pipes/redirects/heredoc/&&/||). Exit 124 = timeout
        # fired (SIGTERM); 137 = SIGKILL after grace.
        wrapped_cmd = ["timeout", "--kill-after=2", f"{timeout}s", "bash", "-c", command]

        exec_id = api.exec_create(
            container.id,
            cmd=wrapped_cmd,
            stdin=False,
            stdout=True,
            stderr=True,
            tty=False,
            workdir=self._workdir,
        )["Id"]

        output_stream = api.exec_start(exec_id, detach=False, stream=False, demux=True)
        # demux=True returns (stdout_bytes, stderr_bytes) tuple when stream=False.

        if isinstance(output_stream, tuple):
            stdout_b, stderr_b = output_stream
        else:
            # Fallback for older SDK: single stream, assume stdout.
            stdout_b, stderr_b = output_stream, b""

        inspect = api.exec_inspect(exec_id)
        exit_code = inspect.get("ExitCode")
        if exit_code is None:
            exit_code = -1

        timed_out = exit_code == 124

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
        if container is None:
            return
        try:
            self._snapshot_workspace(agent_id, container)
        except Exception:
            log.exception(
                "docker sandbox: snapshot failed for %s; FileExists eval "
                "criteria will return None for this agent's workspace",
                agent_id,
            )
        try:
            container.remove(force=True)
        except Exception:
            log.exception(
                "docker sandbox: container.remove failed for %s; "
                "container may leak (check `docker ps -a`)",
                agent_id,
            )

    def _snapshot_workspace(self, agent_id: str, container: Any) -> None:
        snap_dir = self._snapshot_root / f"agent-{agent_id}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots[agent_id] = snap_dir
        stream, _ = container.get_archive(self._workdir)
        buf = io.BytesIO(b"".join(stream))
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r") as tar:
            # Strip the leading workdir component so paths are relative
            root_name = Path(self._workdir).name
            for member in tar.getmembers():
                name = Path(member.name)
                parts = name.parts
                if parts and parts[0] == root_name:
                    member.name = str(Path(*parts[1:])) if len(parts) > 1 else ""
                if not member.name:
                    continue
                tar.extract(member, path=snap_dir)

    async def read_file(self, agent_id: str, path: str) -> str | None:
        """Read a file from the agent's workspace.

        Tries the live container first (via `get_archive`) so mid-run reads
        work. Falls back to the post-teardown snapshot. Matches SWE-bench /
        HCAST semantics where the evaluator can inspect files while the
        agent is still running.
        """
        return await asyncio.to_thread(self._read_file_blocking, agent_id, path)

    def _read_file_blocking(self, agent_id: str, path: str) -> str | None:
        container = self._containers.get(agent_id)
        if container is not None:
            content = self._read_from_container(container, path)
            if content is not None:
                return content
        snap = self._snapshots.get(agent_id)
        if snap is None:
            return None
        target = snap / path
        if not target.is_file():
            return None
        try:
            return target.read_text()
        except (OSError, UnicodeDecodeError):
            return None

    def _read_from_container(self, container: Any, path: str) -> str | None:
        try:
            stream, _ = container.get_archive(f"{self._workdir}/{path}")
            buf = io.BytesIO(b"".join(stream))
            buf.seek(0)
            with tarfile.open(fileobj=buf, mode="r") as tar:
                members = tar.getmembers()
                if not members:
                    return None
                f = tar.extractfile(members[0])
                if f is None:
                    return None
                return f.read().decode(errors="replace")
        except Exception:
            return None
