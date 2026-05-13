"""Sandbox backend factory."""

import os
from typing import Literal

from .base import Sandbox

SandboxBackend = Literal["docker", "local"]


def build_sandbox(
    backend: SandboxBackend | None = None,
    *,
    docker_mem_limit: str = "4g",
    docker_cpus: float | None = None,
    docker_network_disabled: bool = True,
) -> Sandbox:
    """Construct a Sandbox of the requested backend.

    If `backend` is None, read from the `SANDBOX_BACKEND` env var, defaulting to
    `"local"` so code works out of the box without a Docker daemon.

    Docker-specific knobs (`docker_*`) are ignored by the local backend.
    `docker_cpus=None` means no CPU pinning (matches typical CLI deployment).
    """
    chosen = backend or os.environ.get("SANDBOX_BACKEND", "local")
    if chosen == "docker":
        from .docker import DockerSandbox
        return DockerSandbox(
            mem_limit=docker_mem_limit,
            nano_cpus=int(docker_cpus * 1_000_000_000) if docker_cpus else None,
            network_disabled=docker_network_disabled,
        )
    if chosen == "local":
        from .local import LocalSubprocessSandbox
        return LocalSubprocessSandbox()
    raise ValueError(f"Unknown sandbox backend: {chosen!r}")
