"""Sandbox backend factory."""

import os
from typing import Literal

from .base import Sandbox

SandboxBackend = Literal["docker", "local"]


def build_sandbox(backend: SandboxBackend | None = None) -> Sandbox:
    """Construct a Sandbox of the requested backend.

    If `backend` is None, read from the `SANDBOX_BACKEND` env var, defaulting to
    `"local"` so code works out of the box without a Docker daemon.
    """
    chosen = backend or os.environ.get("SANDBOX_BACKEND", "local")
    if chosen == "docker":
        from .docker import DockerSandbox
        return DockerSandbox()
    if chosen == "local":
        from .local import LocalSubprocessSandbox
        return LocalSubprocessSandbox()
    raise ValueError(f"Unknown sandbox backend: {chosen!r}")
