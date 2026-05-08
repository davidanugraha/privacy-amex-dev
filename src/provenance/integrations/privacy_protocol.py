"""Reference integration for this repo's `PrivacyProtocol` and `Sandbox`.

`RecordingSandbox` wraps a real `Sandbox` and reports every file
materialized at setup to the recorder. Wiring on the protocol side is
done by passing `recorder=` to `PrivacyProtocol(...)`; the protocol's
`_dispatch` calls the three hook methods directly.
"""

from src.privacy.core import ExecuteCommandResult
from src.privacy.sandbox.base import Sandbox

from ..recorder import ProvenanceRecorder


class RecordingSandbox:
    """Sandbox decorator that reports artifact materialization to a recorder."""

    def __init__(self, inner: Sandbox, recorder: ProvenanceRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    async def setup(
        self,
        agent_id: str,
        *,
        image: str = "",
        files: dict[str, str] | None = None,
    ) -> None:
        for path, content in (files or {}).items():
            self._recorder.record_artifact(agent_id, path, content)
        await self._inner.setup(agent_id, image=image, files=files)

    async def run(
        self,
        *,
        agent_id: str,
        command: list[str],
        stdin: str | None,
        timeout: int,
    ) -> ExecuteCommandResult:
        return await self._inner.run(
            agent_id=agent_id, command=command, stdin=stdin, timeout=timeout,
        )

    async def teardown(self, agent_id: str) -> None:
        await self._inner.teardown(agent_id)

    async def read_file(self, agent_id: str, path: str) -> str | None:
        return await self._inner.read_file(agent_id, path)
