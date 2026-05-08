"""Policy hook surface for `ProvenanceRecorder`.

A `Policy` plugs into the recorder and is consulted on every dispatch via
`check_send`. Policies declare their data requirements via `requires`
(subset of {"metadata", "content"}); the recorder fails fast at construction
if a content-needing policy is paired with a metadata-only recorder.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .recorder import ProvenanceRecorder


class PolicyViolation(Exception):
    """Raised by an enforcing policy to veto a send.

    The protocol layer catches this at dispatch and converts it to an
    error result; the recorder still logs the attempted send with
    `delivered=False`.
    """


@dataclass(frozen=True)
class SendContext:
    """Argument bundle passed to `Policy.check_send`."""

    sender_agent_id: str
    recipient_agent_id: str | None
    channel: str | None
    content: str
    recorder: "ProvenanceRecorder"


@runtime_checkable
class Policy(Protocol):
    """A pre-send policy. Raise `PolicyViolation` to block a send."""

    requires: frozenset[str]

    async def check_send(self, ctx: SendContext) -> None: ...


class ChainedPolicy:
    """Compose multiple policies; first violation short-circuits."""

    def __init__(self, policies: list[Policy]) -> None:
        self._policies = list(policies)
        self.requires = frozenset().union(*(p.requires for p in self._policies))

    async def check_send(self, ctx: SendContext) -> None:
        for p in self._policies:
            await p.check_send(ctx)
