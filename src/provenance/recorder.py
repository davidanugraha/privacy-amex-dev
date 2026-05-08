"""Provenance recorder — the host-facing API for building the lineage graph.

A host framework integrates by calling three methods at the right points:
  - `record_artifact` when materializing a file/data into an agent's env
  - `record_tool_use` when an agent's tool/command call completes
  - `record_send` when dispatching a DM or channel/broadcast message

Two orthogonal modes:
  - storage axis (`store_content`): metadata-only (default) vs. full content
  - enforcement axis (`policy`): observer (default, no policy) vs. enforcer
"""

import hashlib
import uuid
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from .attribution import Attributor, PathSubstringAttributor
from .graph import ProvenanceGraph
from .nodes import ArtifactNode, MessageNode, ProvenanceEdge
from .policy import Policy, SendContext


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def _preview(s: str, n: int = 300) -> str:
    return s[:n]


class ProvenanceRecorder:
    """Stateful recorder. One instance per scenario / experiment run."""

    def __init__(
        self,
        *,
        store_content: bool = False,
        policy: Policy | None = None,
        attributor: Attributor | None = None,
    ) -> None:
        if policy is not None and "content" in policy.requires and not store_content:
            raise ValueError(
                f"Policy {type(policy).__name__} requires content but recorder "
                f"is in metadata-only mode. Pass store_content=True or use a "
                f"metadata-only policy."
            )
        self._store_content = store_content
        self._policy = policy
        self._graph = ProvenanceGraph()
        self._attributor: Attributor = attributor or PathSubstringAttributor()
        self._artifacts_by_agent: dict[str, dict[str, ArtifactNode]] = {}
        self._ingested_by_agent: dict[str, set[str]] = {}
        self._full_content: dict[str, str] = {}
        self._lock = Lock()

    @property
    def graph(self) -> ProvenanceGraph:
        return self._graph

    def get_full_content(self, node_id: str) -> str | None:
        """Return the full content for `node_id`, or None in metadata mode."""
        return self._full_content.get(node_id)

    def record_artifact(
        self,
        agent_id: str,
        path: str,
        content: str,
        declared_sensitivity: str | None = None,
    ) -> ArtifactNode:
        """Register a piece of source data available to `agent_id`. Idempotent."""
        node_id = f"artifact::{agent_id}::{path}"
        with self._lock:
            existing = self._artifacts_by_agent.get(agent_id, {}).get(node_id)
            if existing is not None:
                return existing
            node = ArtifactNode(
                node_id=node_id,
                agent_id=agent_id,
                path=path,
                content_hash=_hash(content),
                content_preview=_preview(content),
                declared_sensitivity=declared_sensitivity,
                created_at=datetime.now(UTC),
            )
            self._artifacts_by_agent.setdefault(agent_id, {})[node_id] = node
            if self._store_content:
                self._full_content[node_id] = content
        self._graph.add_node(node)
        return node

    def record_tool_use(
        self,
        agent_id: str,
        tool_name: str,
        args: dict[str, Any],
        output: str,
        exit_code: int = 0,  # noqa: ARG002 — reserved for future failure-aware attribution
    ) -> None:
        """Note a tool use; tag the agent with any artifacts it ingested."""
        with self._lock:
            known = dict(self._artifacts_by_agent.get(agent_id, {}))
        ingested_ids = self._attributor.attribute_tool_use(
            agent_id, tool_name, args, output, known,
        )
        if not ingested_ids:
            return
        with self._lock:
            self._ingested_by_agent.setdefault(agent_id, set()).update(ingested_ids)

    def record_send(
        self,
        *,
        sender_agent_id: str,
        recipient_agent_id: str | None,
        channel: str | None,
        content: str,
        delivered: bool,
        delivery_error: str | None = None,
    ) -> MessageNode:
        """Create a message node and link in every artifact the sender has ingested."""
        node_id = f"message::{uuid.uuid4()}"
        msg_node = MessageNode(
            node_id=node_id,
            sender_agent_id=sender_agent_id,
            recipient_agent_id=recipient_agent_id,
            channel=channel,
            content_hash=_hash(content),
            content_preview=_preview(content),
            attempted_at=datetime.now(UTC),
            delivered=delivered,
            delivery_error=delivery_error,
        )
        self._graph.add_node(msg_node)

        with self._lock:
            if self._store_content:
                self._full_content[node_id] = content
            sources = list(self._ingested_by_agent.get(sender_agent_id, ()))
        now = datetime.now(UTC)
        for src_id in sources:
            edge = ProvenanceEdge(
                edge_id=str(uuid.uuid4()),
                source_node_id=src_id,
                target_node_id=node_id,
                transforming_agent_id=sender_agent_id,
                edge_kind="read_then_send",
                created_at=now,
            )
            self._graph.add_edge(edge)
        return msg_node

    async def check_send(
        self,
        *,
        sender_agent_id: str,
        recipient_agent_id: str | None,
        channel: str | None,
        content: str,
    ) -> None:
        """Delegate to the wired policy; no-op when none is configured.

        Raises `PolicyViolation` (from the policy) to block a send.
        """
        if self._policy is None:
            return
        ctx = SendContext(
            sender_agent_id=sender_agent_id,
            recipient_agent_id=recipient_agent_id,
            channel=channel,
            content=content,
            recorder=self,
        )
        await self._policy.check_send(ctx)
