"""Provenance graph node types.

Nodes represent information-bearing entities. Agents are NOT nodes — they
sit on edges as the transformation that produced a target node from its
sources. New node kinds (e.g. intermediate transformation nodes) can be
added later as discriminated members of `Node` without changing existing
records.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class ArtifactNode(BaseModel):
    """A piece of source data materialized into an agent's environment."""

    kind: Literal["artifact"] = "artifact"
    node_id: str
    agent_id: str
    path: str
    content_hash: str
    content_preview: str
    declared_sensitivity: str | None = None
    created_at: datetime


class MessageNode(BaseModel):
    """A pre-send message — the transformed state at the dispatch boundary."""

    kind: Literal["message"] = "message"
    node_id: str
    sender_agent_id: str
    recipient_agent_id: str | None
    channel: str | None
    content_hash: str
    content_preview: str
    attempted_at: datetime
    delivered: bool
    delivery_error: str | None = None


Node = Annotated[ArtifactNode | MessageNode, Field(discriminator="kind")]
NodeAdapter: TypeAdapter[Node] = TypeAdapter(Node)


class ProvenanceEdge(BaseModel):
    """A directed source→target flow with the agent that produced it."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    transforming_agent_id: str
    edge_kind: Literal["read_then_send"] = "read_then_send"
    created_at: datetime
