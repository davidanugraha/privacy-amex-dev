"""Provenance: a framework-agnostic data-lineage graph for multi-agent systems.

Public surface:
  - `ProvenanceRecorder`: host-facing API. Call `record_artifact`,
    `record_tool_use`, `record_send` from your dispatch layer.
  - `ProvenanceGraph`: queryable lineage graph. Use `upstream(node_id)`
    for ancestor walks and `dump()` for serialization.
  - `Policy` / `SendContext` / `PolicyViolation` / `ChainedPolicy`: the
    enforcement-axis hook surface.
  - Node and edge types for downstream analyzers / serialization.

Host integrations (e.g. for this repo's `PrivacyProtocol`) live under
`provenance.integrations.*` and are not part of the core surface.
"""

from .attribution import Attributor, PathSubstringAttributor
from .graph import ProvenanceGraph
from .nodes import ArtifactNode, MessageNode, Node, ProvenanceEdge
from .policy import ChainedPolicy, Policy, PolicyViolation, SendContext
from .recorder import ProvenanceRecorder

__all__ = [
    "ArtifactNode",
    "Attributor",
    "ChainedPolicy",
    "MessageNode",
    "Node",
    "PathSubstringAttributor",
    "Policy",
    "PolicyViolation",
    "ProvenanceEdge",
    "ProvenanceGraph",
    "ProvenanceRecorder",
    "SendContext",
]
