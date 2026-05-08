"""In-memory data-lineage graph.

Nodes are addressed by string id; edges are kept in a flat list with
adjacency indexes for fast upstream/downstream traversal. The graph is
purely a data structure — recording semantics live in `recorder.py`.
"""

from collections import deque
from threading import Lock
from typing import Any

from .nodes import Node, ProvenanceEdge


class ProvenanceGraph:
    """Thread-safe lineage graph keyed by node id."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, ProvenanceEdge] = {}
        self._incoming: dict[str, list[str]] = {}
        self._outgoing: dict[str, list[str]] = {}
        self._lock = Lock()

    def add_node(self, node: Node) -> None:
        with self._lock:
            self._nodes[node.node_id] = node
            self._incoming.setdefault(node.node_id, [])
            self._outgoing.setdefault(node.node_id, [])

    def add_edge(self, edge: ProvenanceEdge) -> None:
        with self._lock:
            self._edges[edge.edge_id] = edge
            self._outgoing.setdefault(edge.source_node_id, []).append(edge.edge_id)
            self._incoming.setdefault(edge.target_node_id, []).append(edge.edge_id)

    def get_node(self, node_id: str) -> Node | None:
        with self._lock:
            return self._nodes.get(node_id)

    def upstream(self, node_id: str) -> list[Node]:
        """All ancestors reachable by walking incoming edges (BFS)."""
        with self._lock:
            seen: set[str] = set()
            order: list[str] = []
            queue: deque[str] = deque([node_id])
            while queue:
                cur = queue.popleft()
                for edge_id in self._incoming.get(cur, []):
                    src = self._edges[edge_id].source_node_id
                    if src in seen:
                        continue
                    seen.add(src)
                    order.append(src)
                    queue.append(src)
            return [self._nodes[nid] for nid in order if nid in self._nodes]

    def nodes(self) -> list[Node]:
        with self._lock:
            return list(self._nodes.values())

    def edges(self) -> list[ProvenanceEdge]:
        with self._lock:
            return list(self._edges.values())

    def dump(self) -> dict[str, Any]:
        """JSON-serializable snapshot."""
        with self._lock:
            return {
                "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
                "edges": [e.model_dump(mode="json") for e in self._edges.values()],
            }
