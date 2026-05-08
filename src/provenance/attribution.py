"""Attribution policies — decide which artifacts a tool use ingested.

The default `PathSubstringAttributor` works uniformly across shell-style
hosts (argv carries the path) and structured-tool hosts (args["path"]
carries it) by flattening the call into a string and substring-matching
against known artifact paths. Future LLM-based attributors implement the
same `Attributor` protocol.
"""

import json
from typing import Any, Protocol

from .nodes import ArtifactNode


class Attributor(Protocol):
    """Maps a tool use to the artifact node ids it ingested."""

    def attribute_tool_use(
        self,
        agent_id: str,
        tool_name: str,
        args: dict[str, Any],
        output: str,
        known_artifacts: dict[str, ArtifactNode],
    ) -> list[str]: ...


class PathSubstringAttributor:
    """Match known artifact paths as substrings of the call + output prefix."""

    def __init__(self, output_scan_chars: int = 1000) -> None:
        self._output_scan_chars = output_scan_chars

    def attribute_tool_use(
        self,
        agent_id: str,  # noqa: ARG002 — same scan applies to every agent
        tool_name: str,  # noqa: ARG002
        args: dict[str, Any],
        output: str,
        known_artifacts: dict[str, ArtifactNode],
    ) -> list[str]:
        if not known_artifacts:
            return []

        try:
            args_blob = json.dumps(args, default=str)
        except (TypeError, ValueError):
            args_blob = str(args)
        haystack = args_blob + "\n" + output[: self._output_scan_chars]

        ingested: list[str] = []
        for node_id, art in known_artifacts.items():
            if art.path and art.path in haystack:
                ingested.append(node_id)
        return ingested
