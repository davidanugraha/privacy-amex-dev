"""Dump per-agent trajectories and action logs to disk after an experiment."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.privacy.agents.base import BaseSimplePrivacyAgent
from src.privacy.database.base import BaseDatabaseController

from .scenario import Scenario


async def dump_trajectories(
    agents: list[BaseSimplePrivacyAgent],
    database: BaseDatabaseController,
    scenario: Scenario,
    output_root: Path | str = "outputs",
) -> Path:
    """Write per-agent JSONL trajectories + action log to a timestamped folder.

    Returns the output directory path.
    """
    output_root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{scenario.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-agent trajectory files
    for agent in agents:
        agent_file = output_dir / f"{agent.id}.jsonl"
        with open(agent_file, "w") as f:
            # Write agent metadata as first line
            meta = {
                "_type": "agent_metadata",
                "agent_id": agent.id,
                "profile": agent.profile.model_dump(mode="json"),
                "total_messages": len(agent._messages),
            }
            f.write(json.dumps(meta) + "\n")

            # Write each conversation message
            for msg in agent._messages:
                f.write(json.dumps(msg, default=str) + "\n")

    # Full action log
    actions_file = output_dir / "actions.jsonl"
    rows = await database.actions.get_all()
    with open(actions_file, "w") as f:
        for row in rows:
            entry = {
                "id": row.id,
                "index": row.index,
                "created_at": str(row.created_at),
                "agent_id": row.data.agent_id,
                "request_name": row.data.request.name,
                "request_parameters": row.data.request.parameters,
                "result_content": row.data.result.content,
                "result_is_error": row.data.result.is_error,
            }
            f.write(json.dumps(entry, default=str) + "\n")

    # Human-readable summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Scenario: {scenario.name}\n")
        f.write(f"Description: {scenario.description}\n")
        f.write(f"Agents: {[a.id for a in scenario.agents]}\n")
        f.write(f"Sandbox backend: {scenario.sandbox_backend}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\n{'='*60}\nAction Log\n{'='*60}\n\n")

        dm_count = 0
        channel_count = 0
        command_count = 0

        for row in rows:
            name = row.data.request.name
            params = row.data.request.parameters
            agent_id = row.data.agent_id

            if name == "SendMessage":
                to = params.get("to_agent_id", "?")
                content = params.get("message", {}).get("content", "")[:120]
                f.write(f"[DM] {agent_id} -> {to}: {content}\n")
                dm_count += 1
            elif name == "ChannelMessage":
                channel = params.get("channel", "?")
                content = params.get("message", {}).get("content", "")[:120]
                f.write(f"[#{channel}] {agent_id}: {content}\n")
                channel_count += 1
            elif name == "ExecuteCommand":
                cmd = params.get("command", [])
                stdout = row.data.result.content
                if isinstance(stdout, dict):
                    stdout = stdout.get("stdout", "")[:80]
                else:
                    stdout = str(stdout)[:80]
                f.write(f"[CMD] {agent_id}: {' '.join(cmd)[:80]}\n")
                f.write(f"      stdout: {stdout}\n")
                command_count += 1
            elif name == "FetchMessages":
                pass

        f.write(f"\nTotals: {dm_count} DMs, {channel_count} channel msgs, {command_count} commands\n")

    return output_dir
