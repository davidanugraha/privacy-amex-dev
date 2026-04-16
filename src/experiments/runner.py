"""Experiment runner — loads a scenario, builds agents, runs, prints results.

Usage:
    python -m src.experiments.runner scenarios/healthcare_triage.yaml
"""

import asyncio
import sys
from dataclasses import dataclass
from typing import Any

from src.privacy.agents.privacy_agent.agent import PrivacyAgent
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.memory import MemoryDatabase
from src.privacy.protocol.protocol import PrivacyProtocol
from src.privacy.sandbox import build_sandbox

from .launcher import AgentLauncher
from .scenario import Scenario, ScenarioAgent, load_scenario


@dataclass
class ExperimentResult:
    """Container for post-experiment inspection."""

    database: BaseDatabaseController
    scenario: Scenario


async def run_scenario(scenario: Scenario) -> ExperimentResult:
    """Build agents from a scenario spec, kick off, run to completion."""
    db = MemoryDatabase()
    sandbox = build_sandbox(scenario.sandbox_backend)
    protocol = PrivacyProtocol(sandbox=sandbox)
    launcher = AgentLauncher(protocol, db)

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"Agents: {[a.id for a in scenario.agents]}")
    print(f"{'='*60}\n")

    agents: list[PrivacyAgent] = []
    all_ids = [sa.id for sa in scenario.agents]

    for sa in scenario.agents:
        # Override sandbox files via the agent's sandbox_files hook
        agent = _build_agent(sa, protocol, db, all_ids, scenario)
        agents.append(agent)

    await launcher.kickoff(
        scenario.kickoff_message,
        from_agent_id=scenario.kickoff_from,
        channel=scenario.kickoff_channel,
    )

    await launcher.run_agents(*agents)

    print(f"\n{'='*60}")
    print("Experiment complete. Printing action log summary...\n")
    _print_action_summary(db)

    return ExperimentResult(database=db, scenario=scenario)


def _build_agent(
    sa: ScenarioAgent,
    protocol: PrivacyProtocol,
    db: MemoryDatabase,
    all_ids: list[str],
    scenario: Scenario,
) -> PrivacyAgent:
    agent = PrivacyAgent(
        sa.profile,
        protocol=protocol,
        database=db,
        peer_ids=[aid for aid in all_ids if aid != sa.id],
        max_steps=scenario.max_steps,
        max_idle_steps=scenario.max_idle_steps,
    )

    # Override sandbox_files to return scenario-provisioned files
    original_sandbox_files = agent.sandbox_files
    agent.sandbox_files = lambda: sa.sandbox_files  # type: ignore[assignment]

    return agent


def _print_action_summary(db: BaseDatabaseController) -> None:
    """Print a human-readable summary of all actions that occurred."""
    import asyncio

    async def _get_actions():
        return await db.actions.get_all()

    rows = asyncio.get_event_loop().run_until_complete(_get_actions())

    dm_count = 0
    channel_count = 0
    command_count = 0
    other_count = 0

    for row in rows:
        name = row.data.request.name
        params = row.data.request.parameters
        agent_id = row.data.agent_id

        if name == "SendMessage":
            to = params.get("to_agent_id", "?")
            content = params.get("message", {}).get("content", "")[:80]
            print(f"  [DM] {agent_id} → {to}: {content}")
            dm_count += 1
        elif name == "ChannelMessage":
            channel = params.get("channel", "?")
            content = params.get("message", {}).get("content", "")[:80]
            print(f"  [#{channel}] {agent_id}: {content}")
            channel_count += 1
        elif name == "ExecuteCommand":
            cmd = params.get("command", [])
            print(f"  [CMD] {agent_id}: {' '.join(cmd)[:80]}")
            command_count += 1
        elif name == "FetchMessages":
            pass  # Skip fetch noise
        else:
            other_count += 1

    print(f"\nTotals: {dm_count} DMs, {channel_count} channel msgs, {command_count} commands")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.runner <scenario.yaml>")
        sys.exit(1)

    scenario = load_scenario(sys.argv[1])
    asyncio.run(run_scenario(scenario))


if __name__ == "__main__":
    main()
