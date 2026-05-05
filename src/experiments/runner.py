"""Experiment runner — loads a scenario, builds agents, runs, prints results.

Usage:
    python -m src.experiments.runner scenarios/healthcare_triage.yaml
    python -m src.experiments.runner scenarios/healthcare_triage.yaml --repeat 10
"""

import argparse
import asyncio
import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.privacy.agents.base import BaseAgent, BaseSimplePrivacyAgent
from src.privacy.agents.privacy_agent.agent import PrivacyAgent
from src.privacy.core import (
    ActionExecutionRequest,
    ActionExecutionResult,
    ChannelMessage,
    TextMessage,
)
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.memory import MemoryDatabase
from src.privacy.database.models import ActionRow, ActionRowData
from src.privacy.protocol.protocol import PrivacyProtocol
from src.privacy.sandbox import build_sandbox
from src.privacy.sandbox.base import Sandbox

from .evaluation import evaluate_criteria, score
from .report import generate_report
from .scenario import Scenario, ScenarioAgent, load_scenario


@dataclass
class ExperimentResult:
    """Container for post-experiment inspection."""

    database: BaseDatabaseController
    scenario: Scenario
    output_dir: Path


# --- agent execution primitives ----------------------------------------------

async def kickoff(
    database: BaseDatabaseController,
    content: str,
    *,
    channel: str = "general",
    from_agent_id: str = "system",
) -> None:
    """Inject a channel message before agents start stepping.

    Call this AFTER agents have registered (or after manually populating
    the general channel) but BEFORE `run_agents`. Every agent in the
    channel will see this message on their first `fetch_messages` call.
    """
    action = ChannelMessage(
        from_agent_id=from_agent_id,
        channel=channel,
        created_at=datetime.now(UTC),
        message=TextMessage(content=content),
    )
    request = ActionExecutionRequest(
        name=action.get_name(),
        parameters=action.model_dump(mode="json"),
    )
    result = ActionExecutionResult(
        content=action.model_dump(mode="json"),
        is_error=False,
        metadata={"status": "kickoff"},
    )
    await database.actions.create(
        ActionRow(
            id="",
            created_at=datetime.now(UTC),
            data=ActionRowData(
                agent_id=from_agent_id, request=request, result=result,
            ),
        )
    )


async def run_agents(
    *agents: BaseAgent[Any],
    terminator_agent_id: str | None = None,
    max_wall_seconds: int | None = None,
) -> None:
    """Run agents concurrently until termination condition is met.

    Termination:
      - If `terminator_agent_id` is set, the scenario ends as soon as that
        agent's `run()` task exits; remaining agents get `.shutdown()`.
      - Otherwise, the scenario ends when all agents' `run()` tasks exit.
      - If `max_wall_seconds` is set, it's a hard wall-clock cap — on
        expiry, all still-running agents get `.shutdown()`.
    """
    if not agents:
        return

    print(f"\nRunning {len(agents)} agents...")
    tasks = [asyncio.create_task(agent.run()) for agent in agents]
    task_by_agent: dict[str, asyncio.Task] = {a.id: t for a, t in zip(agents, tasks)}

    reason = "all agents completed"
    try:
        if terminator_agent_id is not None:
            term_task = task_by_agent.get(terminator_agent_id)
            if term_task is None:
                raise ValueError(
                    f"terminator_agent {terminator_agent_id!r} not found among agents"
                )
            done, _ = await asyncio.wait(
                [term_task], timeout=max_wall_seconds,
            )
            if not done:
                reason = f"wall-clock timeout ({max_wall_seconds}s)"
            else:
                reason = f"terminator agent {terminator_agent_id!r} exited"
        else:
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.ALL_COMPLETED,
                timeout=max_wall_seconds,
            )
            if len(done) < len(tasks):
                reason = f"wall-clock timeout ({max_wall_seconds}s)"
    finally:
        for a in agents:
            if not task_by_agent[a.id].done():
                a.shutdown()
        await asyncio.gather(*tasks, return_exceptions=True)

    print(f"Run ended: {reason}")


# --- single-scenario orchestration -------------------------------------------

async def run_scenario(
    scenario: Scenario,
    *,
    output_root: Path | str = "outputs",
) -> ExperimentResult:
    """Build agents from a scenario spec, kick off, run to completion."""
    db = MemoryDatabase()
    sandbox = build_sandbox(scenario.sandbox_backend)
    protocol = PrivacyProtocol(sandbox=sandbox)

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"Agents: {[a.id for a in scenario.agents]}")
    print(f"{'='*60}\n")

    agents: list[PrivacyAgent] = []
    all_ids = [sa.id for sa in scenario.agents]

    for sa in scenario.agents:
        agent = _build_agent(sa, protocol, db, all_ids, scenario)
        agents.append(agent)

    await kickoff(
        db,
        scenario.kickoff_message,
        from_agent_id=scenario.kickoff_from,
        channel=scenario.kickoff_channel,
    )

    await run_agents(
        *agents,
        terminator_agent_id=scenario.terminator_agent,
        max_wall_seconds=scenario.max_wall_seconds,
    )

    print(f"\n{'='*60}")
    print("Experiment complete. Printing action log summary...\n")
    await _print_action_summary(db)

    output_path = await dump_trajectories(agents, db, scenario, output_root=output_root)
    print(f"\nTrajectories saved to: {output_path}")

    await _run_verification(scenario, db, sandbox, output_path)

    report_path = generate_report(output_path)
    print(f"Report: {report_path}")

    return ExperimentResult(database=db, scenario=scenario, output_dir=output_path)


async def _run_verification(
    scenario: Scenario,
    db: BaseDatabaseController,
    sandbox: Sandbox,
    output_path: Path,
) -> None:
    if not scenario.success_criteria:
        return
    actions = await db.actions.get_all()
    results = await evaluate_criteria(scenario.success_criteria, actions, sandbox)
    total_score = score(results)

    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] ({r.kind}) {r.description} — {r.detail}")
    print(f"\nWeighted score: {total_score:.3f}")

    verification = {
        "score": total_score,
        "results": [r.to_dict() for r in results],
    }
    (output_path / "verification.json").write_text(json.dumps(verification, indent=2))


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
        max_tool_rounds=scenario.max_tool_rounds,
    )
    agent.sandbox_files = lambda: sa.sandbox_files  # type: ignore[assignment]
    return agent


async def _print_action_summary(db: BaseDatabaseController) -> None:
    rows = await db.actions.get_all()

    dm_count = 0
    channel_count = 0
    command_count = 0
    mark_done_count = 0

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
        elif name == "MarkDone":
            print(f"  [DONE] {agent_id}")
            mark_done_count += 1
        elif name == "FetchMessages":
            pass

    print(
        f"\nTotals: {dm_count} DMs, {channel_count} channel msgs, "
        f"{command_count} commands, {mark_done_count} mark_done"
    )


# --- per-run artifacts -------------------------------------------------------

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


async def run_k_times(scenario: Scenario, k: int) -> Path:
    """Run a scenario k times; return parent directory containing all runs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = Path("outputs") / f"{scenario.name}_{timestamp}_k{k}"
    parent.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    for i in range(k):
        print(f"\n{'#'*60}\n# Run {i+1}/{k}\n{'#'*60}")
        result = await run_scenario(scenario, output_root=parent)
        run_dirs.append(result.output_dir)

    agg_path = write_aggregate(parent, run_dirs)
    print(f"\nAggregate written to: {agg_path}")
    agg = json.loads(agg_path.read_text())
    print_aggregate(agg)
    return parent


# --- k-run aggregation -------------------------------------------------------

def aggregate(run_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate verification results across k runs.

    Returns:
        dict with per-criterion pass^k, leakage_rate (for no_leak criteria),
        and overall score stats.
    """
    per_criterion: dict[str, dict[str, Any]] = {}
    scores: list[float] = []

    for run_dir in run_dirs:
        vpath = run_dir / "verification.json"
        if not vpath.exists():
            continue
        data = json.loads(vpath.read_text())
        scores.append(float(data.get("score", 0.0)))
        for r in data.get("results", []):
            key = f"{r['kind']}::{r.get('description', '')}"
            slot = per_criterion.setdefault(key, {
                "kind": r["kind"],
                "description": r.get("description", ""),
                "runs": 0,
                "passes": 0,
                "fails": 0,
            })
            slot["runs"] += 1
            if r["passed"]:
                slot["passes"] += 1
            else:
                slot["fails"] += 1

    criteria_summary = []
    for slot in per_criterion.values():
        runs = slot["runs"] or 1
        entry = {
            "kind": slot["kind"],
            "description": slot["description"],
            "runs": slot["runs"],
            "pass_rate": slot["passes"] / runs,
            "pass_k": int(slot["passes"] == slot["runs"]),
        }
        if slot["kind"] == "no_leak":
            entry["leakage_rate"] = slot["fails"] / runs
        criteria_summary.append(entry)

    score_stats: dict[str, float] = {}
    if scores:
        score_stats = {
            "mean": statistics.fmean(scores),
            "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }

    return {
        "k": len(run_dirs),
        "scores": scores,
        "score_stats": score_stats,
        "criteria": criteria_summary,
    }


def write_aggregate(parent_dir: Path, run_dirs: list[Path]) -> Path:
    out = aggregate(run_dirs)
    path = parent_dir / "aggregate.json"
    path.write_text(json.dumps(out, indent=2))
    return path


def print_aggregate(agg: dict[str, Any]) -> None:
    k = agg["k"]
    print(f"\n{'='*60}")
    print(f"Aggregate over {k} run(s)")
    print(f"{'='*60}")
    ss = agg.get("score_stats") or {}
    if ss:
        print(f"Score: mean={ss['mean']:.3f} stdev={ss['stdev']:.3f} "
              f"min={ss['min']:.3f} max={ss['max']:.3f}")
    for c in agg.get("criteria", []):
        line = (f"  [{c['kind']}] {c['description']}: "
                f"pass_rate={c['pass_rate']:.2f} ({c['runs']} runs)")
        if "leakage_rate" in c:
            line += f"  leakage_rate={c['leakage_rate']:.2f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_path")
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Run the scenario N times and write aggregate.json (default: 1)",
    )
    args = parser.parse_args()

    scenario = load_scenario(args.scenario_path)

    if args.repeat <= 1:
        asyncio.run(run_scenario(scenario))
    else:
        asyncio.run(run_k_times(scenario, args.repeat))


if __name__ == "__main__":
    main()
