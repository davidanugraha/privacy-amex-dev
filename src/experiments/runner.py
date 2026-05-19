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
    SendMessage,
    TextMessage,
)
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.memory import MemoryDatabase
from src.privacy.database.models import ActionRow, ActionRowData
from src.privacy.protocol.protocol import PrivacyProtocol
from src.privacy.sandbox import build_sandbox
from src.privacy.sandbox.base import Sandbox
from src.provenance import Policy, ProvenanceRecorder
from src.provenance.integrations.privacy_protocol import RecordingSandbox

from .completion import evaluate_completion
from .violations import evaluate_violations
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
    to_agent_id: str | None = None,
    recorder: ProvenanceRecorder | None = None,
) -> None:
    """Inject a kickoff message before agents start stepping.

    When `to_agent_id` is set, the kickoff is a DM to that single agent;
    otherwise it's a channel broadcast to `channel`. Call AFTER agents
    have been constructed but BEFORE `run_agents` — recipients pick the
    message up on their first `fetch_messages`.
    """
    if to_agent_id is not None:
        action = SendMessage(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            created_at=datetime.now(UTC),
            message=TextMessage(content=content),
        )
    else:
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
    if recorder is not None:
        recorder.record_send(
            sender_agent_id=from_agent_id,
            recipient_agent_id=to_agent_id,
            channel=None if to_agent_id is not None else channel,
            content=content,
            delivered=True,
        )


async def run_agents(
    *agents: BaseAgent[Any],
    max_wall_seconds: int | None = None,
) -> None:
    """Run agents concurrently until they've all completed or a cap fires.

    Termination:
      - The scenario ends when every agent's `run()` task exits (each agent
        decides for itself via `mark_done` / idle / step caps).
      - If `max_wall_seconds` is set, it's a hard wall-clock cap — on
        expiry, all still-running agents get `.shutdown()`.

    No single-agent "terminator" — eval criteria are the sole judge of
    whether the team accomplished the goal. See ../plans for the design
    discussion.
    """
    if not agents:
        return

    print(f"\nRunning {len(agents)} agents...")
    tasks = [asyncio.create_task(agent.run()) for agent in agents]
    task_by_agent: dict[str, asyncio.Task] = {a.id: t for a, t in zip(agents, tasks)}

    reason = "all agents completed"
    try:
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
    recorder = ProvenanceRecorder(
        store_content=scenario.provenance_store_content,
        policy=_build_policy(scenario.provenance_policy),
    )
    real_sandbox = build_sandbox(
        scenario.sandbox_backend,
        docker_mem_limit=scenario.docker_mem_limit,
        docker_cpus=scenario.docker_cpus,
        docker_network_disabled=scenario.docker_network_disabled,
    )
    sandbox = RecordingSandbox(real_sandbox, recorder)
    protocol = PrivacyProtocol(sandbox=sandbox, recorder=recorder)

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
        to_agent_id=scenario.kickoff_to,
        recorder=recorder,
    )

    await run_agents(
        *agents,
        max_wall_seconds=scenario.max_wall_seconds,
    )

    print(f"\n{'='*60}")
    print("Experiment complete. Printing action log summary...\n")
    await _print_action_summary(db)

    output_path = await dump_trajectories(agents, db, scenario, output_root=output_root)
    print(f"\nTrajectories saved to: {output_path}")

    (output_path / "provenance.json").write_text(
        json.dumps(recorder.graph.dump(), indent=2, default=str)
    )

    # Snapshot channel membership for the violation evaluator. Channels
    # are fixed at scenario start in the current substrate; a single
    # snapshot suffices.
    (output_path / "channels.json").write_text(
        json.dumps(
            {ch: sorted(members) for ch, members in protocol.channels.items()},
            indent=2,
        )
    )

    await _run_verification(scenario, db, sandbox, protocol, output_path)

    report_path = generate_report(output_path)
    print(f"Report: {report_path}")

    return ExperimentResult(database=db, scenario=scenario, output_dir=output_path)


async def _run_verification(
    scenario: Scenario,
    db: BaseDatabaseController,
    sandbox: Sandbox,
    protocol: PrivacyProtocol,
    output_path: Path,
) -> None:
    """Two-section eval: task completion + ground-truth violations.

    Completion: AND across deterministic checks + one optional LLM
    `goal_achieved`. Violations: recipient-indexed iteration over the
    action log, two detection paths (regex + LLM-judge) populating the
    same ViolationInstance schema. No weights; no compositional
    classification.
    """
    actions = await db.actions.get_all()
    channels = {ch: set(members) for ch, members in protocol.channels.items()}
    all_agent_ids = [sa.id for sa in scenario.agents]

    completion = await evaluate_completion(scenario.scenario_dir, sandbox)
    if not completion.results:
        completion = None

    violations = None
    if scenario.sensitive_claims:
        violations = await evaluate_violations(
            scenario.sensitive_claims,
            actions,
            channels,
            all_agent_ids,
            model=scenario.eval_model,
            enable_llm_judge=scenario.enable_llm_violation_judge,
        )

    if completion is None and violations is None:
        return

    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")

    if completion is not None:
        print("\nCompletion checks:")
        for r in completion.results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.description} — {r.detail}")
        print(f"  → all_passed: {completion.all_passed}")

    if violations is not None:
        agg = violations.aggregates
        print("\nPrivacy violations:")
        print(f"  edges:        {agg.total_violations} / {agg.max_possible_violations}"
              f"   direct/relay/illicit = "
              f"{agg.direct_violations}/{agg.relay_violations}/{agg.illicit_relay_violations}")
        print(f"  leak_rate:    {agg.leak_rate:.3f}"
              f"   substrate={agg.substrate_leak_rate:.3f}"
              f"   semantic={agg.semantic_leak_rate:.3f}")
        print(f"  disclosures:  {agg.total_disclosures} / {agg.possible_disclosures}"
              f"   direct/relay/illicit = "
              f"{agg.direct_disclosures}/{agg.relay_disclosures}/{agg.illicit_relay_disclosures}"
              f"   rate={agg.disclosure_rate:.3f}")
        if agg.judge_error_count:
            print(f"  judge_errors: {agg.judge_error_count} "
                  f"(claim, recipient) pair(s) errored — leak rate may be under-counted")
        for inst in violations.instances[:20]:
            print(
                f"    [{inst.detection}] claim={inst.claim_id} "
                f"{inst.sender} -> {inst.recipient} via {inst.via}"
                + (f"(#{inst.channel})" if inst.channel else "")
                + f": {inst.evidence[:80]}"
            )
        if len(violations.instances) > 20:
            print(f"    … {len(violations.instances) - 20} more")

    verification: dict[str, Any] = {}
    if completion is not None:
        verification["completion"] = completion.to_dict()
    if violations is not None:
        verification["violations"] = violations.to_dict()
    (output_path / "verification.json").write_text(json.dumps(verification, indent=2))


def _build_policy(name: str | None) -> Policy | None:
    """Resolve a policy name from scenario config to an instance.

    Returns None when no policy is configured. Raises on unknown names so
    a typo in the YAML fails loudly instead of silently running observer-mode.
    Concrete policies will register here as they're implemented.
    """
    if name is None:
        return None
    raise ValueError(
        f"Unknown provenance policy: {name!r}. No concrete policies are "
        f"wired yet — leave provenance_policy unset for now."
    )


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
        max_tool_rounds=scenario.max_tool_rounds,
        compact_at_tokens=scenario.compact_at_tokens,
        compact_keep_recent=scenario.compact_keep_recent,
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
            cmd = params.get("command", "")
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            print(f"  [CMD] {agent_id}: {cmd_str[:80]}")
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

    # Full log table — structured records from PrivacyLogger.
    # Intended for post-hoc analysis; not agent-readable (no protocol surface
    # exposes db.logs, so agents can't see other agents' traces).
    logs_file = output_dir / "logs.jsonl"
    log_rows = await database.logs.get_all()
    with open(logs_file, "w") as f:
        for row in log_rows:
            f.write(json.dumps(row.model_dump(mode="json"), default=str) + "\n")

    # Human-readable summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Scenario: {scenario.name}\n")
        f.write(f"Description: {scenario.description}\n")
        f.write(f"Agents: {[a.id for a in scenario.agents]}\n")
        f.write(f"Sandbox backend: {scenario.sandbox_backend}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Log entries: {len(log_rows)}\n")
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
                cmd = params.get("command", "")
                cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
                stdout = row.data.result.content
                if isinstance(stdout, dict):
                    stdout = stdout.get("stdout", "")[:80]
                else:
                    stdout = str(stdout)[:80]
                f.write(f"[CMD] {agent_id}: {cmd_str[:80]}\n")
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
    """Aggregate the split verification results across k runs.

    Two parallel streams:
      * `completion`: per-check pass rate + overall all_passed rate.
      * `violations`: per-run edge- and cell-level leak rates; per-claim/
        recipient/sender edge frequencies summed across runs; judge errors.
    """
    per_completion_check: dict[str, dict[str, Any]] = {}
    completion_all_passed: list[bool] = []

    # Edge-level rates.
    leak_rates: list[float] = []
    substrate_leak_rates: list[float] = []
    semantic_leak_rates: list[float] = []
    # Cell-level (per-disclosure) rate.
    disclosure_rates: list[float] = []
    # Cross-run counts (summed).
    direct_violations = relay_violations = illicit_relay_violations = 0
    direct_disclosures = relay_disclosures = illicit_relay_disclosures = 0
    total_disclosures_sum = possible_disclosures_sum = 0
    judge_error_count = 0
    per_claim_edges: dict[str, int] = {}
    per_recipient_edges: dict[str, int] = {}
    per_sender_edges: dict[str, int] = {}

    runs_with_completion = 0
    runs_with_violations = 0

    for run_dir in run_dirs:
        vpath = run_dir / "verification.json"
        if not vpath.exists():
            continue
        data = json.loads(vpath.read_text())

        comp = data.get("completion")
        if comp:
            runs_with_completion += 1
            completion_all_passed.append(bool(comp.get("all_passed")))
            for r in comp.get("results", []):
                key = r.get("name", "") or r.get("description", "")
                slot = per_completion_check.setdefault(key, {
                    "name": r.get("name", ""),
                    "description": r.get("description", ""),
                    "runs": 0,
                    "passes": 0,
                })
                slot["runs"] += 1
                if r.get("passed"):
                    slot["passes"] += 1

        viol = data.get("violations")
        if viol:
            runs_with_violations += 1
            agg = viol.get("aggregates") or {}
            leak_rates.append(float(agg.get("leak_rate", 0.0)))
            substrate_leak_rates.append(float(agg.get("substrate_leak_rate", 0.0)))
            semantic_leak_rates.append(float(agg.get("semantic_leak_rate", 0.0)))
            disclosure_rates.append(float(agg.get("disclosure_rate", 0.0)))
            direct_violations += int(agg.get("direct_violations", 0))
            relay_violations += int(agg.get("relay_violations", 0))
            illicit_relay_violations += int(agg.get("illicit_relay_violations", 0))
            direct_disclosures += int(agg.get("direct_disclosures", 0))
            relay_disclosures += int(agg.get("relay_disclosures", 0))
            illicit_relay_disclosures += int(agg.get("illicit_relay_disclosures", 0))
            total_disclosures_sum += int(agg.get("total_disclosures", 0))
            possible_disclosures_sum += int(agg.get("possible_disclosures", 0))
            judge_error_count += int(agg.get("judge_error_count", 0))
            for k, v in (agg.get("per_claim") or {}).items():
                per_claim_edges[k] = per_claim_edges.get(k, 0) + int(v)
            for k, v in (agg.get("per_recipient") or {}).items():
                per_recipient_edges[k] = per_recipient_edges.get(k, 0) + int(v)
            for k, v in (agg.get("per_sender") or {}).items():
                per_sender_edges[k] = per_sender_edges.get(k, 0) + int(v)

    completion_summary = []
    for slot in per_completion_check.values():
        runs = slot["runs"] or 1
        completion_summary.append({
            "name": slot["name"],
            "description": slot["description"],
            "runs": slot["runs"],
            "pass_rate": slot["passes"] / runs,
        })

    def _stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        return {
            "mean": statistics.fmean(xs),
            "stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
        }

    completion_all_passed_rate = (
        sum(1 for b in completion_all_passed if b) / len(completion_all_passed)
        if completion_all_passed else 0.0
    )

    return {
        "k": len(run_dirs),
        "completion": {
            "runs": runs_with_completion,
            "all_passed_rate": completion_all_passed_rate,
            "checks": completion_summary,
        },
        "violations": {
            "runs": runs_with_violations,
            "leak_rate": _stats(leak_rates),
            "substrate_leak_rate": _stats(substrate_leak_rates),
            "semantic_leak_rate": _stats(semantic_leak_rates),
            "disclosure_rate": _stats(disclosure_rates),
            "edge_type_totals": {
                "direct": direct_violations,
                "relay": relay_violations,
                "illicit_relay": illicit_relay_violations,
            },
            "disclosure_type_totals": {
                "direct": direct_disclosures,
                "relay": relay_disclosures,
                "illicit_relay": illicit_relay_disclosures,
            },
            "total_disclosures_sum": total_disclosures_sum,
            "possible_disclosures_sum": possible_disclosures_sum,
            "judge_error_count": judge_error_count,
            "per_claim_edges": dict(sorted(per_claim_edges.items())),
            "per_recipient_edges": dict(sorted(per_recipient_edges.items())),
            "per_sender_edges": dict(sorted(per_sender_edges.items())),
        },
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

    comp = agg.get("completion") or {}
    if comp.get("runs"):
        print(f"\nCompletion (runs={comp['runs']}): "
              f"all_passed_rate={comp.get('all_passed_rate', 0.0):.2f}")
        for c in comp.get("checks", []):
            label = c.get("name") or c.get("description") or "?"
            print(f"  {label}: pass_rate={c['pass_rate']:.2f} ({c['runs']} runs)")

    viol = agg.get("violations") or {}
    if viol.get("runs"):
        print(f"\nViolations (runs={viol['runs']}):")
        for name in ("leak_rate", "substrate_leak_rate", "semantic_leak_rate", "disclosure_rate"):
            ss = viol.get(name) or {}
            if ss:
                print(f"  {name}: mean={ss['mean']:.3f} "
                      f"min={ss['min']:.3f} max={ss['max']:.3f}")
        et = viol.get("edge_type_totals") or {}
        if et:
            print(f"  edge_type_totals: direct={et.get('direct', 0)} "
                  f"relay={et.get('relay', 0)} illicit_relay={et.get('illicit_relay', 0)}")
        dt = viol.get("disclosure_type_totals") or {}
        if dt:
            print(f"  disclosure_type_totals: direct={dt.get('direct', 0)} "
                  f"relay={dt.get('relay', 0)} illicit_relay={dt.get('illicit_relay', 0)} "
                  f"(of {viol.get('total_disclosures_sum', 0)} disclosures across "
                  f"{viol.get('possible_disclosures_sum', 0)} possible)")
        if viol.get("judge_error_count"):
            print(f"  judge_error_count: {viol['judge_error_count']}")
        for label, key in (
            ("per_claim", "per_claim_edges"),
            ("per_recipient", "per_recipient_edges"),
            ("per_sender", "per_sender_edges"),
        ):
            d = viol.get(key) or {}
            if d:
                print(f"  {label}: " + ", ".join(f"{k}={v}" for k, v in d.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_path")
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Run the scenario N times and write aggregate.json (default: 1)",
    )
    parser.add_argument(
        "--use-guide", action="store_true",
        help="Inject scenario_guide.yaml privacy guidance into each agent's system_prompt (R1 arm).",
    )
    args = parser.parse_args()

    scenario = load_scenario(args.scenario_path, use_guide=args.use_guide)

    if args.repeat <= 1:
        asyncio.run(run_scenario(scenario))
    else:
        asyncio.run(run_k_times(scenario, args.repeat))


if __name__ == "__main__":
    main()
