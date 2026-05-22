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
from typing import Any, Callable

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
from src.privacy.logger import PrivacyLogger
from src.privacy.protocol.protocol import PrivacyProtocol
from src.privacy.sandbox import build_sandbox
from src.privacy.sandbox.base import Sandbox
from src.provenance import Policy, ProvenanceRecorder
from src.provenance.integrations.privacy_protocol import RecordingSandbox

from .completion import evaluate_completion
from .violations import evaluate_violations
from .report import generate_report
from .scenario import Scenario, ScenarioAgent, load_scenario


class LiveStreamWriter:
    """Per-run JSONL streamer for actions + per-agent LLM messages.

    Flushes after every write so external `tail -f` processes see lines
    immediately (Python's default block buffering would hold them back).
    Output shape matches the end-of-run aggregate writes in `dump_trajectories`
    so downstream tooling sees identical JSONL whether streamed or dumped.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._actions_fp = open(self.output_dir / "actions.jsonl", "w")
        self._agent_fps: dict[str, Any] = {}

    def open_agent(self, agent_id: str, profile_dict: dict[str, Any]) -> None:
        fp = open(self.output_dir / f"{agent_id}.jsonl", "w")
        meta = {"_type": "agent_metadata", "agent_id": agent_id, "profile": profile_dict}
        fp.write(json.dumps(meta) + "\n")
        fp.flush()
        self._agent_fps[agent_id] = fp

    def write_action(self, row: ActionRow) -> None:
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
        self._actions_fp.write(json.dumps(entry, default=str) + "\n")
        self._actions_fp.flush()

    def write_agent_message(self, agent_id: str, message: dict[str, Any]) -> None:
        fp = self._agent_fps.get(agent_id)
        if fp is None:
            return
        fp.write(json.dumps(message, default=str) + "\n")
        fp.flush()

    def close(self) -> None:
        try:
            self._actions_fp.close()
        except Exception:
            pass
        for fp in self._agent_fps.values():
            try:
                fp.close()
            except Exception:
                pass
        self._agent_fps.clear()


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
    action_sink: Callable[[ActionRow], None] | None = None,
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
    row = ActionRow(
        id="",
        created_at=datetime.now(UTC),
        data=ActionRowData(
            agent_id=from_agent_id, request=request, result=result,
        ),
    )
    await database.actions.create(row)
    if action_sink is not None:
        try:
            action_sink(row)
        except Exception:
            pass
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
    seed: int = 0,
) -> ExperimentResult:
    """Build agents from a scenario spec, kick off, run to completion."""
    output_root = Path(output_root)
    date_stamp = datetime.now().strftime("%Y%m%d")
    output_path = output_root / scenario.name / f"{date_stamp}_seed{seed}"
    live_writer = LiveStreamWriter(output_path)

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
    protocol = PrivacyProtocol(
        sandbox=sandbox,
        recorder=recorder,
        action_sink=live_writer.write_action,
    )

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"Agents: {[a.id for a in scenario.agents]}")
    print(f"Output dir: {output_path}")
    print(f"{'='*60}\n")

    agents: list[PrivacyAgent] = []
    all_ids = [sa.id for sa in scenario.agents]

    for sa in scenario.agents:
        agent = _build_agent(sa, protocol, db, all_ids, scenario, live_writer)
        agents.append(agent)
        live_writer.open_agent(agent.id, agent.profile.model_dump(mode="json"))

    await kickoff(
        db,
        scenario.kickoff_message,
        from_agent_id=scenario.kickoff_from,
        channel=scenario.kickoff_channel,
        to_agent_id=scenario.kickoff_to,
        recorder=recorder,
        action_sink=live_writer.write_action,
    )

    try:
        await run_agents(
            *agents,
            max_wall_seconds=scenario.max_wall_seconds,
        )

        print(f"\n{'='*60}")
        print("Experiment complete. Printing action log summary...\n")
        await _print_action_summary(db)

        await dump_trajectories(agents, db, scenario, output_path)
    finally:
        live_writer.close()
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
    action log, judge (single-shot LLM or tool-using agent per
    `scenario.judge_mode`) over each (claim, unauthorized-recipient) pair.
    No compositional classification.
    """
    actions = await db.actions.get_all()
    channels = {ch: set(members) for ch, members in protocol.channels.items()}
    all_agent_ids = [sa.id for sa in scenario.agents]

    # Logger tied to the same in-memory DB so judge / completion LLM calls land
    # in logs.jsonl alongside agent-turn calls. Phase metadata distinguishes them.
    eval_logger = PrivacyLogger("eval", db)

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
            judge_mode=scenario.judge_mode,
            max_rounds=scenario.judge_max_rounds,
            max_pairs=scenario.judge_max_pairs,
            logger=eval_logger,
        )

    await eval_logger.flush()

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
        print(f"  leak_rate:    {agg.leak_rate:.3f}")
        print(f"  disclosures:  {agg.total_disclosures} / {agg.possible_disclosures}"
              f"   direct/relay/illicit = "
              f"{agg.direct_disclosures}/{agg.relay_disclosures}/{agg.illicit_relay_disclosures}"
              f"   rate={agg.disclosure_rate:.3f}")
        if agg.judge_error_count:
            print(f"  judge_errors: {agg.judge_error_count} "
                  f"(claim, recipient) pair(s) errored or exhausted budget — leak rate may be under-counted")
        for inst in violations.instances[:20]:
            print(
                f"    claim={inst.claim_id} "
                f"{inst.sender} -> {inst.recipient} via {inst.via}"
                + (f"(#{inst.channel})" if inst.channel else "")
                + f": {inst.evidence[:80]}"
            )
        if len(violations.instances) > 20:
            print(f"    … {len(violations.instances) - 20} more")

    verification: dict[str, Any] = {}

    # Headline summary at the top — the three granularity-invariant leak
    # rates plus completion pass/fail, surfaced for at-a-glance reading
    # before the detailed `violations.aggregates` / `violations.verdicts`
    # blocks below.
    summary: dict[str, Any] = {}
    if completion is not None:
        summary["completion_passed"] = bool(completion.all_passed)
    if violations is not None:
        agg = violations.aggregates
        summary["leak_metrics"] = {
            "edge_leak_rate": agg.leak_rate,
            "edge_violations": f"{agg.total_violations} / {agg.max_possible_violations}",
            "claim_disclosure_rate": agg.disclosure_rate,
            "claim_disclosures": f"{agg.total_disclosures} / {agg.possible_disclosures}",
            "doc_disclosure_rate": agg.doc_disclosure_rate,
            "doc_disclosures": f"{agg.total_doc_disclosures} / {agg.possible_doc_disclosures}",
            "doc_edge_leak_rate": agg.doc_edge_leak_rate,
            "doc_edges": f"{agg.total_doc_edges_leaked} / {agg.possible_doc_edges}",
            "recipient_disclosure_rate": agg.recipient_disclosure_rate,
            "recipients_polluted": f"{agg.total_recipient_disclosures} / {agg.possible_recipient_disclosures}",
            "pair_disclosure_rate": agg.pair_disclosure_rate,
            "pairs_leaked": f"{agg.total_pair_disclosures} / {agg.possible_pair_disclosures}",
            "judge_error_count": agg.judge_error_count,
        }

    # Token usage rollup — read from db.logs so it includes both agent-turn
    # calls (phase: scenario, identified by metadata.agent_id) and eval calls
    # (phase: judge | completion, set explicitly via log_metadata).
    usage = await _compute_usage_summary(db)
    summary["usage"] = usage

    if summary:
        verification["summary"] = summary

    if completion is not None:
        verification["completion"] = completion.to_dict()
    if violations is not None:
        verification["violations"] = violations.to_dict()
    verification["usage"] = usage
    (output_path / "verification.json").write_text(json.dumps(verification, indent=2))


def _empty_phase_bucket() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "errors": 0,
    }


async def _compute_usage_summary(db: BaseDatabaseController) -> dict[str, Any]:
    """Roll up token usage from db.logs by phase + by agent.

    Phases:
      - "scenario": agent-turn LLM calls (metadata.agent_id set, no metadata.phase)
      - "judge":    sensitive-claims leak verdicts (metadata.phase == "judge")
      - "completion": completion LLM judge calls (metadata.phase == "completion")
    """
    by_phase: dict[str, dict[str, int]] = {
        "scenario": _empty_phase_bucket(),
        "judge": _empty_phase_bucket(),
        "completion": _empty_phase_bucket(),
    }
    by_agent: dict[str, dict[str, int]] = {}
    totals = _empty_phase_bucket()

    rows = await db.logs.get_all()
    for row in rows:
        log = row.data
        inner_raw = getattr(log, "data", None)
        if hasattr(inner_raw, "model_dump"):
            inner = inner_raw.model_dump()
        elif isinstance(inner_raw, dict):
            inner = inner_raw
        else:
            continue
        if inner.get("type") != "llm_call":
            continue

        metadata = getattr(log, "metadata", None) or {}
        phase = metadata.get("phase") or "scenario"
        if phase not in by_phase:
            by_phase[phase] = _empty_phase_bucket()

        success = bool(inner.get("success", True))
        input_tokens = int(inner.get("input_tokens", 0) or 0)
        output_tokens = int(inner.get("output_tokens", 0) or 0)
        total_tokens = int(inner.get("token_count", 0) or 0) or (input_tokens + output_tokens)

        bucket = by_phase[phase]
        bucket["calls"] += 1
        if not success:
            bucket["errors"] += 1
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["total_tokens"] += total_tokens

        totals["calls"] += 1
        if not success:
            totals["errors"] += 1
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["total_tokens"] += total_tokens

        agent_id = metadata.get("agent_id")
        if agent_id:
            a = by_agent.setdefault(agent_id, _empty_phase_bucket())
            a["calls"] += 1
            if not success:
                a["errors"] += 1
            a["input_tokens"] += input_tokens
            a["output_tokens"] += output_tokens
            a["total_tokens"] += total_tokens

    return {"by_phase": by_phase, "by_agent": by_agent, "totals": totals}


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
    live_writer: LiveStreamWriter | None = None,
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
        message_sink=(live_writer.write_agent_message if live_writer is not None else None),
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
    output_dir: Path,
) -> Path:
    """Write end-of-run aggregate files (`logs.jsonl`, `summary.txt`).

    `actions.jsonl` and per-agent `<agent_id>.jsonl` are written incrementally
    by `LiveStreamWriter` during the run, so they're skipped here.
    """
    output_dir = Path(output_dir)
    rows = await database.actions.get_all()

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


async def run_k_times(scenario: Scenario, k: int, *, seed: int = 0) -> Path:
    """Run a scenario k times; return parent directory containing all runs.

    Per-run seeds are derived as seed, seed+1, ..., seed+k-1 so each child
    run gets a distinct label in `outputs/<scenario>/<timestamp>_seedN/`.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = Path("outputs") / scenario.name / f"{timestamp}_k{k}"
    parent.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    for i in range(k):
        print(f"\n{'#'*60}\n# Run {i+1}/{k}\n{'#'*60}")
        result = await run_scenario(scenario, output_root=parent, seed=seed + i)
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

    # Per-phase token totals across the k runs.
    phase_names = ["scenario", "judge", "completion"]
    usage_per_phase: dict[str, dict[str, list[int] | int]] = {
        p: {
            "totals": [],          # list of per-run total_tokens
            "input_sum": 0,
            "output_sum": 0,
            "total_sum": 0,
            "calls_sum": 0,
            "errors_sum": 0,
        }
        for p in phase_names
    }
    usage_totals = {"input_sum": 0, "output_sum": 0, "total_sum": 0, "calls_sum": 0}
    runs_with_usage = 0

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

        usage = data.get("usage")
        if usage:
            runs_with_usage += 1
            for p in phase_names:
                bucket = (usage.get("by_phase") or {}).get(p) or {}
                slot = usage_per_phase[p]
                slot["totals"].append(int(bucket.get("total_tokens", 0)))
                slot["input_sum"] += int(bucket.get("input_tokens", 0))
                slot["output_sum"] += int(bucket.get("output_tokens", 0))
                slot["total_sum"] += int(bucket.get("total_tokens", 0))
                slot["calls_sum"] += int(bucket.get("calls", 0))
                slot["errors_sum"] += int(bucket.get("errors", 0))
            t = usage.get("totals") or {}
            usage_totals["input_sum"] += int(t.get("input_tokens", 0))
            usage_totals["output_sum"] += int(t.get("output_tokens", 0))
            usage_totals["total_sum"] += int(t.get("total_tokens", 0))
            usage_totals["calls_sum"] += int(t.get("calls", 0))

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

    usage_block: dict[str, Any] = {"runs": runs_with_usage, "by_phase": {}}
    for p in phase_names:
        slot = usage_per_phase[p]
        totals_list = [float(x) for x in slot["totals"]]
        usage_block["by_phase"][p] = {
            "total_tokens_mean": statistics.fmean(totals_list) if totals_list else 0.0,
            "total_tokens_stdev": statistics.pstdev(totals_list) if len(totals_list) > 1 else 0.0,
            "total_tokens_sum": slot["total_sum"],
            "input_tokens_sum": slot["input_sum"],
            "output_tokens_sum": slot["output_sum"],
            "calls_sum": slot["calls_sum"],
            "errors_sum": slot["errors_sum"],
        }
    usage_block["totals"] = {
        "total_tokens_sum": usage_totals["total_sum"],
        "input_tokens_sum": usage_totals["input_sum"],
        "output_tokens_sum": usage_totals["output_sum"],
        "calls_sum": usage_totals["calls_sum"],
    }

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
        "usage": usage_block,
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
        for name in ("leak_rate", "disclosure_rate"):
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

    usage = agg.get("usage") or {}
    if usage.get("runs"):
        print(f"\nToken usage (runs={usage['runs']}):")
        for p in ("scenario", "judge", "completion"):
            ub = (usage.get("by_phase") or {}).get(p) or {}
            if not ub.get("calls_sum"):
                continue
            print(f"  {p:<10s} calls={ub['calls_sum']:<5d} "
                  f"in={ub['input_tokens_sum']:<8d} out={ub['output_tokens_sum']:<8d} "
                  f"total={ub['total_tokens_sum']:<9d} "
                  f"per-run mean={ub['total_tokens_mean']:.0f} stdev={ub['total_tokens_stdev']:.0f}")
        ut = usage.get("totals") or {}
        if ut.get("calls_sum"):
            print(f"  {'TOTAL':<10s} calls={ut['calls_sum']:<5d} "
                  f"in={ut['input_tokens_sum']:<8d} out={ut['output_tokens_sum']:<8d} "
                  f"total={ut['total_tokens_sum']:<9d}")


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
    parser.add_argument(
        "--judge-mode", choices=("off", "llm", "agent"), default=None,
        help="Override scenario.judge_mode (default: scenario file value).",
    )
    parser.add_argument(
        "--judge-max-rounds", type=int, default=None,
        help="Override scenario.judge_max_rounds (agent-judge turn budget per pair).",
    )
    parser.add_argument(
        "--judge-max-pairs", type=int, default=None,
        help="Override scenario.judge_max_pairs (random-sample N pairs for dev).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Run seed label, encoded into the output folder (default: 0). "
             "Currently a label only; not wired into LLM sampling.",
    )
    args = parser.parse_args()

    scenario = load_scenario(args.scenario_path, use_guide=args.use_guide)
    if args.judge_mode is not None:
        scenario.judge_mode = args.judge_mode
    if args.judge_max_rounds is not None:
        scenario.judge_max_rounds = args.judge_max_rounds
    if args.judge_max_pairs is not None:
        scenario.judge_max_pairs = args.judge_max_pairs

    if args.repeat <= 1:
        asyncio.run(run_scenario(scenario, seed=args.seed))
    else:
        asyncio.run(run_k_times(scenario, args.repeat, seed=args.seed))


if __name__ == "__main__":
    main()
