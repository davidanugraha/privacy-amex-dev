"""Scenario data model and YAML loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from src.privacy.agents.privacy_agent.models import PrivacyAgentProfile

from .violations import SensitiveClaim


@dataclass
class ScenarioAgent:
    """An agent definition within a scenario."""

    id: str
    profile: PrivacyAgentProfile
    sandbox_files: dict[str, str] = field(default_factory=dict)


@dataclass
class Scenario:
    """A complete experiment specification loaded from YAML."""

    name: str
    description: str
    agents: list[ScenarioAgent]
    kickoff_message: str
    kickoff_from: str = "system"
    kickoff_channel: str = "general"
    kickoff_to: str | None = None
    max_steps: int = 20
    max_tool_rounds: int = 10
    max_wall_seconds: int | None = None
    sandbox_backend: Literal["docker", "local"] = "local"
    # Docker-only knobs. Defaults align with typical agentic-harness deployment:
    # 4 GB memory, no CPU pinning, network disabled (RQ1 substrate claim).
    docker_mem_limit: str = "4g"
    docker_cpus: float | None = None
    docker_network_disabled: bool = True
    # Memory compaction: opt-in agent-side history summarization.
    # `compact_at_tokens=None` → no compaction (today's behavior, verbatim history).
    # When set, the agent will summarize older turns before each step's LLM call
    # if its conversation history exceeds the threshold. Matches Claude Code /
    # AutoGen / LangGraph patterns for context-window management.
    compact_at_tokens: int | None = None
    compact_keep_recent: int = 3
    # Completion checks live in `scenario_dir/completion.py` and are
    # auto-discovered by `evaluate_completion`. `scenario_dir` is set by
    # `load_scenario` from the YAML path.
    scenario_dir: Path = field(default_factory=lambda: Path("."))
    # Violation eval reads `sensitive_claims` directly. Each claim's
    # `detectors` + `authorized_for` is the ground truth the violation
    # evaluator iterates over.
    goal: str | None = None
    sensitive_claims: list[SensitiveClaim] = field(default_factory=list)
    # Optional model override for LLM-judged eval (goal_achieved + the
    # LLM-judge path in evaluate_violations). When None, the LLM client
    # uses its environment-configured default.
    eval_model: str | None = None
    # Toggle the LLM-judge path of evaluate_violations. The substrate
    # (regex) path always runs.
    enable_llm_violation_judge: bool = True
    expected_failure_modes: list[dict[str, Any]] = field(default_factory=list)
    provenance_store_content: bool = False
    provenance_policy: str | None = None


def _resolve_sandbox_files(
    raw_files: dict[str, str], scenario_dir: Path
) -> dict[str, str]:
    """Resolve sandbox file values.

    If a value starts with ``file:`` the remainder is treated as a path
    relative to the scenario YAML and its contents are read from disk.
    Otherwise the value is used as-is (inline content).
    """
    resolved: dict[str, str] = {}
    for name, value in raw_files.items():
        if value.strip().startswith("file:"):
            rel = value.strip().removeprefix("file:").strip()
            file_path = scenario_dir / rel
            resolved[name] = file_path.read_text()
        else:
            resolved[name] = value
    return resolved


def _apply_guide(
    agents: list[ScenarioAgent], scenario_dir: Path
) -> None:
    """Append per-agent privacy guidance to each system_prompt (R1 arm).

    Reads `scenario_guide.yaml` from `scenario_dir`. The file is a
    mapping with one `agents:` list whose entries are
    `{id: <agent_id>, guidance: <multi-line text>}`. Each matching
    agent's `system_prompt` is extended with the guidance text
    under a labelled section so downstream analysis can distinguish
    guided runs (R1) from baseline runs (R0).
    """
    guide_path = scenario_dir / "scenario_guide.yaml"
    if not guide_path.exists():
        raise FileNotFoundError(
            f"use_guide=True but no scenario_guide.yaml found at {guide_path}"
        )
    with open(guide_path) as f:
        guide_raw = yaml.safe_load(f) or {}
    guide_by_id = {
        entry["id"]: entry["guidance"]
        for entry in guide_raw.get("agents", [])
    }
    for sa in agents:
        text = guide_by_id.get(sa.id)
        if not text:
            continue
        sa.profile.system_prompt = f"""\
{sa.profile.system_prompt.rstrip()}

### Internal privacy guidance

{text.strip()}
"""


def load_scenario(path: str | Path, use_guide: bool = False) -> Scenario:
    """Load a Scenario from a YAML file.

    When ``use_guide=True``, looks for ``scenario_guide.yaml`` next to
    ``scenario.yaml`` and appends each agent's compliance guidance to
    its ``system_prompt`` before returning. This is the R1
    experimental arm; without ``use_guide``, the scenario runs as the
    R0 baseline.
    """
    path = Path(path)
    scenario_dir = path.parent
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    agents: list[ScenarioAgent] = []
    for agent_cfg in raw.get("agents", []):
        profile = PrivacyAgentProfile.from_config(agent_cfg)
        raw_files = agent_cfg.get("sandbox_files", {})
        sandbox_files = _resolve_sandbox_files(raw_files, scenario_dir)
        agents.append(ScenarioAgent(
            id=profile.id,
            profile=profile,
            sandbox_files=sandbox_files,
        ))

    if use_guide:
        _apply_guide(agents, scenario_dir)

    sensitive_claims = [
        SensitiveClaim.model_validate(c) for c in raw.get("sensitive_claims", [])
    ]

    return Scenario(
        name=raw["name"],
        description=raw.get("description", ""),
        agents=agents,
        kickoff_message=raw["kickoff_message"],
        kickoff_from=raw.get("kickoff_from", "system"),
        kickoff_channel=raw.get("kickoff_channel", "general"),
        kickoff_to=raw.get("kickoff_to"),
        max_steps=raw.get("max_steps", 20),
        max_tool_rounds=raw.get("max_tool_rounds", 10),
        max_wall_seconds=raw.get("max_wall_seconds"),
        sandbox_backend=raw.get("sandbox_backend", "local"),
        docker_mem_limit=raw.get("docker_mem_limit", "4g"),
        docker_cpus=raw.get("docker_cpus"),
        docker_network_disabled=raw.get("docker_network_disabled", True),
        compact_at_tokens=raw.get("compact_at_tokens"),
        compact_keep_recent=raw.get("compact_keep_recent", 3),
        scenario_dir=scenario_dir,
        goal=raw.get("goal"),
        sensitive_claims=sensitive_claims,
        eval_model=raw.get("eval_model"),
        enable_llm_violation_judge=raw.get("enable_llm_violation_judge", True),
        expected_failure_modes=list(raw.get("expected_failure_modes", [])),
        provenance_store_content=raw.get("provenance_store_content", False),
        provenance_policy=raw.get("provenance_policy"),
    )
