"""Scenario data model and YAML loader."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from src.privacy.agents.privacy_agent.models import PrivacyAgentProfile

from .violations import JudgeMode, SensitiveClaim


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
    # When False, there is no shared "general" channel: agents don't auto-join
    # it, and it cannot be posted to or created. Enables hub-and-spoke
    # topologies (contributors talk only to a coordinator via DM).
    shared_general_channel: bool = True
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
    # `claim` natural-language string + `authorized_for` envelope is the
    # ground truth the judge reads against.
    goal: str | None = None
    sensitive_claims: list[SensitiveClaim] = field(default_factory=list)
    # Deterministic cross-subject leak detection (orthogonal to the LLM judge).
    # `task_subjects` names the persona id(s) the task is legitimately about;
    # `cross_subject_identifiers` is the loaded {persona_id: {names/emails/...}}
    # registry. When both are present, `evaluate_cross_subject` flags any
    # NON-subject persona's identifier appearing in an outbound message. The
    # registry defaults to `../../personas/identifiers.yaml` relative to the
    # scenario (override with `cross_subject_registry:` in YAML).
    task_subjects: list[str] = field(default_factory=list)
    cross_subject_identifiers: dict[str, Any] = field(default_factory=dict)
    # Optional model override for LLM-judged eval. Resolution precedence
    # (loader): scenario YAML `eval_model` > `EVAL_MODEL` env > None. When
    # None, the judge inherits the sim's env-configured model (LLM_MODEL).
    eval_model: str | None = None
    # Optional provider override for LLM-judged eval (e.g. run the judge on
    # a different provider than the sim agents). Same precedence: scenario
    # YAML `eval_provider` > `EVAL_PROVIDER` env > None. When None, the judge
    # inherits the sim's env-configured provider (LLM_PROVIDER). `eval_model`
    # alone only switches model within that provider; set this to cross
    # providers (e.g. sim=anthropic, judge=openai).
    eval_provider: str | None = None
    # Which judge implementation to use for leak eval. See
    # `src/experiments/violations.py::evaluate_violations`.
    #   - "llm":   single-shot LLM judge over the full per-recipient transcript (default)
    #   - "agent": tool-using Agent-as-judge — strict superset of llm
    #              (sees same transcript inline, plus grep + verification tools)
    #   - "off":   skip leak eval
    # For the current corpus (streams fit easily in one prompt), llm is
    # the sensible default. Switch to agent for long streams, future
    # file-inspection tooling, or when iterative verification is wanted.
    judge_mode: JudgeMode = "llm"
    # Agent-as-judge knobs. `judge_max_rounds` is the per-pair turn budget
    # (number of LLM calls before forcing a verdict / budget_exhausted).
    # `judge_max_pairs` is a dev-sampling knob: if set, randomly sample
    # that many (claim, recipient) pairs and skip the rest.
    judge_max_rounds: int = 8
    judge_max_pairs: int | None = None
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


def _load_cross_subject_registry(
    raw: dict[str, Any], scenario_dir: Path
) -> dict[str, Any]:
    """Load the persona identifier registry for cross-subject leak detection.

    Path comes from `cross_subject_registry:` in the YAML, defaulting to the
    library convention `../../personas/identifiers.yaml` relative to the
    scenario dir. Returns the `personas:` mapping, or {} if the file is absent.
    """
    rel = raw.get("cross_subject_registry", "../../personas/identifiers.yaml")
    path = scenario_dir / rel
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("personas", {}) or {}


def _resolve_judge_mode(raw: dict[str, Any]) -> JudgeMode:
    """Pick the judge mode from YAML, honoring the legacy bool key for back-compat.

    Precedence: explicit `judge_mode` > legacy `enable_llm_violation_judge` > default "llm".
    The legacy True maps to "llm" (the default judge); False maps to "off".
    """
    if "judge_mode" in raw:
        value = raw["judge_mode"]
        if value not in ("off", "llm", "agent"):
            raise ValueError(
                f"judge_mode must be one of 'off', 'llm', 'agent'; got {value!r}"
            )
        return value
    if "enable_llm_violation_judge" in raw:
        return "llm" if raw["enable_llm_violation_judge"] else "off"
    return "llm"


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

    task_subjects = list(raw.get("task_subjects", []) or [])
    cross_subject_identifiers = (
        _load_cross_subject_registry(raw, scenario_dir) if task_subjects else {}
    )

    return Scenario(
        name=raw["name"],
        description=raw.get("description", ""),
        agents=agents,
        kickoff_message=raw["kickoff_message"],
        kickoff_from=raw.get("kickoff_from", "system"),
        kickoff_channel=raw.get("kickoff_channel", "general"),
        kickoff_to=raw.get("kickoff_to"),
        shared_general_channel=raw.get("shared_general_channel", True),
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
        task_subjects=task_subjects,
        cross_subject_identifiers=cross_subject_identifiers,
        eval_model=raw.get("eval_model") or os.getenv("EVAL_MODEL"),
        eval_provider=raw.get("eval_provider") or os.getenv("EVAL_PROVIDER"),
        judge_mode=_resolve_judge_mode(raw),
        judge_max_rounds=raw.get("judge_max_rounds", 8),
        judge_max_pairs=raw.get("judge_max_pairs"),
        expected_failure_modes=list(raw.get("expected_failure_modes", [])),
        provenance_store_content=raw.get("provenance_store_content", False),
        provenance_policy=raw.get("provenance_policy"),
    )
