"""Scenario data model and YAML loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from src.privacy.agents.privacy_agent.models import PrivacyAgentProfile

from .evaluation import Criterion, DataSubject, SensitiveClaim, parse_criteria


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
    max_steps: int = 20
    max_idle_steps: int = 5
    max_tool_rounds: int = 10
    max_wall_seconds: int | None = None
    terminator_agent: str | None = None
    sandbox_backend: Literal["docker", "local"] = "local"
    success_criteria: list[Criterion] = field(default_factory=list)
    data_subjects: list[DataSubject] = field(default_factory=list)
    sensitive_claims: list[SensitiveClaim] = field(default_factory=list)
    expected_failure_modes: list[dict[str, Any]] = field(default_factory=list)


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


def load_scenario(path: str | Path) -> Scenario:
    """Load a Scenario from a YAML file."""
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

    data_subjects = [
        DataSubject.model_validate(d) for d in raw.get("data_subjects", [])
    ]
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
        max_steps=raw.get("max_steps", 20),
        max_idle_steps=raw.get("max_idle_steps", 5),
        max_tool_rounds=raw.get("max_tool_rounds", 10),
        max_wall_seconds=raw.get("max_wall_seconds"),
        terminator_agent=raw.get("terminator_agent"),
        sandbox_backend=raw.get("sandbox_backend", "local"),
        success_criteria=parse_criteria(
            raw.get("success_criteria"),
            agents=[a.profile for a in agents],
            sensitive_claims=sensitive_claims,
        ),
        data_subjects=data_subjects,
        sensitive_claims=sensitive_claims,
        expected_failure_modes=list(raw.get("expected_failure_modes", [])),
    )
