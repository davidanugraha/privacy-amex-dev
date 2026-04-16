"""Scenario data model and YAML loader."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from src.privacy.agents.privacy_agent.models import PrivacyAgentProfile


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
    sandbox_backend: Literal["docker", "local"] = "local"


def load_scenario(path: str | Path) -> Scenario:
    """Load a Scenario from a YAML file."""
    path = Path(path)
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    agents: list[ScenarioAgent] = []
    for agent_cfg in raw.get("agents", []):
        profile = PrivacyAgentProfile.from_config(agent_cfg)
        sandbox_files = agent_cfg.get("sandbox_files", {})
        agents.append(ScenarioAgent(
            id=profile.id,
            profile=profile,
            sandbox_files=sandbox_files,
        ))

    return Scenario(
        name=raw["name"],
        description=raw.get("description", ""),
        agents=agents,
        kickoff_message=raw["kickoff_message"],
        kickoff_from=raw.get("kickoff_from", "system"),
        kickoff_channel=raw.get("kickoff_channel", "general"),
        max_steps=raw.get("max_steps", 20),
        max_idle_steps=raw.get("max_idle_steps", 5),
        sandbox_backend=raw.get("sandbox_backend", "local"),
    )
