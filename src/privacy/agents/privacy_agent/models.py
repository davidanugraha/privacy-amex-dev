"""Data models for privacy-aware agents."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from src.privacy.core import AgentProfile

SensitivityLevel = Literal["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET"]


class Role(BaseModel):
    """Identity of the organization the agent represents."""

    organization: str = Field(description="Organization name, e.g. 'CardIssuer Inc.'")
    type: str = Field(
        description="Canonical role tag, e.g. 'patient_portal', 'credit_card_issuer'"
    )
    context: str = Field(description="Broad domain, e.g. 'financial', 'healthcare'")
    description: str = Field(
        default="",
        description="Freeform paragraph the agent reads as its identity",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Escape-hatch labels for novel role attributes",
    )


class DeclaredInterest(BaseModel):
    """An openly-pursued secondary goal — a public 'we'd also like to ...'."""

    id: str
    description: str


class ExplicitConstraint(BaseModel):
    """A hard rule from regulation, contract, or public commitment."""

    id: str
    audience: str = Field(
        default="",
        description="Who this commitment is addressed to (e.g. 'customers', 'regulators')",
    )
    text: str = Field(description="The rule itself, in natural language")


class Artifact(BaseModel):
    """A piece of data owned by an organization with a sensitivity label."""

    name: str
    data_type: str = Field(description="e.g. 'financial_record', 'patient_pii'")
    sensitivity: SensitivityLevel
    owner_principal: str = Field(description="Name of the organization that owns this artifact")
    content: str = Field(description="The actual data content")


class PrivacyAgentProfile(AgentProfile):
    """Agent profile carrying role, constraints, interests, artifacts.

    Sandbox fields default to a local subprocess backend so profiles that don't
    exercise command execution don't need to configure anything.
    """

    role: Role
    declared_interests: list[DeclaredInterest] = Field(default_factory=list)
    explicit_constraints: list[ExplicitConstraint] = Field(default_factory=list)
    implicit_constraints: str = Field(
        default="",
        description="Behavioral tendencies in narrative — folded into description in prompt",
    )
    undeclared_interests: str = Field(
        default="",
        description="Private appetites in narrative — folded into description in prompt",
    )
    artifacts: list[Artifact] = Field(default_factory=list)
    system_prompt: str = Field(default="", description="Role-specific persona override")
    sandbox_image: str = Field(
        default="python:3.11-slim",
        description="Container image used when sandbox_backend is 'docker'",
    )
    sandbox_backend: Literal["docker", "local"] = Field(
        default="local",
        description="Which sandbox backend this agent's commands run in",
    )

    @classmethod
    def from_config(cls, config: dict) -> "PrivacyAgentProfile":
        """Create a PrivacyAgentProfile from a YAML-parsed dict."""
        role = Role.model_validate(config["role"])
        sandbox_image = config.get("sandbox_image", "python:3.11-slim")
        sandbox_backend = config.get("sandbox_backend", "local")

        return cls(
            id=config.get("id", str(uuid.uuid4())),
            role=role,
            declared_interests=[
                DeclaredInterest.model_validate(d)
                for d in config.get("declared_interests", [])
            ],
            explicit_constraints=[
                ExplicitConstraint.model_validate(c)
                for c in config.get("explicit_constraints", [])
            ],
            implicit_constraints=config.get("implicit_constraints", ""),
            undeclared_interests=config.get("undeclared_interests", ""),
            artifacts=[Artifact.model_validate(a) for a in config.get("artifacts", [])],
            system_prompt=config.get("system_prompt", ""),
            sandbox_image=sandbox_image,
            sandbox_backend=sandbox_backend,
            metadata={
                **config.get("metadata", {}),
                "organization": role.organization,
                "role_type": role.type,
                "role_context": role.context,
                "sandbox_image": sandbox_image,
            },
        )
