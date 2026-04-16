"""Data models for privacy-aware agents."""

from typing import Literal

from pydantic import BaseModel, Field

from src.privacy.core import AgentProfile

SensitivityLevel = Literal["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET"]


class Principal(BaseModel):
    """An organization — the unit of trust boundary."""

    name: str = Field(description="Organization name, e.g. 'ACME Corp'")
    allowed_receivers: list[str] = Field(
        default_factory=list,
        description="Principal names this org is allowed to send data to",
    )


class Artifact(BaseModel):
    """A piece of data owned by a principal with a sensitivity label."""

    name: str
    data_type: str = Field(description="e.g. 'financial_record', 'patient_pii'")
    sensitivity: SensitivityLevel
    owner_principal: str = Field(description="Name of the principal that owns this artifact")
    content: str = Field(description="The actual data content")


class PrivacyAgentProfile(AgentProfile):
    """Agent profile with principal, artifacts, and sandbox configuration.

    Sandbox fields default to a local subprocess backend so profiles that don't
    exercise command execution don't need to configure anything.
    """

    principal: Principal
    artifacts: list[Artifact] = Field(default_factory=list)
    system_prompt: str = Field(default="", description="Role-specific instructions")
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
        import uuid

        principal_data = config.get("principal", {})
        artifacts_data = config.get("artifacts", [])
        sandbox_image = config.get("sandbox_image", "python:3.11-slim")
        sandbox_backend = config.get("sandbox_backend", "local")

        return cls(
            id=config.get("id", str(uuid.uuid4())),
            principal=Principal.model_validate(principal_data),
            artifacts=[Artifact.model_validate(a) for a in artifacts_data],
            system_prompt=config.get("system_prompt", ""),
            sandbox_image=sandbox_image,
            sandbox_backend=sandbox_backend,
            metadata={
                **config.get("metadata", {}),
                "principal_name": principal_data.get("name", "unknown"),
                "allowed_receivers": principal_data.get("allowed_receivers", []),
                "sandbox_image": sandbox_image,
            },
        )


