"""ExecuteCommand handler — runs a command in a sandbox and records provenance.

The sandbox boundary is treated as a *principal boundary*. When a command's
stdout is returned to the invoking agent, we record a transfer from a synthetic
`sandbox:<image>` principal into the agent. If the agent later forwards that
content to another principal, the existing cross-principal detection in
`ProvenanceStore` catches the transitive flow — that's the whole point of the
integration.
"""

from src.privacy.core import ActionExecutionResult, AgentProfile, ExecuteCommand
from src.privacy.database.base import BaseDatabaseController
from src.privacy.provenance import ProvenanceStore
from src.privacy.sandbox import Sandbox


async def execute_execute_command(
    action: ExecuteCommand,
    agent: AgentProfile,
    database: BaseDatabaseController,  # noqa: ARG001 — kept for signature parity
    sandbox: Sandbox,
    provenance: ProvenanceStore | None,
    image: str,
) -> ActionExecutionResult:
    """Execute a command in the supplied sandbox and record provenance."""
    result = await sandbox.run(
        command=action.command,
        stdin=action.stdin,
        timeout=action.timeout_seconds,
        workdir=action.workdir,
        image=image,
    )

    if provenance is not None and not result.timed_out:
        sender_principal = f"sandbox:{image or 'local'}"
        receiver_principal = agent.metadata.get("principal_name", "unknown")
        allowed_receivers = agent.metadata.get("allowed_receivers", [])
        provenance.record(
            sender_agent_id=sender_principal,
            sender_principal=sender_principal,
            allowed_receivers=list(allowed_receivers),
            receiver_agent_id=action.agent_id,
            receiver_principal=receiver_principal,
            content=result.stdout,
        )

    return ActionExecutionResult(
        content=result.model_dump(mode="json"),
        is_error=result.exit_code != 0 or result.timed_out,
        metadata={
            "sandbox_image": image,
            "timed_out": result.timed_out,
            "duration_ms": result.duration_ms,
        },
    )
