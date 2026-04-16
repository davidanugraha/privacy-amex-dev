"""ExecuteCommand handler — runs a command in the agent's persistent sandbox."""

from src.privacy.core import ActionExecutionResult, AgentProfile, ExecuteCommand
from src.privacy.database.base import BaseDatabaseController
from src.privacy.sandbox import Sandbox


async def execute_execute_command(
    action: ExecuteCommand,
    agent: AgentProfile,
    database: BaseDatabaseController,  # noqa: ARG001 — kept for signature parity
    sandbox: Sandbox,
) -> ActionExecutionResult:
    """Run a command in the agent's sandbox."""
    result = await sandbox.run(
        agent_id=action.agent_id,
        command=action.command,
        stdin=action.stdin,
        timeout=action.timeout_seconds,
    )

    return ActionExecutionResult(
        content=result.model_dump(mode="json"),
        is_error=result.exit_code != 0 or result.timed_out,
        metadata={
            "timed_out": result.timed_out,
            "duration_ms": result.duration_ms,
        },
    )
