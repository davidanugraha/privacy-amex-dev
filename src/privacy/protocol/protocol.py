"""Privacy framework protocol — DM, channels, sandbox execution."""

from datetime import UTC, datetime
from typing import Callable

from src.privacy.core import (
    ActionAdapter,
    ActionExecutionRequest,
    ActionExecutionResult,
    AgentProfile,
    ChannelMessage,
    CreateChannel,
    ExecuteCommand,
    FetchMessages,
    FileMessage,
    MarkDone,
    ReadMessages,
    SendMessage,
)
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.models import ActionRow, ActionRowData
from src.privacy.protocol.base import BasePrivacyProtocol
from src.privacy.sandbox import Sandbox, build_sandbox
from src.provenance import PolicyViolation, ProvenanceRecorder

from .execute_command import execute_execute_command
from .messaging import (
    execute_channel_message,
    execute_create_channel,
    execute_fetch_messages,
    execute_read_messages,
    execute_send_message,
)


class PrivacyProtocol(BasePrivacyProtocol):
    """Protocol with DM, channel messaging, and sandbox execution.

    Optionally accepts a `ProvenanceRecorder` — if present, every send and
    tool use feeds the lineage graph at dispatch time.
    """

    def __init__(
        self,
        sandbox: Sandbox | None = None,
        recorder: ProvenanceRecorder | None = None,
        action_sink: Callable[[ActionRow], None] | None = None,
    ):
        self._sandbox = sandbox or build_sandbox()
        self._channels: dict[str, set[str]] = {}
        # In-memory liveness set. Populated when an agent registers via
        # `register_agent_to_general`. Agents that have marked themselves
        # done stay in this set so peers can still DM them — same as a
        # teammate who has said "I'm done with my part" but is still at
        # the desk and answers questions.
        self._active_agents: set[str] = set()
        # Sticky set of agents that have called `MarkDone`. The scenario
        # ends when every registered agent appears here.
        self._done_marks: set[str] = set()
        self._recorder = recorder
        self._action_sink = action_sink

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    @property
    def channels(self) -> dict[str, set[str]]:
        return self._channels

    @property
    def active_agents(self) -> set[str]:
        return self._active_agents

    def is_marked_done(self, agent_id: str) -> bool:
        return agent_id in self._done_marks

    def all_marked_done(self) -> bool:
        """Every registered agent has called `mark_done`."""
        return bool(self._active_agents) and self._active_agents.issubset(self._done_marks)

    def register_agent_to_general(self, agent_id: str) -> None:
        self._channels.setdefault("general", set()).add(agent_id)
        self._active_agents.add(agent_id)

    def get_actions(self):
        return [SendMessage, ChannelMessage, CreateChannel, FetchMessages, ReadMessages, ExecuteCommand, MarkDone]

    async def initialize(self, database: BaseDatabaseController) -> None:
        """No-op for the in-memory backend."""

    async def execute_action(
        self,
        *,
        agent: AgentProfile,
        action: ActionExecutionRequest,
        database: BaseDatabaseController,
    ) -> ActionExecutionResult:
        """Dispatch an action and persist the action row."""
        parsed_action = ActionAdapter.validate_python(action.parameters)
        result = await self._dispatch(parsed_action, agent, database)

        row = ActionRow(
            id="",
            created_at=datetime.now(UTC),
            data=ActionRowData(agent_id=agent.id, request=action, result=result),
        )
        await database.actions.create(row)
        if self._action_sink is not None:
            try:
                self._action_sink(row)
            except Exception:
                pass
        return result

    async def _dispatch(
        self,
        parsed_action,
        agent: AgentProfile,
        database: BaseDatabaseController,
    ) -> ActionExecutionResult:
        if isinstance(parsed_action, SendMessage):
            content = _message_text(parsed_action.message)
            # Liveness check: refuse DMs to recipients that never registered
            # (e.g., a hallucinated peer ID). Agents that have marked
            # themselves done stay in `_active_agents` so peers can still
            # DM them until everyone has marked done.
            if parsed_action.to_agent_id not in self._active_agents:
                if self._recorder is not None:
                    self._recorder.record_send(
                        sender_agent_id=parsed_action.from_agent_id,
                        recipient_agent_id=parsed_action.to_agent_id,
                        channel=None,
                        content=content,
                        delivered=False,
                        delivery_error="unknown recipient",
                    )
                return ActionExecutionResult(
                    content={"error": f"recipient {parsed_action.to_agent_id!r} is not a registered agent"},
                    is_error=True,
                    metadata={"status": "unknown_recipient"},
                )
            if self._recorder is not None:
                blocked = await self._check_policy(
                    sender_agent_id=parsed_action.from_agent_id,
                    recipient_agent_id=parsed_action.to_agent_id,
                    channel=None,
                    content=content,
                )
                if blocked is not None:
                    return blocked
            result = await execute_send_message(parsed_action, database)
            if not result.is_error and isinstance(parsed_action.message, FileMessage):
                await self._deposit_file(
                    sender_id=parsed_action.from_agent_id,
                    recipient_ids=[parsed_action.to_agent_id],
                    message=parsed_action.message,
                )
            if self._recorder is not None:
                self._recorder.record_send(
                    sender_agent_id=parsed_action.from_agent_id,
                    recipient_agent_id=parsed_action.to_agent_id,
                    channel=None,
                    content=content,
                    delivered=not result.is_error,
                    delivery_error=_error_text(result),
                )
            return result

        if isinstance(parsed_action, ChannelMessage):
            content = _message_text(parsed_action.message)
            if self._recorder is not None:
                blocked = await self._check_policy(
                    sender_agent_id=parsed_action.from_agent_id,
                    recipient_agent_id=None,
                    channel=parsed_action.channel,
                    content=content,
                )
                if blocked is not None:
                    return blocked
            result = await execute_channel_message(parsed_action, self._channels)
            if not result.is_error and isinstance(parsed_action.message, FileMessage):
                members = self._channels.get(parsed_action.channel, set())
                recipients = [m for m in members if m != parsed_action.from_agent_id]
                await self._deposit_file(
                    sender_id=parsed_action.from_agent_id,
                    recipient_ids=recipients,
                    message=parsed_action.message,
                )
            if self._recorder is not None:
                self._recorder.record_send(
                    sender_agent_id=parsed_action.from_agent_id,
                    recipient_agent_id=None,
                    channel=parsed_action.channel,
                    content=content,
                    delivered=not result.is_error,
                    delivery_error=_error_text(result),
                )
            return result

        if isinstance(parsed_action, CreateChannel):
            return await execute_create_channel(parsed_action, self._channels, agent.id)

        if isinstance(parsed_action, FetchMessages):
            return await execute_fetch_messages(parsed_action, agent, database, self._channels)

        if isinstance(parsed_action, ReadMessages):
            return await execute_read_messages(parsed_action, agent, database, self._channels)

        if isinstance(parsed_action, ExecuteCommand):
            result = await execute_execute_command(
                parsed_action, agent, database, sandbox=self._sandbox,
            )
            if self._recorder is not None:
                stdout = ""
                exit_code = 0
                if isinstance(result.content, dict):
                    stdout = str(result.content.get("stdout", ""))
                    exit_code = int(result.content.get("exit_code", 0) or 0)
                self._recorder.record_tool_use(
                    agent_id=parsed_action.agent_id,
                    tool_name="bash",
                    args={"command": parsed_action.command, "stdin": parsed_action.stdin},
                    output=stdout,
                    exit_code=exit_code,
                )
            return result

        if isinstance(parsed_action, MarkDone):
            # The agent has marked the task done. It stays alive (remains
            # in `_active_agents`) so peers can still DM it; the runner
            # ends the scenario when `all_marked_done()` becomes true.
            self._done_marks.add(parsed_action.agent_id)
            return ActionExecutionResult(
                content={
                    "status": "marked_done",
                    "agent_id": parsed_action.agent_id,
                    "marked_done": sorted(self._done_marks),
                    "active_agents": sorted(self._active_agents),
                },
                is_error=False,
                metadata={"status": "marked_done"},
            )

        raise ValueError(f"Unknown action type: {parsed_action.type}")

    async def _deposit_file(
        self,
        *,
        sender_id: str,
        recipient_ids: list[str],
        message: FileMessage,
    ) -> None:
        """Auto-deposit a delivered file into each recipient's sandbox.

        Path convention: `inbox/<sender_id>/<filename>`. Overwrites if a file
        with the same name from the same sender already exists. Failures are
        logged but don't poison the action result — the message is already
        recorded in the action log (the canonical evidence channel); sandbox
        deposit is convenience for shell-level tools.
        """
        deposit_path = f"inbox/{sender_id}/{message.filename}"
        for recipient in recipient_ids:
            try:
                await self._sandbox.write_file(recipient, deposit_path, message.content)
            except Exception as e:  # noqa: BLE001
                # Don't escalate — the message is in the action log regardless.
                # Surface via the recorder if attached so the run trace shows it.
                if self._recorder is not None:
                    self._recorder.record_send(
                        sender_agent_id=sender_id,
                        recipient_agent_id=recipient,
                        channel=None,
                        content=f"<file deposit failed: {deposit_path}>",
                        delivered=False,
                        delivery_error=f"sandbox.write_file raised: {e!r}",
                    )

    async def _check_policy(
        self,
        *,
        sender_agent_id: str,
        recipient_agent_id: str | None,
        channel: str | None,
        content: str,
    ) -> ActionExecutionResult | None:
        """Run the recorder's policy hook; return a blocked-result on veto.

        Caller has already verified `self._recorder is not None`.
        """
        assert self._recorder is not None
        try:
            await self._recorder.check_send(
                sender_agent_id=sender_agent_id,
                recipient_agent_id=recipient_agent_id,
                channel=channel,
                content=content,
            )
        except PolicyViolation as e:
            self._recorder.record_send(
                sender_agent_id=sender_agent_id,
                recipient_agent_id=recipient_agent_id,
                channel=channel,
                content=content,
                delivered=False,
                delivery_error=str(e),
            )
            return ActionExecutionResult(
                content={"error": str(e)},
                is_error=True,
                metadata={"status": "blocked_by_policy"},
            )
        return None


def _message_text(message) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return message.model_dump_json()


def _error_text(result: ActionExecutionResult) -> str | None:
    if not result.is_error:
        return None
    if isinstance(result.content, dict):
        err = result.content.get("error")
        if isinstance(err, str):
            return err
    return str(result.content)
