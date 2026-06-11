"""Privacy-aware agent: LLM-driven with provider-native tool use."""

import asyncio

from src.privacy.agents.base import BaseSimplePrivacyAgent
from src.privacy.database.base import BaseDatabaseController
from src.privacy.protocol.base import BasePrivacyProtocol

from .models import PrivacyAgentProfile
from .prompts import build_system_prompt


class PrivacyAgent(BaseSimplePrivacyAgent[PrivacyAgentProfile]):
    """LLM-driven agent using provider-native tool calling.

    Each step: fetch messages → build prompt → run agentic loop (LLM calls
    tools, gets results, decides next action, loops until done or no more
    tool calls). Behavior is emergent from the system prompt.
    """

    def __init__(
        self,
        profile: PrivacyAgentProfile,
        protocol: BasePrivacyProtocol,
        database: BaseDatabaseController,
        peer_ids: list[str] | None = None,
        polling_interval: float = 10.0,
        max_steps: int | None = None,
        max_tool_rounds: int = 10,
        compact_at_tokens: int | None = None,
        compact_keep_recent: int = 3,
        **kwargs,
    ):
        super().__init__(profile, protocol, database, **kwargs)
        self._peer_ids: list[str] = peer_ids or []
        self._polling_interval = polling_interval
        self._max_steps = max_steps
        self._max_tool_rounds = max_tool_rounds
        self._compact_at_tokens = compact_at_tokens
        self._compact_keep_recent = compact_keep_recent
        self._step_count = 0
        self._last_step_truncated = False

    def set_peer_ids(self, peer_ids: list[str]) -> None:
        self._peer_ids = [pid for pid in peer_ids if pid != self.id]

    def sandbox_image(self) -> str:
        return self.profile.sandbox_image

    async def on_started(self) -> None:
        self.logger.info(
            f"PrivacyAgent started: {self.profile.role.organization}",
            data={
                "organization": self.profile.role.organization,
                "role_type": self.profile.role.type,
                "artifacts": [a.name for a in self.profile.artifacts],
                "sandbox_image": self.profile.sandbox_image,
                "sandbox_backend": self.profile.sandbox_backend,
            },
        )

    async def step(self) -> None:
        self._step_count += 1
        if self._max_steps is not None and self._step_count > self._max_steps:
            self.shutdown()
            return

        response = await self.fetch_messages()

        if response.messages:
            user_content = self._build_inbox_notification(response.messages)
        elif self._last_step_truncated:
            user_content = """\
Your previous step ended because you hit the tool-round cap mid-work.
Continue your task from where you left off."""
        else:
            # Genuinely idle: poll once, then give the LLM the choice.
            await asyncio.sleep(self._polling_interval)
            response = await self.fetch_messages()
            if response.messages:
                user_content = self._build_inbox_notification(response.messages)
            else:
                user_content = """\
No new messages in your inbox. You can:
  - Send a brief status check to a peer if a meaningful interval has passed (don't spam reminders).
  - Call mark_done if the whole team has finished.
  - Respond with no tool calls to wait silently."""

        system_prompt = build_system_prompt(
            self.profile,
            self._peer_ids,
            shared_general_channel=getattr(self._protocol, "auto_general", True),
        )
        self._last_step_truncated = await self._run_agentic_loop(
            user_content,
            system=system_prompt,
            max_tool_rounds=self._max_tool_rounds,
            compact_at_tokens=self._compact_at_tokens,
            compact_keep_recent=self._compact_keep_recent,
        )

    def _build_inbox_notification(self, messages) -> str:
        """Render newly-arrived messages as a metadata-only inbox notification.

        Bodies are pulled by the agent via read_messages(message_ids=[...]).
        File messages also auto-deposit to inbox/<sender>/<filename> in the
        recipient's sandbox.
        """
        lines = []
        for msg in messages:
            if msg.channel is not None:
                where = f"#{msg.channel}"
                arrow = ""
            else:
                where = "DM"
                arrow = "  → you"
            size = len(msg.message.content)
            if msg.message.type == "file":
                tag = f"file: {msg.message.filename}"
            else:
                tag = f"{size} chars"
            ts = msg.created_at.strftime("%H:%M:%SZ")
            lines.append(
                f"  #{msg.index}  {ts}  {where:<10}  {msg.from_agent_id}{arrow}  {tag}"
            )
        body = "\n".join(lines)
        return f"""\
Inbox — {len(messages)} new message(s) since last check:
{body}
Use read_messages(message_ids=[...]) to read bodies. File messages also
land at inbox/<sender>/<filename> in your sandbox."""
