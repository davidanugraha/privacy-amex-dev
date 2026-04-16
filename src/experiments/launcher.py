"""Agent launcher — runs agents against a shared protocol + database."""

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from src.privacy.agents.base import BaseAgent
from src.privacy.core import (
    ActionExecutionRequest,
    ActionExecutionResult,
    ChannelMessage,
    TextMessage,
)
from src.privacy.database.base import BaseDatabaseController
from src.privacy.database.memory import MemoryDatabase
from src.privacy.database.models import ActionRow, ActionRowData
from src.privacy.logger import PrivacyLogger
from src.privacy.protocol.base import BasePrivacyProtocol


class AgentLauncher:
    """Orchestrates concurrent execution of agents sharing a protocol + database.

    Agents call the protocol directly and read/write the shared in-process
    database. This class provides the kickoff mechanism, concurrent run helpers,
    and a convenience logger.
    """

    def __init__(
        self,
        protocol: BasePrivacyProtocol,
        database: BaseDatabaseController | None = None,
    ):
        self.protocol = protocol
        self.database = database or MemoryDatabase()

    def create_logger(self, name: str = __name__) -> PrivacyLogger:
        return PrivacyLogger(name, self.database)

    async def kickoff(
        self,
        content: str,
        *,
        channel: str = "general",
        from_agent_id: str = "system",
    ) -> None:
        """Inject a channel message before agents start stepping.

        Call this AFTER agents have registered (or after manually populating
        the general channel) but BEFORE `run_agents`. Every agent in the
        channel will see this message on their first `fetch_messages` call.
        """
        action = ChannelMessage(
            from_agent_id=from_agent_id,
            channel=channel,
            created_at=datetime.now(UTC),
            message=TextMessage(content=content),
        )
        request = ActionExecutionRequest(
            name=action.get_name(),
            parameters=action.model_dump(mode="json"),
        )
        result = ActionExecutionResult(
            content=action.model_dump(mode="json"),
            is_error=False,
            metadata={"status": "kickoff"},
        )
        await self.database.actions.create(
            ActionRow(
                id="",
                created_at=datetime.now(UTC),
                data=ActionRowData(
                    agent_id=from_agent_id, request=request, result=result,
                ),
            )
        )

    async def run_agents(self, *agents: BaseAgent[Any]) -> None:
        """Run agents concurrently until each terminates."""
        if not agents:
            return

        print(f"\nRunning {len(agents)} agents...")
        tasks = [asyncio.create_task(agent.run()) for agent in agents]
        await asyncio.gather(*tasks)
        print("All agents completed")

    async def run_agents_with_dependencies(
        self,
        primary_agents: Sequence[BaseAgent[Any]],
        dependent_agents: Sequence[BaseAgent[Any]],
    ) -> None:
        """Run primary + dependent agents; signal dependents to stop once primaries finish."""
        if not primary_agents and not dependent_agents:
            return

        print(
            f"\nRunning {len(primary_agents)} primary + {len(dependent_agents)} dependent agents..."
        )
        primary_tasks = [asyncio.create_task(a.run()) for a in primary_agents]
        dependent_tasks = [asyncio.create_task(a.run()) for a in dependent_agents]

        try:
            await asyncio.gather(*primary_tasks)
            print("All primary agents completed")

            for a in dependent_agents:
                a.shutdown()
            await asyncio.sleep(0.1)
            await asyncio.gather(*dependent_tasks)
            print("All dependent agents shut down gracefully")
            await asyncio.sleep(0.2)

        except Exception as e:
            print(f"Error during execution: {e}")
            for a in list(primary_agents) + list(dependent_agents):
                a.shutdown()
            await asyncio.sleep(0.1)
            await asyncio.gather(
                *primary_tasks, *dependent_tasks, return_exceptions=True
            )
            await asyncio.sleep(0.2)
            raise
