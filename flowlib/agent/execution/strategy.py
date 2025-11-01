"""Base execution strategy for agents."""

from abc import ABC, abstractmethod
from enum import Enum

from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.models.state import AgentState


class ExecutionMode(str, Enum):
    """Supported agent execution modes."""

    REPL = "repl"
    DAEMON = "daemon"
    AUTONOMOUS = "autonomous"
    REMOTE = "remote"


class ExecutionStrategy(ABC):
    """Base class for agent execution strategies.

    An execution strategy defines HOW an agent runs:
    - When it gets triggered (on input, on timer, on message)
    - How long it runs (once, loop, continuous)
    - What happens between executions (wait for input, sleep, poll queue)

    Example:
        >>> strategy = AutonomousStrategy(max_cycles=10)
        >>> await strategy.execute(agent)
        >>> await strategy.cleanup()
    """

    @abstractmethod
    async def execute(self, agent: BaseAgent) -> AgentState | None:
        """Execute the agent according to this strategy.

        Args:
            agent: Initialized BaseAgent instance

        Returns:
            Final AgentState if applicable, None for continuous strategies

        Raises:
            NotInitializedError: If agent not initialized
            ExecutionError: If execution fails
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources used by this strategy.

        Called after execution completes or on error.
        Must be idempotent - safe to call multiple times.
        """
        pass
