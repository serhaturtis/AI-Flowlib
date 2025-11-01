"""REPL execution strategy - interactive command-line interface."""

import logging

from pydantic import Field

from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.execution.strategy import ExecutionStrategy
from flowlib.core.models import StrictBaseModel

logger = logging.getLogger(__name__)


class REPLConfig(StrictBaseModel):
    """Configuration for REPL execution strategy."""

    history_file: str = Field(
        default=".flowlib_repl_history.txt", description="Path to readline history file"
    )
    enable_commands: bool = Field(default=True, description="Enable / commands")
    enable_streaming: bool = Field(default=True, description="Stream LLM responses")


class REPLStrategy(ExecutionStrategy):
    """Execute agent in interactive REPL mode.

    This strategy wraps the existing REPL implementation and provides
    a consistent strategy interface. The actual REPL functionality
    is in flowlib.agent.runners.repl.

    Example:
        >>> config = REPLConfig(history_file="my_history.txt")
        >>> strategy = REPLStrategy(config)
        >>> await strategy.execute(agent)  # Blocks until user exits
    """

    def __init__(self, config: REPLConfig):
        self.config = config

    async def execute(self, agent: BaseAgent) -> None:
        """Run interactive REPL session.

        Args:
            agent: Initialized BaseAgent instance

        Returns:
            None (REPL runs until user exits)
        """
        logger.info(f"Starting REPL for agent '{agent.name}'")

        # Import and start REPL implementation
        from flowlib.agent.execution.strategies.repl_impl import start_agent_repl

        await start_agent_repl(
            agent_id=f"repl_{agent.name}",
            config=agent.config,
            history_file=self.config.history_file,
        )

        logger.info("REPL session ended")

    async def cleanup(self) -> None:
        """Cleanup REPL resources.

        The existing REPL implementation handles its own cleanup.
        """
        pass
