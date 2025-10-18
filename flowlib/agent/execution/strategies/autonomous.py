"""Autonomous execution strategy - run until complete or max cycles."""

import logging
from typing import Optional

from pydantic import Field

from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.core.errors import ExecutionError, NotInitializedError
from flowlib.agent.execution.strategy import ExecutionStrategy
from flowlib.agent.models.state import AgentState
from flowlib.core.models import StrictBaseModel

logger = logging.getLogger(__name__)


class AutonomousConfig(StrictBaseModel):
    """Configuration for autonomous execution strategy."""

    max_cycles: Optional[int] = Field(
        default=None,
        description="Maximum execution cycles. None uses agent's default."
    )
    auto_save: bool = Field(
        default=True,
        description="Auto-save state after each cycle"
    )


class AutonomousStrategy(ExecutionStrategy):
    """Execute agent autonomously for N cycles.

    This strategy runs the agent's reasoning/execution loop until:
    - Task is marked complete by the agent
    - max_cycles is reached
    - An error occurs

    Example:
        >>> config = AutonomousConfig(max_cycles=10)
        >>> strategy = AutonomousStrategy(config)
        >>> final_state = await strategy.execute(agent)
    """

    def __init__(self, config: AutonomousConfig):
        self.config = config

    async def execute(self, agent: BaseAgent) -> AgentState:
        """Run agent until task complete or max_cycles reached.

        Args:
            agent: Initialized BaseAgent instance

        Returns:
            Final AgentState after execution

        Raises:
            NotInitializedError: If agent not initialized
            ExecutionError: If engine unavailable or state invalid
        """
        # Validation
        if not agent.initialized:
            raise NotInitializedError(
                component_name=agent.name,
                operation="autonomous_execution",
                message="Agent must be initialized before execution"
            )

        if not agent._engine:
            raise ExecutionError(
                message="No engine available for autonomous execution",
                agent=agent.name
            )

        if agent._state_manager.current_state is None:
            raise ExecutionError(
                message="Agent state not initialized",
                agent=agent.name
            )

        # Determine cycle limit
        max_cycles = (
            self.config.max_cycles
            or getattr(agent._engine, 'max_iterations', None)
            or 10
        )

        logger.info(
            f"Starting autonomous execution: agent='{agent.name}', "
            f"task_id='{agent._state_manager.current_state.task_id}', "
            f"max_cycles={max_cycles}"
        )

        # Execute cycles
        cycle_count = 0
        current_state = agent._state_manager.current_state

        while cycle_count < max_cycles:
            cycle_count += 1
            logger.info(f"Cycle {cycle_count}/{max_cycles}")

            try:
                # Execute one cycle
                continue_execution = await agent._engine.execute_cycle(
                    state=current_state,
                    memory_context=f"task_{current_state.task_id}"
                )

                # Check completion
                if not continue_execution:
                    logger.info(f"Task complete after {cycle_count} cycles")
                    break

                # Update state reference
                current_state = agent._state_manager.current_state

                # Auto-save if configured
                if self.config.auto_save and agent.config.state_config and agent.config.state_config.auto_save:
                    try:
                        await agent.save_state()
                        logger.debug(f"Auto-saved state after cycle {cycle_count}")
                    except Exception as e:
                        logger.warning(f"Failed to save state: {e}")

            except Exception as e:
                logger.error(f"Error in cycle {cycle_count}: {e}", exc_info=True)
                raise ExecutionError(
                    message=f"Autonomous execution failed in cycle {cycle_count}",
                    agent=agent.name,
                    cause=e
                )

        if cycle_count >= max_cycles:
            logger.warning(
                f"Reached max_cycles ({max_cycles}) before task completion"
            )

        # Final save
        if self.config.auto_save and agent.config.state_config and agent.config.state_config.auto_save:
            try:
                await agent.save_state()
                logger.info("Saved final state")
            except Exception as e:
                logger.error(f"Failed to save final state: {e}")

        final_state = agent._state_manager.current_state
        if final_state is None:
            raise ExecutionError(
                message="State is None after execution",
                agent=agent.name
            )

        logger.info(f"Autonomous execution complete: {cycle_count} cycles")
        return final_state

    async def cleanup(self) -> None:
        """No cleanup needed for autonomous strategy."""
        pass
