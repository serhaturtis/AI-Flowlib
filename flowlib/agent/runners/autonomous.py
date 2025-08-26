"""
Autonomous runner for agents.
"""

import logging
import asyncio
from typing import TYPE_CHECKING, Optional

# Avoid circular import, only type hint BaseAgent
if TYPE_CHECKING:
    from ..core.agent import BaseAgent
    from flowlib.agent.models.state import AgentState

# Import exceptions
from ..core.errors import NotInitializedError, ExecutionError

logger = logging.getLogger(__name__)


async def run_autonomous(
    agent: 'BaseAgent',
    max_cycles: Optional[int] = None,
    **kwargs
) -> 'AgentState':
    """Runs the agent autonomously until the task is complete or max_cycles is reached.

    Relies on the agent's initialized state (including task description)
    and uses the agent's engine to execute cycles.

    Args:
        agent: An initialized BaseAgent instance.
        max_cycles: Maximum number of cycles to execute. If None, uses the
                    value from the agent's engine configuration.
        **kwargs: Additional arguments passed to the engine's execute_cycle.

    Returns:
        The final AgentState after execution stops.

    Raises:
        NotInitializedError: If the agent or its engine is not initialized.
        ExecutionError: If the agent's engine is not available or a cycle fails.
    """
    if not agent or not agent.initialized:
        raise NotInitializedError(
            component_name=getattr(agent, 'name', 'Agent'),
            operation="run_autonomous",
            message="Agent must be initialized before running autonomously."
        )

    if not agent._engine:
        raise ExecutionError(
            message="No engine available for autonomous task execution",
            agent=agent.name
        )

    # Configure the max cycles
    cycles_limit = max_cycles or agent._engine._config.max_iterations
    logger.info(f"Starting autonomous run for agent '{agent.name}' (task: {agent._state_manager.current_state.task_id}), max_cycles={cycles_limit}")

    try:
        cycle_count = 0
        current_state = agent._state_manager.current_state

        while cycle_count < cycles_limit:
            cycle_count += 1
            logger.info(f"Starting autonomous cycle {cycle_count}/{cycles_limit}")
            
            # Execute a single cycle using the agent's engine
            continue_execution = await agent._engine.execute_cycle(
                state=current_state,
                memory_context=f"task_{current_state.task_id}",
                **kwargs
            )

            # Check if we should continue
            if not continue_execution:
                logger.info(f"Autonomous task execution deemed complete by engine after {cycle_count} cycles.")
                break
                
            # Update our reference to the potentially modified state
            current_state = agent._state_manager.current_state
            
            # Save checkpoint after each cycle if configured
            if agent._state_manager._state_persister and agent.config.state_config and agent.config.state_config.auto_save:
                 try:
                      await agent.save_state()
                      logger.debug(f"Auto-saved state after cycle {cycle_count}")
                 except Exception as save_err:
                      logger.warning(f"Failed to auto-save state after cycle {cycle_count}: {save_err}")
        
        if cycle_count >= cycles_limit:
             logger.warning(f"Autonomous run reached max_cycles limit ({cycles_limit}) before task completion.")

        # Save final state (regardless of completion status)
        if agent._state_manager._state_persister and agent.config.state_config and agent.config.state_config.auto_save:
            try:
                await agent.save_state()
                logger.info("Saved final state after autonomous run.")
            except Exception as save_err:
                 logger.error(f"Failed to save final state after autonomous run: {save_err}")
            
        logger.info(f"Autonomous run finished for agent '{agent.name}'.")
        return agent._state_manager.current_state # Return the final state

    except Exception as e:
        # Log and wrap the error with execution context
        error_msg = f"Autonomous task execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update state with error if possible
        if agent._state_manager.current_state and hasattr(agent._state_manager.current_state, 'add_error'):
            agent._state_manager.current_state.add_error(str(e))
        
        # Wrap and re-raise with execution context
        raise ExecutionError(
            message=error_msg,
            agent=agent.name,
            state=agent._state_manager.current_state,
            cause=e
        ) 