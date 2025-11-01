"""Remote execution strategy - message queue consumer."""

import logging
from typing import Any

from pydantic import Field

from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.execution.strategies.autonomous import (
    AutonomousConfig,
    AutonomousStrategy,
)
from flowlib.agent.execution.strategy import ExecutionStrategy
from flowlib.core.models import StrictBaseModel

logger = logging.getLogger(__name__)


class RemoteConfig(StrictBaseModel):
    """Configuration for remote execution strategy."""

    mq_provider_name: str = Field(..., description="Message queue provider config name")
    task_queue_name: str = Field(..., description="Queue to consume tasks from")
    results_queue_name: str = Field(..., description="Queue to publish results to")
    state_persister_name: str = Field(..., description="State persistence provider config name")


class RemoteStrategy(ExecutionStrategy):
    """Execute agent as remote worker consuming from message queue.

    This strategy connects to a message queue, consumes task messages,
    executes them autonomously, and publishes results back.

    Example:
        >>> config = RemoteConfig(
        ...     mq_provider_name="rabbitmq-default",
        ...     task_queue_name="agent_tasks",
        ...     results_queue_name="agent_results",
        ...     state_persister_name="redis-state"
        ... )
        >>> strategy = RemoteStrategy(config)
        >>> await strategy.execute(agent)  # Runs until stopped
    """

    def __init__(self, config: RemoteConfig):
        self.config = config
        self._running = False
        self._mq_provider: Any | None = None

    async def execute(self, agent: BaseAgent) -> None:
        """Run as remote worker until stopped.

        Args:
            agent: Initialized BaseAgent instance

        Returns:
            None (worker runs until stopped)
        """
        # Get MQ provider
        from flowlib.providers.core.registry import provider_registry

        self._mq_provider = await provider_registry.get_by_config(self.config.mq_provider_name)

        logger.info(
            f"Starting remote worker for agent '{agent.name}': queue={self.config.task_queue_name}"
        )

        self._running = True

        # Start consuming messages
        await self._mq_provider.consume(
            queue_name=self.config.task_queue_name,
            callback=lambda msg: self._process_task(agent, msg),
        )

    async def _process_task(self, agent: BaseAgent, message: Any) -> None:
        """Process a task message.

        Args:
            agent: Agent to execute
            message: Message from queue
        """
        try:
            # Deserialize task
            from flowlib.agent.execution.strategies.remote_impl.models import (
                AgentTaskMessage,
            )

            task_msg = AgentTaskMessage.model_validate_json(message.body)

            logger.info(f"Processing task: {task_msg.task_id}")

            # Load state
            from flowlib.providers.core.registry import provider_registry

            state_persister = await provider_registry.get_by_config(
                self.config.state_persister_name
            )
            state = await state_persister.load_state(task_msg.task_id)
            agent._state_manager.current_state = state

            # Execute autonomously
            autonomous_config = AutonomousConfig(max_cycles=task_msg.max_cycles or 10)
            strategy = AutonomousStrategy(autonomous_config)
            final_state = await strategy.execute(agent)

            # Save state
            await state_persister.save_state(final_state)

            # Publish result
            from flowlib.agent.execution.strategies.remote_impl.models import (
                AgentResultMessage,
            )

            result_msg = AgentResultMessage(task_id=task_msg.task_id, result=final_state)
            await self._mq_provider.publish(
                queue_name=self.config.results_queue_name, message=result_msg.model_dump_json()
            )

            logger.info(f"Task complete: {task_msg.task_id}")

        except Exception as e:
            logger.error(f"Error processing task: {e}", exc_info=True)

    async def cleanup(self) -> None:
        """Cleanup remote worker resources."""
        if self._mq_provider:
            try:
                await self._mq_provider.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting MQ provider: {e}")

        self._running = False
        logger.info("Remote worker cleanup complete")
