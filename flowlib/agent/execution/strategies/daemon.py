"""Daemon execution strategy - runs message sources continuously."""

import asyncio
import logging

from pydantic import Field

from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.core.message_source import MessageSource, MessageSourceConfig
from flowlib.agent.execution.strategy import ExecutionStrategy
from flowlib.core.models import StrictBaseModel

logger = logging.getLogger(__name__)


class DaemonConfig(StrictBaseModel):
    """Configuration for daemon execution."""

    message_sources: list[MessageSourceConfig] = Field(
        default_factory=list, description="Message sources to activate"
    )
    shutdown_timeout_seconds: float = Field(default=30.0, gt=0)
    enable_health_check: bool = Field(default=False, description="Enable health check endpoint")
    health_check_port: int = Field(default=8080, description="Health check port")


class DaemonStrategy(ExecutionStrategy):
    """Execute agent as daemon with message sources.

    This strategy:
    1. Starts agent's queue processing thread
    2. Activates all configured message sources
    3. Sources produce messages → agent's input_queue
    4. Agent processes messages via its normal loop
    5. Runs until stopped

    Example:
        >>> from flowlib.agent.core.message_sources import TimerMessageSourceConfig
        >>> config = DaemonConfig(
        ...     message_sources=[
        ...         TimerMessageSourceConfig(
        ...             name="hourly",
        ...             interval_seconds=3600
        ...         )
        ...     ]
        ... )
        >>> strategy = DaemonStrategy(config)
        >>> await strategy.execute(agent)  # Runs until stopped
    """

    def __init__(self, config: DaemonConfig):
        self.config = config
        self._sources: list[MessageSource] = []
        self._source_tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    async def execute(self, agent: BaseAgent) -> None:
        """Run daemon continuously.

        Args:
            agent: Initialized BaseAgent instance

        Raises:
            ValueError: If agent not initialized
        """
        if not agent.initialized:
            raise ValueError("Agent must be initialized before daemon execution")

        logger.info(
            f"Starting daemon for agent '{agent.name}' with "
            f"{len(self.config.message_sources)} message sources"
        )

        # Start agent's queue processing in background thread
        if not agent._running:
            agent.start()
            logger.info(f"Started agent '{agent.name}' queue processing")

        # Create and start message sources
        for source_config in self.config.message_sources:
            try:
                source = self._create_source(source_config)
                self._sources.append(source)

                # Start each source in its own task
                task = asyncio.create_task(
                    source.start(agent.input_queue), name=f"source_{source.config.name}"
                )
                self._source_tasks.append(task)

                logger.info(
                    f"Started message source: {source.config.name} (enabled={source.enabled})"
                )
            except Exception as e:
                logger.error(
                    f"Failed to start source '{source_config.name}': {e}",
                    exc_info=True,
                )

        if not self._sources:
            logger.warning("No message sources configured, daemon will idle")

        logger.info("Daemon running, waiting for shutdown signal...")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        logger.info("Daemon shutting down...")

    def _create_source(self, config: MessageSourceConfig) -> MessageSource:
        """Create message source from config.

        Args:
            config: Source configuration

        Returns:
            MessageSource instance

        Raises:
            ValueError: If source type is unknown
        """
        from flowlib.agent.core.message_sources import (
            EmailMessageSource,
            EmailMessageSourceConfig,
            TimerMessageSource,
            TimerMessageSourceConfig,
        )

        if isinstance(config, TimerMessageSourceConfig):
            return TimerMessageSource(config)
        elif isinstance(config, EmailMessageSourceConfig):
            return EmailMessageSource(config)
        else:
            raise ValueError(f"Unknown message source config type: {type(config).__name__}")

    async def cleanup(self) -> None:
        """Stop all sources and shutdown.

        This method stops all message sources cleanly and waits for
        their tasks to complete.
        """
        logger.info("Stopping all message sources...")
        self._shutdown_event.set()

        # Stop all sources
        for source in self._sources:
            try:
                await source.stop()
                logger.debug(f"Stopped source: {source.config.name}")
            except Exception as e:
                logger.error(f"Error stopping source '{source.config.name}': {e}")

        # Cancel all source tasks
        for task in self._source_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks with timeout
        if self._source_tasks:
            done, pending = await asyncio.wait(
                self._source_tasks, timeout=self.config.shutdown_timeout_seconds
            )

            if pending:
                logger.warning(f"{len(pending)} source tasks did not complete within timeout")

        logger.info("Daemon cleanup complete")
