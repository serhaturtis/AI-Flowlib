"""Timer-based message source."""

import asyncio
import logging
import queue
from datetime import datetime
from typing import Optional

from pydantic import Field

from flowlib.agent.core.message_source import MessageSource, MessageSourceConfig
from flowlib.agent.core.models.messages import AgentMessage, AgentMessageType

logger = logging.getLogger(__name__)


class TimerMessageSourceConfig(MessageSourceConfig):
    """Configuration for timer-based message source."""

    interval_seconds: float = Field(..., gt=0, description="Interval between messages")
    run_on_start: bool = Field(
        default=True, description="Send message immediately on start"
    )
    message_content: str = Field(
        default="Timer triggered", description="Content for timer messages"
    )


class TimerMessageSource(MessageSource):
    """Produces messages at regular intervals.

    This source sends a message to the agent's input queue at fixed intervals.
    Useful for periodic tasks, health checks, or scheduled operations.

    Example:
        >>> config = TimerMessageSourceConfig(
        ...     name="hourly_check",
        ...     interval_seconds=3600,
        ...     run_on_start=True,
        ...     message_content="Hourly status check"
        ... )
        >>> source = TimerMessageSource(config)
        >>> await source.start(agent.input_queue)
    """

    def __init__(self, config: TimerMessageSourceConfig):
        super().__init__(config)
        self.config: TimerMessageSourceConfig = config
        self._task: Optional[asyncio.Task] = None

    async def start(self, input_queue: queue.Queue[AgentMessage]) -> None:
        """Start timer and produce messages at intervals.

        Args:
            input_queue: Agent's input queue to send messages to
        """
        await super().start(input_queue)

        logger.info(
            f"Starting timer source '{self.config.name}' "
            f"with interval {self.config.interval_seconds}s"
        )

        # Run immediately if configured
        if self.config.run_on_start and self._enabled:
            self._send_message()

        # Main timer loop
        while not self._stopped:
            try:
                await asyncio.sleep(self.config.interval_seconds)
                if not self._stopped and self._enabled:
                    self._send_message()
            except asyncio.CancelledError:
                logger.debug(f"Timer source '{self.config.name}' cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Error in timer source '{self.config.name}': {e}", exc_info=True
                )

        logger.info(f"Timer source '{self.config.name}' stopped")

    def _send_message(self) -> None:
        """Send a timer message to the input queue."""
        if not self._input_queue:
            logger.warning(f"Timer source '{self.config.name}' has no input queue")
            return

        message = AgentMessage(
            message_type=AgentMessageType.SYSTEM,
            content=self.config.message_content,
            context={
                "source_name": self.config.name,
                "source_type": "timer",
                "trigger_time": datetime.now().isoformat(),
            },
            response_queue_id="timer_source",
        )

        self._input_queue.put(message)
        logger.debug(f"Timer source '{self.config.name}' sent message")

    async def stop(self) -> None:
        """Stop the timer.

        Signals the timer loop to stop and releases resources.
        """
        logger.info(f"Stopping timer source '{self.config.name}'")
        await super().stop()
