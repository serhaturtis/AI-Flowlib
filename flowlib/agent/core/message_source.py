"""Message source abstraction for agent inputs.

This module provides the core abstraction for all agent message sources.
All inputs to an agent (user, timer, email, webhooks, queues) are unified
through the MessageSource interface.
"""

import queue
from abc import ABC, abstractmethod

from pydantic import Field

from flowlib.agent.core.models.messages import AgentMessage
from flowlib.core.models import StrictBaseModel


class MessageSourceConfig(StrictBaseModel):
    """Base configuration for message sources."""

    name: str = Field(..., description="Unique source name")
    enabled: bool = Field(default=True, description="Whether source is enabled")


class MessageSource(ABC):
    """Base class for all message sources.

    A message source produces AgentMessage objects and places them
    on the agent's input queue. This unifies:
    - User input (REPL, CLI, API)
    - Timer-based triggers
    - Event-based triggers (email, webhooks, filesystem)
    - Queue consumers (MQ, Kafka)

    The agent processes all messages uniformly through its input queue,
    regardless of source.

    Example:
        >>> class MySource(MessageSource):
        ...     async def start(self, input_queue):
        ...         await super().start(input_queue)
        ...         while not self._stopped:
        ...             message = AgentMessage(...)
        ...             input_queue.put(message)
        ...             await asyncio.sleep(1)
        ...
        ...     async def stop(self):
        ...         await super().stop()
    """

    def __init__(self, config: MessageSourceConfig):
        """Initialize message source.

        Args:
            config: Source configuration
        """
        self.config = config
        self._enabled = config.enabled
        self._stopped = False
        self._input_queue: queue.Queue[AgentMessage] | None = None

    @abstractmethod
    async def start(self, input_queue: queue.Queue[AgentMessage]) -> None:
        """Start producing messages.

        This method should begin the source's message production loop.
        Messages are sent by calling: input_queue.put(message)

        Args:
            input_queue: Agent's input queue to send messages to
        """
        self._input_queue = input_queue
        self._stopped = False

    @abstractmethod
    async def stop(self) -> None:
        """Stop producing messages.

        This method should cleanly stop the source's message production
        and release any resources.
        """
        self._stopped = True

    def enable(self) -> None:
        """Enable this source."""
        self._enabled = True

    def disable(self) -> None:
        """Disable this source."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if source is enabled."""
        return self._enabled
