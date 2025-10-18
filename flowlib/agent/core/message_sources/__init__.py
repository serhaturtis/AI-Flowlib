"""Message sources for agent inputs."""

from flowlib.agent.core.message_sources.email import (
    EmailMessageSource,
    EmailMessageSourceConfig,
)
from flowlib.agent.core.message_sources.timer import (
    TimerMessageSource,
    TimerMessageSourceConfig,
)

__all__ = [
    "TimerMessageSource",
    "TimerMessageSourceConfig",
    "EmailMessageSource",
    "EmailMessageSourceConfig",
]
