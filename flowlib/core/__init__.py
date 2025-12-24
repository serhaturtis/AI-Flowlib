"""Core foundational models and utilities.

Provides strict Pydantic models following CLAUDE.md principles.
"""

# Strict base models - enforce CLAUDE.md principles
from .models import (
    MutableStrictBaseModel,
    StrictBaseModel,
)

# Message source configs - shared contracts between resources and agent modules
from .message_source_config import (
    EmailMessageSourceConfig,
    MessageSourceConfig,
    QueueMessageSourceConfig,
    TimerMessageSourceConfig,
    WebhookMessageSourceConfig,
)

__all__ = [
    # Strict Models - CLAUDE.md compliance
    "StrictBaseModel",
    "MutableStrictBaseModel",
    # Message Source Configs
    "MessageSourceConfig",
    "TimerMessageSourceConfig",
    "EmailMessageSourceConfig",
    "WebhookMessageSourceConfig",
    "QueueMessageSourceConfig",
]
