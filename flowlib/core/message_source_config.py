"""Message source configuration models.

This module contains the pure data contracts for message source configurations.
These are shared between:
- Resource definitions (flowlib.resources.models.message_source_resource)
- Agent execution (flowlib.agent.core.message_sources.*)
- Config scaffolding (flowlib.server.server.services.config_scaffold)

By keeping these in flowlib.core, we avoid circular dependencies between
the resources and agent modules.

This module is the SINGLE SOURCE OF TRUTH for message source defaults.
"""

from pydantic import Field

from flowlib.core.models import StrictBaseModel


# =============================================================================
# Default Values - Single Source of Truth
# =============================================================================
# All message source defaults are defined here. Other modules (resource models,
# decorators, scaffold service) should import and use these constants.


class MessageSourceDefaults:
    """Default values for message source configurations.

    Import and use these constants instead of hardcoding defaults.
    """

    # Base defaults
    ENABLED: bool = True

    # Timer source defaults
    TIMER_RUN_ON_START: bool = True
    TIMER_MESSAGE_CONTENT: str = "Timer triggered"

    # Email source defaults
    EMAIL_CHECK_INTERVAL_SECONDS: float = 300.0
    EMAIL_FOLDER: str = "INBOX"
    EMAIL_ONLY_UNREAD: bool = True
    EMAIL_MARK_AS_READ: bool = True

    # Webhook source defaults
    WEBHOOK_METHODS: list[str] = ["POST"]
    WEBHOOK_SECRET_HEADER: str | None = None

    # Queue source defaults
    QUEUE_CONSUMER_GROUP: str | None = None


class MessageSourceConfig(StrictBaseModel):
    """Base configuration for message sources."""

    name: str = Field(..., description="Unique source name")
    enabled: bool = Field(
        default=MessageSourceDefaults.ENABLED,
        description="Whether source is enabled",
    )


class TimerMessageSourceConfig(MessageSourceConfig):
    """Configuration for timer-based message source."""

    interval_seconds: float = Field(..., gt=0, description="Interval between messages")
    run_on_start: bool = Field(
        default=MessageSourceDefaults.TIMER_RUN_ON_START,
        description="Send message immediately on start",
    )
    message_content: str = Field(
        default=MessageSourceDefaults.TIMER_MESSAGE_CONTENT,
        description="Content for timer messages",
    )


class EmailMessageSourceConfig(MessageSourceConfig):
    """Configuration for email-based message source."""

    email_provider_name: str = Field(..., description="Email provider config name")
    check_interval_seconds: float = Field(
        default=MessageSourceDefaults.EMAIL_CHECK_INTERVAL_SECONDS,
        gt=0,
        description="Polling interval in seconds",
    )
    folder: str = Field(
        default=MessageSourceDefaults.EMAIL_FOLDER,
        description="Email folder to monitor",
    )
    only_unread: bool = Field(
        default=MessageSourceDefaults.EMAIL_ONLY_UNREAD,
        description="Only process unread emails",
    )
    mark_as_read: bool = Field(
        default=MessageSourceDefaults.EMAIL_MARK_AS_READ,
        description="Mark emails as read after processing",
    )


class WebhookMessageSourceConfig(MessageSourceConfig):
    """Configuration for webhook-based message source."""

    path: str = Field(..., description="Webhook path (e.g., '/webhook/slack')")
    methods: list[str] = Field(
        default_factory=lambda: list(MessageSourceDefaults.WEBHOOK_METHODS),
        description="Allowed HTTP methods",
    )
    secret_header: str | None = Field(
        default=MessageSourceDefaults.WEBHOOK_SECRET_HEADER,
        description="Header name for secret validation",
    )


class QueueMessageSourceConfig(MessageSourceConfig):
    """Configuration for queue-based message source."""

    queue_provider_name: str = Field(
        ..., description="Reference to queue provider (Redis, RabbitMQ)"
    )
    queue_name: str = Field(..., description="Queue name to consume from")
    consumer_group: str | None = Field(
        default=MessageSourceDefaults.QUEUE_CONSUMER_GROUP,
        description="Consumer group for load balancing",
    )
