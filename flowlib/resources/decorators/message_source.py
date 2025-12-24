"""Message source decorators for registering message source resources.

These decorators register message source configurations in the resource registry,
allowing agents to reference them by name.

Example:
    @timer_source("hourly-check")
    class HourlyCheck:
        interval_seconds = 3600
        run_on_start = True
        message_content = "Hourly status check"

    @email_source("inbox-monitor")
    class InboxMonitor:
        email_provider_name = "default-email"
        check_interval_seconds = 300
        folder = "INBOX"
"""

from collections.abc import Callable
from typing import Any

from flowlib.core.message_source_config import MessageSourceDefaults
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.models.message_source_resource import (
    EmailSourceResource,
    MessageSourceType,
    QueueSourceResource,
    TimerSourceResource,
    WebhookSourceResource,
)
from flowlib.resources.registry.registry import resource_registry


def _block_direct_instantiation(cls: type, name: str, resource_type: str) -> None:
    """Prevent direct instantiation of resource classes."""

    def __init_blocked(self: Any, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            f"Cannot instantiate {resource_type} class '{cls.__name__}' directly.\n"
            f"Resources must be retrieved from the resource registry:\n"
            f"  resource = resource_registry.get('{name}')\n\n"
            f"This ensures single source of truth and prevents duplicate instances."
        )

    cls.__init__ = __init_blocked  # type: ignore[method-assign]


def timer_source(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a timer message source resource.

    Timer sources produce messages at regular intervals. Useful for periodic tasks,
    health checks, or scheduled operations.

    Required class attributes:
        interval_seconds: float - Interval between triggers in seconds

    Optional class attributes:
        run_on_start: bool - Send message immediately on start (default: True)
        message_content: str - Content for timer messages (default: "Timer triggered")
        enabled: bool - Whether source is enabled (default: True)

    Args:
        name: Unique name for the timer source
        **metadata: Additional metadata

    Example:
        @timer_source("hourly-check")
        class HourlyCheck:
            interval_seconds = 3600
            run_on_start = True
            message_content = "Hourly status check"
    """

    def decorator(cls: type) -> type:
        # Extract required interval_seconds
        interval_seconds = getattr(cls, "interval_seconds", None)
        if interval_seconds is None:
            raise ValueError(f"timer_source '{name}' requires 'interval_seconds' attribute")

        # Extract optional attributes (using centralized defaults)
        run_on_start = getattr(cls, "run_on_start", MessageSourceDefaults.TIMER_RUN_ON_START)
        message_content = getattr(cls, "message_content", MessageSourceDefaults.TIMER_MESSAGE_CONTENT)
        enabled = getattr(cls, "enabled", MessageSourceDefaults.ENABLED)

        # Attach metadata to class
        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.MESSAGE_SOURCE  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.MESSAGE_SOURCE,
            "source_type": MessageSourceType.TIMER.value,
            **metadata,
        }  # type: ignore[attr-defined]

        # Create and register resource
        instance = TimerSourceResource(
            name=name,
            type=ResourceType.MESSAGE_SOURCE,
            enabled=enabled,
            interval_seconds=interval_seconds,
            run_on_start=run_on_start,
            message_content=message_content,
        )

        resource_registry.register(
            name=name, obj=instance, resource_type=ResourceType.MESSAGE_SOURCE, **metadata
        )
        _block_direct_instantiation(cls, name, "timer source")
        return cls

    return decorator


def email_source(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as an email message source resource.

    Email sources monitor an inbox and convert emails to agent messages.
    Useful for email-driven workflows.

    Required class attributes:
        email_provider_name: str - Reference to email provider config

    Optional class attributes:
        check_interval_seconds: float - Polling interval (default: 300)
        folder: str - Email folder to monitor (default: "INBOX")
        only_unread: bool - Only process unread emails (default: True)
        mark_as_read: bool - Mark emails as read after processing (default: True)
        enabled: bool - Whether source is enabled (default: True)

    Args:
        name: Unique name for the email source
        **metadata: Additional metadata

    Example:
        @email_source("inbox-monitor")
        class InboxMonitor:
            email_provider_name = "default-email"
            check_interval_seconds = 300
            folder = "INBOX"
    """

    def decorator(cls: type) -> type:
        # Extract required email_provider_name
        email_provider_name = getattr(cls, "email_provider_name", None)
        if email_provider_name is None:
            raise ValueError(f"email_source '{name}' requires 'email_provider_name' attribute")

        # Extract optional attributes (using centralized defaults)
        check_interval_seconds = getattr(
            cls, "check_interval_seconds", MessageSourceDefaults.EMAIL_CHECK_INTERVAL_SECONDS
        )
        folder = getattr(cls, "folder", MessageSourceDefaults.EMAIL_FOLDER)
        only_unread = getattr(cls, "only_unread", MessageSourceDefaults.EMAIL_ONLY_UNREAD)
        mark_as_read = getattr(cls, "mark_as_read", MessageSourceDefaults.EMAIL_MARK_AS_READ)
        enabled = getattr(cls, "enabled", MessageSourceDefaults.ENABLED)

        # Attach metadata to class
        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.MESSAGE_SOURCE  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.MESSAGE_SOURCE,
            "source_type": MessageSourceType.EMAIL.value,
            **metadata,
        }  # type: ignore[attr-defined]

        # Create and register resource
        instance = EmailSourceResource(
            name=name,
            type=ResourceType.MESSAGE_SOURCE,
            enabled=enabled,
            email_provider_name=email_provider_name,
            check_interval_seconds=check_interval_seconds,
            folder=folder,
            only_unread=only_unread,
            mark_as_read=mark_as_read,
        )

        resource_registry.register(
            name=name, obj=instance, resource_type=ResourceType.MESSAGE_SOURCE, **metadata
        )
        _block_direct_instantiation(cls, name, "email source")
        return cls

    return decorator


def webhook_source(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a webhook message source resource.

    Webhook sources listen for incoming HTTP requests and convert them
    to agent messages. Useful for event-driven integrations.

    Required class attributes:
        path: str - Webhook path (e.g., "/webhook/slack")

    Optional class attributes:
        methods: list[str] - Allowed HTTP methods (default: ["POST"])
        secret_header: str | None - Header name for secret validation (default: None)
        enabled: bool - Whether source is enabled (default: True)

    Args:
        name: Unique name for the webhook source
        **metadata: Additional metadata

    Example:
        @webhook_source("slack-webhook")
        class SlackWebhook:
            path = "/webhook/slack"
            methods = ["POST"]
            secret_header = "X-Slack-Signature"
    """

    def decorator(cls: type) -> type:
        # Extract required path
        path = getattr(cls, "path", None)
        if path is None:
            raise ValueError(f"webhook_source '{name}' requires 'path' attribute")

        # Extract optional attributes (using centralized defaults)
        methods = getattr(cls, "methods", list(MessageSourceDefaults.WEBHOOK_METHODS))
        secret_header = getattr(cls, "secret_header", MessageSourceDefaults.WEBHOOK_SECRET_HEADER)
        enabled = getattr(cls, "enabled", MessageSourceDefaults.ENABLED)

        # Attach metadata to class
        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.MESSAGE_SOURCE  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.MESSAGE_SOURCE,
            "source_type": MessageSourceType.WEBHOOK.value,
            **metadata,
        }  # type: ignore[attr-defined]

        # Create and register resource
        instance = WebhookSourceResource(
            name=name,
            type=ResourceType.MESSAGE_SOURCE,
            enabled=enabled,
            path=path,
            methods=methods,
            secret_header=secret_header,
        )

        resource_registry.register(
            name=name, obj=instance, resource_type=ResourceType.MESSAGE_SOURCE, **metadata
        )
        _block_direct_instantiation(cls, name, "webhook source")
        return cls

    return decorator


def queue_source(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a queue message source resource.

    Queue sources consume messages from a message queue and convert them
    to agent messages. Useful for distributed task processing.

    Required class attributes:
        queue_provider_name: str - Reference to queue provider (Redis, RabbitMQ)
        queue_name: str - Queue name to consume from

    Optional class attributes:
        consumer_group: str | None - Consumer group for load balancing (default: None)
        enabled: bool - Whether source is enabled (default: True)

    Args:
        name: Unique name for the queue source
        **metadata: Additional metadata

    Example:
        @queue_source("task-queue")
        class TaskQueue:
            queue_provider_name = "default-mq"
            queue_name = "agent-tasks"
            consumer_group = "agent-workers"
    """

    def decorator(cls: type) -> type:
        # Extract required attributes
        queue_provider_name = getattr(cls, "queue_provider_name", None)
        if queue_provider_name is None:
            raise ValueError(f"queue_source '{name}' requires 'queue_provider_name' attribute")

        queue_name = getattr(cls, "queue_name", None)
        if queue_name is None:
            raise ValueError(f"queue_source '{name}' requires 'queue_name' attribute")

        # Extract optional attributes (using centralized defaults)
        consumer_group = getattr(cls, "consumer_group", MessageSourceDefaults.QUEUE_CONSUMER_GROUP)
        enabled = getattr(cls, "enabled", MessageSourceDefaults.ENABLED)

        # Attach metadata to class
        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.MESSAGE_SOURCE  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.MESSAGE_SOURCE,
            "source_type": MessageSourceType.QUEUE.value,
            **metadata,
        }  # type: ignore[attr-defined]

        # Create and register resource
        instance = QueueSourceResource(
            name=name,
            type=ResourceType.MESSAGE_SOURCE,
            enabled=enabled,
            queue_provider_name=queue_provider_name,
            queue_name=queue_name,
            consumer_group=consumer_group,
        )

        resource_registry.register(
            name=name, obj=instance, resource_type=ResourceType.MESSAGE_SOURCE, **metadata
        )
        _block_direct_instantiation(cls, name, "queue source")
        return cls

    return decorator
