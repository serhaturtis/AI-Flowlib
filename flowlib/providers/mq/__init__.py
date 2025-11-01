"""Message Queue provider package.

This package contains providers for message queue integration, offering a common
interface for working with different message queue systems.
"""

from .base import MessageMetadata, MQProvider, MQProviderSettings
from .kafka.provider import KafkaMQProvider, KafkaProviderSettings
from .rabbitmq.provider import RabbitMQProvider, RabbitMQProviderSettings

__all__ = [
    "MQProvider",
    "MQProviderSettings",
    "MessageMetadata",
    "RabbitMQProvider",
    "RabbitMQProviderSettings",
    "KafkaMQProvider",
    "KafkaProviderSettings",
]
