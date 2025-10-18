"""Kafka message queue provider implementation.

This module provides a concrete implementation of the MQProvider
for Apache Kafka messaging using aiokafka.
"""

import asyncio
import json
import logging
from asyncio import Task
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
from flowlib.providers.mq.base import MessageMetadata, MQProvider, MQProviderSettings

# Removed ProviderType import - using config-driven provider access

logger = logging.getLogger(__name__)

try:
    from aiokafka import (  # type: ignore[import-not-found]
        AIOKafkaConsumer,
        AIOKafkaProducer,
    )
except ImportError:
    logger.warning("aiokafka package not found. Install with 'pip install aiokafka'")


class KafkaProviderSettings(MQProviderSettings):
    """Kafka provider settings - direct inheritance, only Kafka-specific fields.
    
    Kafka requires:
    1. Bootstrap servers for cluster discovery
    2. Producer/consumer configuration
    3. Security settings (SASL, SSL)
    4. Message handling settings
    """

    # Connection settings - Kafka uses bootstrap servers
    host: str = Field(default="localhost", description="Kafka broker host")
    port: int = Field(default=9092, description="Kafka broker port")
    bootstrap_servers: Optional[str] = Field(default=None, description="Comma-separated Kafka bootstrap servers (overrides host/port)")

    # Authentication - inherited from MQProviderSettings as Optional[str]

    # Kafka client settings
    client_id: Optional[str] = Field(default=None, description="Kafka client ID")
    group_id: str = Field(default="flowlib_consumer_group", description="Consumer group ID")
    auto_offset_reset: str = Field(default="latest", description="Offset reset strategy ('earliest', 'latest', 'none')")
    enable_auto_commit: bool = Field(default=True, description="Whether to auto-commit offsets")
    auto_commit_interval_ms: int = Field(default=5000, description="Auto-commit interval in milliseconds")

    # Security settings
    security_protocol: str = Field(default="PLAINTEXT", description="Security protocol ('PLAINTEXT', 'SSL', 'SASL_PLAINTEXT', 'SASL_SSL')")
    sasl_mechanism: Optional[str] = Field(default=None, description="SASL mechanism (PLAIN, GSSAPI, SCRAM-SHA-256, SCRAM-SHA-512)")
    sasl_username: Optional[str] = Field(default=None, description="SASL username")
    sasl_password: Optional[str] = Field(default=None, description="SASL password")
    ssl_context: Optional[Any] = Field(default=None, description="SSL context")
    ssl_check_hostname: bool = Field(default=True, description="Whether to check SSL hostname")

    # Producer settings
    acks: Union[str, int] = Field(default="all", description="Producer acks setting (0, 1, 'all')")
    compression_type: Optional[str] = Field(default=None, description="Compression type ('gzip', 'snappy', 'lz4', None)")

    # Additional connection arguments
    connect_args: Dict[str, Any] = Field(default_factory=dict, description="Additional connection arguments")


@provider(provider_type="message_queue", name="kafka", settings_class=KafkaProviderSettings)
class KafkaMQProvider(MQProvider[KafkaProviderSettings]):
    """Kafka implementation of the MQProvider.
    
    This provider implements message queue operations using aiokafka,
    an asynchronous client for Apache Kafka.
    """

    def __init__(self, name: str = "kafka", settings: Optional[KafkaProviderSettings] = None):
        """Initialize Kafka provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        if settings is None:
            settings = KafkaProviderSettings(
                host="localhost",
                port=9092,
                username="guest",
                password="guest"
            )
        super().__init__(name=name, settings=settings)
        if not isinstance(self.settings, KafkaProviderSettings):
            raise TypeError(f"settings must be a KafkaProviderSettings instance, got {type(self.settings)}")
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumers: Dict[str, Any] = {}
        self._consumer_tasks: Dict[str, Task[None]] = {}

    async def _initialize(self) -> None:
        """Initialize Kafka connection.
        
        Raises:
            ProviderError: If initialization fails
        """
        bootstrap_servers = None
        try:
            # Get bootstrap servers
            bootstrap_servers = self.settings.bootstrap_servers
            if not bootstrap_servers:
                bootstrap_servers = f"{self.settings.host}:{self.settings.port}"

            # Prepare producer config
            producer_config = {
                "bootstrap_servers": bootstrap_servers,
                "acks": self.settings.acks,
                "compression_type": self.settings.compression_type,
                "security_protocol": self.settings.security_protocol,
                **self.settings.connect_args
            }

            # Add client ID if provided
            if self.settings.client_id:
                producer_config["client_id"] = self.settings.client_id

            # Add SASL settings if provided
            if self.settings.sasl_mechanism and self.settings.sasl_username and self.settings.sasl_password:
                producer_config["sasl_mechanism"] = self.settings.sasl_mechanism
                producer_config["sasl_plain_username"] = self.settings.sasl_username
                producer_config["sasl_plain_password"] = self.settings.sasl_password

            # Add SSL settings if provided
            if self.settings.ssl_context:
                producer_config["ssl_context"] = self.settings.ssl_context
                producer_config["ssl_check_hostname"] = self.settings.ssl_check_hostname

            # Create producer
            self._producer = AIOKafkaProducer(**producer_config)

            # Start producer
            await self._producer.start()

            logger.info(f"Connected to Kafka: {bootstrap_servers}")

        except Exception as e:
            self._producer = None
            raise ProviderError(
                message=f"Failed to connect to Kafka: {str(e)}",
                context=ErrorContext.create(
                    flow_name="kafka_provider",
                    error_type="connection_error",
                    error_location="KafkaMQProvider._initialize",
                    component="kafka_provider",
                    operation="initialize"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="initialize",
                    retry_count=0
                ),
                cause=e
            )

    async def _shutdown(self) -> None:
        """Close Kafka connection."""
        try:
            # Stop all consumers
            for topic, consumer_task in self._consumer_tasks.items():
                if not consumer_task.done():
                    consumer_task.cancel()

            for topic, consumer in self._consumers.items():
                try:
                    await consumer.stop()
                except Exception as e:
                    logger.warning(f"Error stopping consumer for topic {topic}: {str(e)}")

            # Stop producer
            if self._producer:
                await self._producer.stop()
                self._producer = None
                logger.info("Stopped Kafka producer")

            # Clear consumers
            self._consumers = {}
            self._consumer_tasks = {}

        except Exception as e:
            logger.error(f"Error during Kafka shutdown: {str(e)}")

    async def publish(self,
                     exchange: str,
                     routing_key: str,
                     message: Any,
                     metadata: Optional[MessageMetadata] = None) -> bool:
        """Publish a message to Kafka.

        Args:
            exchange: Kafka topic (using exchange parameter for interface compatibility)
            routing_key: Message key (optional)
            message: Message to publish
            metadata: Optional message metadata

        Returns:
            True if successful

        Raises:
            ProviderError: If publish fails
        """
        if not self._producer:
            await self.initialize()

        try:
            # Serialize message
            if isinstance(message, (dict, list)):
                value = json.dumps(message).encode()
            elif isinstance(message, str):
                value = message.encode()
            elif isinstance(message, bytes):
                value = message
            else:
                value = str(message).encode()

            # Prepare key (using routing_key as the Kafka key)
            encoded_key: Optional[bytes] = None
            if routing_key:
                encoded_key = routing_key.encode()
            elif metadata and metadata.message_id:
                encoded_key = metadata.message_id.encode()

            # Prepare headers
            headers = []
            if metadata:
                metadata_dict = metadata.model_dump()
                for k, v in metadata_dict.items():
                    if v is not None:
                        if isinstance(v, (str, int, float, bool)):
                            headers.append((k, str(v).encode()))

            # Add timestamp as header
            timestamp = int(datetime.now().timestamp() * 1000)
            headers.append(("timestamp", str(timestamp).encode()))

            # Publish message
            if self._producer is None:
                raise RuntimeError("Producer not initialized")
            await self._producer.send_and_wait(
                topic=exchange,
                value=value,
                key=encoded_key,
                headers=headers
            )

            logger.debug(f"Published message to topic {exchange}")
            return True

        except Exception as e:
            raise ProviderError(
                message=f"Failed to publish message: {str(e)}",
                context=ErrorContext.create(
                    flow_name="kafka_provider",
                    error_type="publish_error",
                    error_location="KafkaMQProvider.publish",
                    component="kafka_provider",
                    operation="publish"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="publish",
                    retry_count=0
                ),
                cause=e
            )

    async def subscribe(self,
                       topic: str,
                       callback: Callable[[Any, MessageMetadata], Any],
                       group_id: Optional[str] = None,
                       auto_offset_reset: Optional[str] = None) -> None:
        """Subscribe to messages from Kafka.
        
        Args:
            topic: Topic to subscribe to
            callback: Callback function
            group_id: Optional consumer group ID (overrides settings)
            auto_offset_reset: Optional offset reset strategy (overrides settings)
            
        Raises:
            ProviderError: If subscribe fails
        """
        if not self._producer:  # We just need any connection
            await self.initialize()

        try:
            # Check if already subscribed
            if topic in self._consumers:
                raise ProviderError(
                    message=f"Already subscribed to topic {topic}",
                    context=ErrorContext.create(
                        flow_name="kafka_provider",
                        error_type="subscription_error",
                        error_location="KafkaMQProvider.subscribe",
                        component="kafka_provider",
                        operation="subscribe"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="mq",
                        operation="subscribe",
                        retry_count=0
                    )
                )

            # Get bootstrap servers
            bootstrap_servers = self.settings.bootstrap_servers
            if not bootstrap_servers:
                bootstrap_servers = f"{self.settings.host}:{self.settings.port}"

            # Use provided values or defaults from settings
            consumer_group_id = group_id or self.settings.group_id
            consumer_auto_offset_reset = auto_offset_reset or self.settings.auto_offset_reset

            # Prepare consumer config
            consumer_config = {
                "bootstrap_servers": bootstrap_servers,
                "group_id": consumer_group_id,
                "auto_offset_reset": consumer_auto_offset_reset,
                "enable_auto_commit": self.settings.enable_auto_commit,
                "auto_commit_interval_ms": self.settings.auto_commit_interval_ms,
                "security_protocol": self.settings.security_protocol,
                **self.settings.connect_args
            }

            # Add client ID if provided
            if self.settings.client_id:
                consumer_config["client_id"] = f"{self.settings.client_id}_consumer"

            # Add SASL settings if provided
            if self.settings.sasl_mechanism and self.settings.sasl_username and self.settings.sasl_password:
                consumer_config["sasl_mechanism"] = self.settings.sasl_mechanism
                consumer_config["sasl_plain_username"] = self.settings.sasl_username
                consumer_config["sasl_plain_password"] = self.settings.sasl_password

            # Add SSL settings if provided
            if self.settings.ssl_context:
                consumer_config["ssl_context"] = self.settings.ssl_context
                consumer_config["ssl_check_hostname"] = self.settings.ssl_check_hostname

            # Create consumer
            consumer = AIOKafkaConsumer(topic, **consumer_config)

            # Store consumer
            self._consumers[topic] = consumer

            # Define consumer task
            async def consume_task() -> None:
                try:
                    await consumer.start()
                    logger.info(f"Started consuming from topic {topic} with group ID {consumer_group_id}")

                    async for message in consumer:
                        try:
                            # Get message value
                            value = message.value

                            # Parse message based on content
                            try:
                                data = json.loads(value.decode())
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                data = value

                            # Extract metadata from headers
                            meta_dict = {
                                "topic": message.topic,
                                "partition": message.partition,
                                "offset": message.offset,
                                "timestamp": message.timestamp,
                                "message_id": message.key.decode() if message.key else None,
                                "routing_key": message.topic
                            }

                            # Add headers to metadata
                            if message.headers:
                                for key, value in message.headers:
                                    key_str = key.decode() if isinstance(key, bytes) else key
                                    try:
                                        value_str = value.decode() if isinstance(value, bytes) else value
                                        meta_dict[key_str] = value_str
                                    except UnicodeDecodeError:
                                        # Skip binary headers
                                        pass

                            # Create metadata object
                            meta = MessageMetadata(**meta_dict)

                            # Call callback
                            await callback(data, meta)

                        except Exception as e:
                            logger.error(f"Error processing Kafka message: {str(e)}")
                            # Continue consuming
                            continue

                except asyncio.CancelledError:
                    logger.info(f"Stopped consuming from topic {topic}")
                    await consumer.stop()

                except Exception as e:
                    logger.error(f"Error in Kafka consumer task: {str(e)}")
                    await consumer.stop()
                    # Remove from consumers
                    if topic in self._consumers:
                        del self._consumers[topic]
                    if topic in self._consumer_tasks:
                        del self._consumer_tasks[topic]

            # Start consumer task
            task = asyncio.create_task(consume_task())
            self._consumer_tasks[topic] = task

        except Exception as e:
            # Cleanup on error
            if topic in self._consumers:
                del self._consumers[topic]
            if topic in self._consumer_tasks:
                del self._consumer_tasks[topic]

            raise ProviderError(
                message=f"Failed to subscribe to topic: {str(e)}",
                context=ErrorContext.create(
                    flow_name="kafka_provider",
                    error_type="subscription_error",
                    error_location="KafkaMQProvider.subscribe",
                    component="kafka_provider",
                    operation="subscribe"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="subscribe",
                    retry_count=0
                ),
                cause=e
            )

    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            
        Raises:
            ProviderError: If unsubscribe fails
        """
        try:
            # Check if subscribed
            if topic not in self._consumers:
                raise ProviderError(
                    message=f"Not subscribed to topic {topic}",
                    context=ErrorContext.create(
                        flow_name="kafka_provider",
                        error_type="unsubscription_error",
                        error_location="KafkaMQProvider.unsubscribe",
                        component="kafka_provider",
                        operation="unsubscribe"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="mq",
                        operation="unsubscribe",
                        retry_count=0
                    )
                )

            # Get consumer and task
            consumer = self._consumers[topic]
            task = self._consumer_tasks.get(topic)

            # Cancel task
            if task and not task.done():
                task.cancel()

            # Stop consumer
            if consumer:
                await consumer.stop()

            # Remove from maps
            del self._consumers[topic]
            if topic in self._consumer_tasks:
                del self._consumer_tasks[topic]

            logger.info(f"Unsubscribed from topic {topic}")

        except Exception as e:
            if isinstance(e, ProviderError):
                raise e

            raise ProviderError(
                message=f"Failed to unsubscribe from topic: {str(e)}",
                context=ErrorContext.create(
                    flow_name="kafka_provider",
                    error_type="unsubscription_error",
                    error_location="KafkaMQProvider.unsubscribe",
                    component="kafka_provider",
                    operation="unsubscribe"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="unsubscribe",
                    retry_count=0
                ),
                cause=e
            )

    async def create_topic(self,
                          topic: str,
                          num_partitions: int = 1,
                          replication_factor: int = 1) -> None:
        """Create a Kafka topic.
        
        Note: This requires admin privileges and may not work in all Kafka deployments.
        
        Args:
            topic: Topic name
            num_partitions: Number of partitions
            replication_factor: Replication factor
            
        Raises:
            ProviderError: If topic creation fails
        """
        if not self._producer:
            await self.initialize()

        try:
            # Import kafka-python for admin operations
            try:
                from kafka.admin import KafkaAdminClient, NewTopic
            except ImportError:
                raise ProviderError(
                    message="kafka-python package not found. Install with 'pip install kafka-python'",
                    context=ErrorContext.create(
                        flow_name="kafka_provider",
                        error_type="dependency_error",
                        error_location="KafkaMQProvider.create_topic",
                        component="kafka_provider",
                        operation="create_topic"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="mq",
                        operation="create_topic",
                        retry_count=0
                    )
                )

            # Get bootstrap servers
            bootstrap_servers = self.settings.bootstrap_servers
            if not bootstrap_servers:
                bootstrap_servers = f"{self.settings.host}:{self.settings.port}"

            # Create admin client
            admin_config = {
                "bootstrap_servers": bootstrap_servers.split(",") if "," in bootstrap_servers else bootstrap_servers,
                "security_protocol": self.settings.security_protocol
            }

            # Add client ID if provided
            if self.settings.client_id:
                admin_config["client_id"] = f"{self.settings.client_id}_admin"

            # Add SASL settings if provided
            if self.settings.sasl_mechanism and self.settings.sasl_username and self.settings.sasl_password:
                admin_config["sasl_mechanism"] = self.settings.sasl_mechanism
                admin_config["sasl_plain_username"] = self.settings.sasl_username
                admin_config["sasl_plain_password"] = self.settings.sasl_password

            admin_client = KafkaAdminClient(**admin_config)

            # Create topic
            topic_list = [
                NewTopic(
                    name=topic,
                    num_partitions=num_partitions,
                    replication_factor=replication_factor
                )
            ]

            # Create topics
            admin_client.create_topics(new_topics=topic_list, validate_only=False)

            # Close admin client
            admin_client.close()

            logger.info(f"Created Kafka topic {topic} with {num_partitions} partitions and replication factor {replication_factor}")

        except Exception as e:
            # Check if topic already exists
            if hasattr(e, "__class__") and e.__class__.__name__ == "TopicAlreadyExistsError":
                logger.info(f"Kafka topic {topic} already exists")
                return

            raise ProviderError(
                message=f"Failed to create Kafka topic: {str(e)}",
                context=ErrorContext.create(
                    flow_name="kafka_provider",
                    error_type="topic_creation_error",
                    error_location="KafkaMQProvider.create_topic",
                    component="kafka_provider",
                    operation="create_topic"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="create_topic",
                    retry_count=0
                ),
                cause=e
            )

    async def delete_topic(self, topic: str) -> None:
        """Delete a Kafka topic.
        
        Note: This requires admin privileges and may not work in all Kafka deployments.
        
        Args:
            topic: Topic name
            
        Raises:
            ProviderError: If topic deletion fails
        """
        if not self._producer:
            await self.initialize()

        try:
            # Import kafka-python for admin operations
            try:
                from kafka.admin import KafkaAdminClient
            except ImportError:
                raise ProviderError(
                    message="kafka-python package not found. Install with 'pip install kafka-python'",
                    context=ErrorContext.create(
                        flow_name="kafka_provider",
                        error_type="dependency_error",
                        error_location="KafkaMQProvider.delete_topic",
                        component="kafka_provider",
                        operation="delete_topic"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="mq",
                        operation="delete_topic",
                        retry_count=0
                    )
                )

            # Get bootstrap servers
            bootstrap_servers = self.settings.bootstrap_servers
            if not bootstrap_servers:
                bootstrap_servers = f"{self.settings.host}:{self.settings.port}"

            # Create admin client
            admin_config = {
                "bootstrap_servers": bootstrap_servers.split(",") if "," in bootstrap_servers else bootstrap_servers,
                "security_protocol": self.settings.security_protocol
            }

            # Add client ID if provided
            if self.settings.client_id:
                admin_config["client_id"] = f"{self.settings.client_id}_admin"

            # Add SASL settings if provided
            if self.settings.sasl_mechanism and self.settings.sasl_username and self.settings.sasl_password:
                admin_config["sasl_mechanism"] = self.settings.sasl_mechanism
                admin_config["sasl_plain_username"] = self.settings.sasl_username
                admin_config["sasl_plain_password"] = self.settings.sasl_password

            admin_client = KafkaAdminClient(**admin_config)

            # Unsubscribe if subscribed
            if topic in self._consumers:
                await self.unsubscribe(topic)

            # Delete topic
            admin_client.delete_topics([topic])

            # Close admin client
            admin_client.close()

            logger.info(f"Deleted Kafka topic {topic}")

        except Exception as e:
            # Check if topic doesn't exist
            if hasattr(e, "__class__") and e.__class__.__name__ == "UnknownTopicOrPartitionError":
                logger.info(f"Kafka topic {topic} does not exist")
                return

            raise ProviderError(
                message=f"Failed to delete Kafka topic: {str(e)}",
                context=ErrorContext.create(
                    flow_name="kafka_provider",
                    error_type="topic_deletion_error",
                    error_location="KafkaMQProvider.delete_topic",
                    component="kafka_provider",
                    operation="delete_topic"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="delete_topic",
                    retry_count=0
                ),
                cause=e
            )
