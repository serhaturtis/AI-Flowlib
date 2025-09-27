"""RabbitMQ message queue provider implementation.

This module provides a concrete implementation of the MQProvider
for RabbitMQ messaging using aio-pika.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Callable
import uuid
from datetime import datetime

from pydantic import Field
from flowlib.core.errors.errors import ProviderError, ErrorContext
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.mq.base import MQProvider, MQProviderSettings, MessageMetadata
from flowlib.providers.core.decorators import provider
# Removed ProviderType import - using config-driven provider access

logger = logging.getLogger(__name__)

try:
    import aio_pika  # type: ignore[import-not-found]
    from aio_pika import Message
    from aio_pika.abc import AbstractIncomingMessage  # type: ignore[import-not-found]
except ImportError:
    logger.warning("aio_pika package not found. Install with 'pip install aio-pika'")


class RabbitMQProviderSettings(MQProviderSettings):
    """RabbitMQ provider settings - direct inheritance, only RabbitMQ-specific fields.
    
    RabbitMQ requires:
    1. Connection details (host, port, virtual_host)
    2. Authentication (username, password)
    3. Exchange and queue configuration
    4. SSL and reliability settings
    """
    
    # RabbitMQ-specific connection settings
    connection_string: Optional[str] = Field(default=None, description="Connection string (overrides host/port if provided)")
    connection_timeout: float = Field(default=10.0, description="Connection timeout in seconds")
    
    # SSL settings
    ssl: bool = Field(default=False, description="Whether to use SSL")
    ssl_options: Optional[Dict[str, Any]] = Field(default=None, description="SSL options")
    
    # Exchange and queue settings
    exchange_name: Optional[str] = Field(default=None, description="Exchange name")
    exchange_type: str = Field(default="topic", description="Exchange type (direct, fanout, topic, headers)")
    exchange_durable: bool = Field(default=True, description="Whether exchange is durable")
    queue_durable: bool = Field(default=True, description="Whether queues are durable")
    queue_auto_delete: bool = Field(default=False, description="Whether queues are auto-deleted")
    delivery_mode: int = Field(default=2, description="Delivery mode (1 = non-persistent, 2 = persistent)")
    
    # Additional connection arguments
    connect_args: Dict[str, Any] = Field(default_factory=dict, description="Additional connection arguments")


@provider(provider_type="message_queue", name="rabbitmq", settings_class=RabbitMQProviderSettings)
class RabbitMQProvider(MQProvider[RabbitMQProviderSettings]):
    """RabbitMQ implementation of the MQProvider.
    
    This provider implements message queue operations using aio_pika,
    an asynchronous client for RabbitMQ.
    """
    
    def __init__(self, name: str = "rabbitmq", settings: Optional[RabbitMQProviderSettings] = None):
        """Initialize RabbitMQ provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        if settings is None:
            settings = RabbitMQProviderSettings(
                host="localhost",
                port=5672,
                username="guest",
                password="guest"
            )
        super().__init__(name=name, settings=settings)
        if not isinstance(self.settings, RabbitMQProviderSettings):
            raise TypeError(f"settings must be a RabbitMQProviderSettings instance, got {type(self.settings)}")
        
        self._connection: Optional[aio_pika.Connection] = None
        self._channel: Optional[aio_pika.Channel] = None
        self._exchange: Optional[aio_pika.Exchange] = None
        self._queues: Dict[str, aio_pika.Queue] = {}
        self._consumer_tags: set[str] = set()
        self._tag_to_queue_map: Dict[str, str] = {}
        
    async def _initialize(self) -> None:
        """Initialize RabbitMQ connection.
        
        Raises:
            ProviderError: If initialization fails
        """
        try:
            # Create connection
            if self.settings.connection_string:
                # Use connection string if provided
                self._connection = await aio_pika.connect_robust(
                    self.settings.connection_string,
                    timeout=self.settings.connection_timeout,
                    **self.settings.connect_args
                )
            else:
                # Construct connection URL
                connection_url = f"amqp://{self.settings.username}:{self.settings.password}@{self.settings.host}:{self.settings.port}/{self.settings.virtual_host}"
                
                # Connect to RabbitMQ
                self._connection = await aio_pika.connect_robust(
                    connection_url,
                    timeout=self.settings.connection_timeout,
                    **self.settings.connect_args
                )
            
            # Create channel
            self._channel = await self._connection.channel()
            
            # Create exchange
            self._exchange = await self._channel.declare_exchange(
                name=self.settings.exchange_name or self.name,
                type=self.settings.exchange_type,
                durable=self.settings.exchange_durable
            )
            
            logger.info(f"Connected to RabbitMQ: {self.settings.host}:{self.settings.port}/{self.settings.virtual_host}")
            logger.debug(f"RabbitMQ channel successfully created: {self._channel}")
            
        except Exception as e:
            self._connection = None
            self._channel = None
            self._exchange = None
            raise ProviderError(
                message=f"Failed to connect to RabbitMQ: {str(e)}",
                context=ErrorContext.create(
                    flow_name="rabbitmq_provider",
                    error_type="ConnectionError",
                    error_location="_initialize",
                    component=self.name,
                    operation="connect"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="connect",
                    retry_count=0
                ),
                cause=e
            )
    
    async def _shutdown(self) -> None:
        """Close RabbitMQ connection."""
        try:
            # Cancel all consumers
            for consumer_tag in self._consumer_tags:
                try:
                    if self._channel:
                        await self._channel.cancel(consumer_tag)
                except Exception as e:
                    logger.warning(f"Error canceling consumer {consumer_tag}: {str(e)}")
            
            # Close connection
            if self._connection:
                await self._connection.close()
                self._connection = None
                self._channel = None
                self._exchange = None
                self._queues = {}
                self._consumer_tags = set()
                self._tag_to_queue_map = {}
                logger.info(f"Closed RabbitMQ connection: {self.settings.host}:{self.settings.port}")
        except Exception as e:
            logger.error(f"Error during RabbitMQ connection shutdown: {str(e)}")
    
    async def publish(self,
                     exchange: str,
                     routing_key: str,
                     message: Any,
                     metadata: Optional[MessageMetadata] = None) -> bool:
        """Publish a message to RabbitMQ.

        Args:
            exchange: Exchange name (uses default if empty)
            routing_key: Routing key
            message: Message to publish
            metadata: Optional message metadata

        Returns:
            True if publish was successful

        Raises:
            ProviderError: If publish fails
        """
        if not self._connection or not self._channel or not self._exchange:
            await self.initialize()
            
        try:
            # Serialize message
            if isinstance(message, (dict, list)):
                body = json.dumps(message).encode()
            elif isinstance(message, str):
                body = message.encode()
            elif isinstance(message, bytes):
                body = message
            else:
                body = str(message).encode()

            # Prepare metadata
            headers = {}
            content_type = "application/json"
            expiration = None
            if metadata:
                headers = metadata.model_dump()
                if metadata.content_type:
                    content_type = metadata.content_type
                expiration = metadata.expiration

            # Create message
            amqp_message = Message(
                body=body,
                content_type=content_type,
                delivery_mode=self.settings.delivery_mode,
                message_id=str(uuid.uuid4()),
                timestamp=int(datetime.now().timestamp()),
                expiration=str(expiration) if expiration else None,
                headers=headers
            )

            # Publish message
            if not self._exchange:
                raise ProviderError(
                    message="Exchange not initialized",
                    context=ErrorContext.create(
                        flow_name="rabbitmq_provider",
                        error_type="InitializationError",
                        error_location="publish",
                        component=self.name,
                        operation="message_publish"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="mq",
                        operation="message_publish",
                        retry_count=0
                    )
                )
            await self._exchange.publish(
                message=amqp_message,
                routing_key=routing_key
            )

            logger.debug(f"Published message to {routing_key} via exchange {exchange or 'default'}")
            return True
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to publish message: {str(e)}",
                context=ErrorContext.create(
                    flow_name="rabbitmq_provider",
                    error_type="PublishError",
                    error_location="publish",
                    component=self.name,
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
    
    async def consume(self,
                     queue: str,
                     callback: Callable[[Any, MessageMetadata], Any],
                     consumer_tag: Optional[str] = None) -> Any:
        """Consume messages from RabbitMQ.

        Args:
            queue: Queue name to consume from
            callback: Callback function (receives message body, metadata)
            consumer_tag: Optional consumer tag

        Returns:
            The consumer tag used for this consumer

        Raises:
            ProviderError: If subscribe fails
        """
        if not self._connection or not self._channel or not self._exchange:
            await self.initialize()

        try:
            # Declare queue
            if not self._channel:
                raise ProviderError(
                    message="Channel not initialized",
                    context=ErrorContext.create(
                        flow_name="rabbitmq_provider",
                        error_type="InitializationError",
                        error_location="consume",
                        component=self.name,
                        operation="queue_declaration"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="mq",
                        operation="queue_declaration",
                        retry_count=0
                    )
                )
            amqp_queue = await self._channel.declare_queue(
                name=queue,
                durable=self.settings.queue_durable,
                auto_delete=self.settings.queue_auto_delete
            )

            # Bind queue to exchange (use queue name as routing key)
            await amqp_queue.bind(self._exchange, routing_key=queue)

            # Store queue
            self._queues[queue] = amqp_queue
            
            # Define message handler
            async def message_handler(message: AbstractIncomingMessage) -> None:
                async with message.process(ignore_processed=True): # Use ignore_processed for safety
                    try:
                        # Get message body
                        body = message.body
                        
                        # Parse message based on content type
                        if message.content_type == "application/json":
                            try:
                                data = json.loads(body.decode())
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON, passing raw data. Msg ID: {message.message_id}")
                                data = body # Pass raw bytes if decode fails
                        else:
                            data = body
                        
                        # Extract metadata from aio_pika message
                        meta = MessageMetadata(
                            message_id=message.message_id,
                            correlation_id=message.correlation_id,
                            timestamp=message.timestamp.timestamp() if message.timestamp else None,
                            expiration=int(message.expiration) if message.expiration else None,
                            priority=message.priority,
                            content_type=message.content_type,
                            headers=dict(message.headers) if message.headers else {},
                            # Add delivery_tag for potential manual ack/nack outside callback
                            delivery_tag=message.delivery_tag 
                        )
                        
                        # Call the user-provided callback
                        await callback(data, meta)
                        
                        # Manual Ack needed (always manual in base interface)
                        await message.ack()

                    except Exception as e:
                        logger.exception(f"Error processing message (ID: {message.message_id}): {str(e)}")
                        # Manual Nack on error
                        try:
                            await message.nack(requeue=False)
                        except Exception as nack_err:
                            logger.error(f"Failed to NACK message (ID: {message.message_id}): {nack_err}")

            # Start consuming
            # Use provided consumer_tag or let aio_pika generate one
            tag = await amqp_queue.consume(message_handler, consumer_tag=consumer_tag, no_ack=False)
            self._consumer_tags.add(tag)
            # Store mapping from tag to queue name for cancellation
            self._tag_to_queue_map[tag] = queue 
            
            logger.info(f"Started consuming from queue '{queue}' with consumer tag '{tag}'")
            return tag  # Return the consumer tag
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to start consuming from queue '{queue}': {str(e)}",
                context=ErrorContext.create(
                    flow_name="rabbitmq_provider",
                    error_type="ConsumeError",
                    error_location="consume",
                    component=self.name,
                    operation="consume"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="consume",
                    retry_count=0
                ),
                cause=e
            )
    
    async def stop_consuming(self, consumer_tag: str) -> None:
        """Stop consuming messages associated with a specific consumer tag.
        
        Args:
            consumer_tag: The tag of the consumer to stop.
            
        Raises:
            ProviderError: If stopping the consumer fails.
        """
        if not self._connection or not self._channel:
            logger.warning(f"Cannot stop consumer '{consumer_tag}': Not connected to RabbitMQ")
            return 
            
        if consumer_tag not in self._consumer_tags:
             logger.warning(f"Consumer tag '{consumer_tag}' not found or already stopped.")
             # Clean up map just in case
             if consumer_tag in self._tag_to_queue_map:
                  del self._tag_to_queue_map[consumer_tag]
             return
             
        # Find the queue object associated with the tag
        queue_name = self._tag_to_queue_map.get(consumer_tag)
        if not queue_name:
             logger.error(f"Could not find queue name for consumer tag '{consumer_tag}'. Cannot cancel.")
             # Clean up tag set
             self._consumer_tags.discard(consumer_tag)
             return
             
        queue = self._queues.get(queue_name)
        if not queue:
            logger.error(f"Could not find queue object for name '{queue_name}' (tag: '{consumer_tag}'). Cannot cancel.")
             # Clean up maps
            self._consumer_tags.discard(consumer_tag)
            del self._tag_to_queue_map[consumer_tag]
            return
            
        try:
            # Call cancel on the queue object, passing the tag
            await queue.cancel(consumer_tag)
            logger.info(f"Stopped consuming for consumer tag '{consumer_tag}' on queue '{queue_name}'")
        except Exception as e:
            logger.error(f"Error cancelling consumer tag '{consumer_tag}' on queue '{queue_name}': {e}", exc_info=True)
            # Still raise ProviderError as cancellation failed
            raise ProviderError(
                message=f"Failed to stop consumer '{consumer_tag}': {str(e)}",
                context=ErrorContext.create(
                    flow_name="rabbitmq_provider",
                    error_type="ConsumerStopError",
                    error_location="stop_consuming",
                    component=self.name,
                    operation="stop_consuming"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="stop_consuming",
                    retry_count=0
                ),
                cause=e
            )
        finally:
             # Clean up regardless of success or failure of cancel call
            self._consumer_tags.discard(consumer_tag)
            if consumer_tag in self._tag_to_queue_map:
                 del self._tag_to_queue_map[consumer_tag]
    
    async def create_queue(self, 
                          queue_name: str, 
                          routing_keys: Optional[List[str]] = None,
                          durable: Optional[bool] = None,
                          auto_delete: Optional[bool] = None) -> None:
        """Create a queue and optionally bind it to routing keys.
        
        Args:
            queue_name: Queue name
            routing_keys: Optional list of routing keys to bind
            durable: Whether queue is durable (overrides settings)
            auto_delete: Whether queue is auto-deleted (overrides settings)
            
        Raises:
            ProviderError: If queue creation fails
        """
        if not self._connection or not self._channel or not self._exchange:
            await self.initialize()
            
        try:
            # Use provided values or defaults from settings
            queue_durable = durable if durable is not None else self.settings.queue_durable
            queue_auto_delete = auto_delete if auto_delete is not None else self.settings.queue_auto_delete
            
            # Declare queue
            if not self._channel:
                raise ProviderError(
                    message="Channel not initialized",
                    context=ErrorContext.create(
                        flow_name="rabbitmq_provider",
                        error_type="InitializationError",
                        error_location="create_queue",
                        component=self.name,
                        operation="queue_declaration"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="mq",
                        operation="queue_declaration",
                        retry_count=0
                    )
                )
            queue = await self._channel.declare_queue(
                name=queue_name,
                durable=queue_durable,
                auto_delete=queue_auto_delete
            )
            
            # Bind queue to exchange with routing keys
            if routing_keys:
                for routing_key in routing_keys:
                    await queue.bind(self._exchange, routing_key=routing_key)
            
            # Store queue
            self._queues[queue_name] = queue
            
            logger.info(f"Created queue {queue_name}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to create queue: {str(e)}",
                context=ErrorContext.create(
                    flow_name="rabbitmq_provider",
                    error_type="QueueCreationError",
                    error_location="create_queue",
                    component=self.name,
                    operation="create_queue"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="create_queue",
                    retry_count=0
                ),
                cause=e
            )
    
    async def delete_queue(self, queue_name: str) -> None:
        """Delete a queue.
        
        Args:
            queue_name: Queue name
            
        Raises:
            ProviderError: If queue deletion fails
        """
        if not self._connection or not self._channel:
            await self.initialize()
            
        try:
            # Delete queue
            if not self._channel:
                raise ProviderError(
                    message="Channel not initialized",
                    context=ErrorContext.create(
                        flow_name="rabbitmq_provider",
                        error_type="InitializationError",
                        error_location="delete_queue",
                        component=self.name,
                        operation="queue_deletion"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="mq",
                        operation="queue_deletion",
                        retry_count=0
                    )
                )
            await self._channel.queue_delete(queue_name)
            
            # Remove queue from tracked queues
            if queue_name in self._queues:
                del self._queues[queue_name]
            
            logger.info(f"Deleted queue {queue_name}")
            
        except Exception as e:
            raise ProviderError(
                message=f"Failed to delete queue: {str(e)}",
                context=ErrorContext.create(
                    flow_name="rabbitmq_provider",
                    error_type="QueueDeletionError",
                    error_location="delete_queue",
                    component=self.name,
                    operation="delete_queue"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="mq",
                    operation="delete_queue",
                    retry_count=0
                ),
                cause=e
            ) 