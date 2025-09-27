"""Message Queue provider base class and related functionality.

This module provides the base class for implementing message queue providers
that share common functionality for publishing and consuming messages.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Callable, Union, Generic
from pydantic import BaseModel, ConfigDict, Field

from flowlib.core.errors.errors import ProviderError, ErrorContext
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.base import Provider, ProviderSettings
# Removed ProviderType import - using config-driven provider access

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class MQProviderSettings(ProviderSettings):
    """Base settings for message queue providers.
    
    Attributes:
        host: Message queue host address
        port: Message queue port
        username: Authentication username
        password: Authentication password (should use SecretStr in implementations)
        virtual_host: Virtual host/namespace
        timeout: Connection timeout in seconds
        heartbeat: Heartbeat interval in seconds
        ssl_enabled: Whether to use SSL for connections
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    # Connection settings
    host: str = Field(default="localhost", description="Message queue server host")
    port: int = Field(default=5672, description="Message queue server port")
    username: Optional[str] = Field(default=None, description="Username for authentication")
    password: Optional[str] = Field(default=None, description="Password for authentication (use SecretStr in implementations)")
    virtual_host: str = Field(default="/", description="Virtual host path")
    
    # Performance settings
    timeout: float = Field(default=30.0, description="Connection timeout in seconds")
    heartbeat: int = Field(default=60, description="Heartbeat interval in seconds")
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS encryption")
    
    # Retry settings
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    
    # Delivery settings
    prefetch_count: int = Field(default=10, description="Number of messages to prefetch")
    auto_ack: bool = Field(default=False, description="Automatically acknowledge messages")


class MessageMetadata(BaseModel):
    """Metadata for message queue messages.
    
    Attributes:
        message_id: Unique message identifier
        correlation_id: Correlation identifier for related messages
        timestamp: Message creation timestamp
        expiration: Message expiration time (if any)
        priority: Message priority (0-9, higher is more important)
        content_type: MIME type of message content
        headers: Custom message headers
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: Optional[Union[int, float]] = None
    expiration: Optional[int] = None
    priority: Optional[int] = None
    content_type: Optional[str] = None
    headers: Dict[str, Any] = {}
    
    # Additional provider-specific fields
    delivery_tag: Optional[Any] = None
    routing_key: Optional[str] = None
    topic: Optional[str] = None
    partition: Optional[int] = None
    offset: Optional[int] = None


SettingsT = TypeVar('SettingsT', bound=MQProviderSettings)


class MQProvider(Provider[SettingsT], Generic[SettingsT]):
    """Base class for message queue providers.
    
    Inherits from Provider and defines the MQ-specific interface.
    """
    
    def __init__(self, name: str = "mq", settings: Optional[SettingsT] = None):
        """Initialize message queue provider.
        
        Args:
            name: Unique provider name
            settings: Provider settings instance
        """
        # Pass provider_type="mq" and settings to the Provider base class
        super().__init__(name=name, settings=settings, provider_type="message_queue")
        # Remove redundant attribute setting, handled by Provider base
        # self._initialized = False
        # self._connection = None 
        
    # Removed incorrect initialize/shutdown methods that overrode Provider base methods
    # @property
    # def initialized(self) -> bool: ...
    # async def initialize(self): ...
    # async def shutdown(self): ...
        
    # Keep abstract methods defining the MQ interface
    async def publish(self, 
                     exchange: str, 
                     routing_key: str, 
                     message: Any,
                     metadata: Optional[MessageMetadata] = None) -> bool:
        raise NotImplementedError("Subclasses must implement publish()")
        
    async def consume(self,
                     queue: str,
                     callback: Callable[[Any, MessageMetadata], Any],
                     consumer_tag: Optional[str] = None) -> Any:
        raise NotImplementedError("Subclasses must implement consume()")
        
    async def acknowledge(self, delivery_tag: Any) -> None:
        raise NotImplementedError("Subclasses must implement acknowledge()")
        
    async def reject(self, delivery_tag: Any, requeue: bool = True) -> None:
        raise NotImplementedError("Subclasses must implement reject()")
        
    async def check_connection(self) -> bool:
        raise NotImplementedError("Subclasses must implement check_connection()")

    async def stop_consuming(self, consumer_tag: Any) -> None:
        """Stop consuming messages for the given consumer tag.

        Args:
            consumer_tag: Consumer identifier to stop

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement stop_consuming()")

    # Keep consume_structured and publish_structured as they provide useful default logic
    async def publish_structured(self,
                                exchange: str,
                                routing_key: str,
                                message: BaseModel,
                                metadata: Optional[MessageMetadata] = None) -> bool:
        """Publish a structured message to a queue.
        
        Args:
            exchange: Exchange name
            routing_key: Routing key or queue name
            message: Pydantic model instance
            metadata: Optional message metadata
            
        Returns:
            True if message was published successfully
            
        Raises:
            ProviderError: If publishing fails
        """
        try:
            message_dict = message.model_dump() # Use model_dump instead of dict
            
            # Set content type in metadata if not provided
            if metadata is None:
                metadata = MessageMetadata(content_type="application/json")
            elif metadata.content_type is None:
                metadata.content_type = "application/json"
                
            # Publish the message
            return await self.publish(exchange, routing_key, message_dict, metadata)
            
        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to publish structured message: {str(e)}",
                context=ErrorContext.create(
                    flow_name=f"mq_publish_{exchange}",
                    error_type="publish_error",
                    error_location=f"{self.__class__.__name__}.publish_structured",
                    component=self.name,
                    operation=f"publish_to_{exchange}/{routing_key}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation=f"publish_structured({exchange}, {routing_key})",
                    retry_count=0
                ),
                cause=e
            )
            
    async def consume_structured(self,
                               queue: str,
                               output_type: Type[T],
                               callback: Callable[[T, MessageMetadata], Any],
                               consumer_tag: Optional[str] = None) -> Any:
        """Start consuming structured messages from a queue.
        
        Args:
            queue: Queue name to consume from
            output_type: Pydantic model for message structure
            callback: Function to call for each parsed message
            consumer_tag: Optional consumer identifier
            
        Returns:
            Consumer instance (implementation-specific)
            
        Raises:
            ProviderError: If consumer creation fails
        """
        try:
            # Create a wrapper callback that parses messages
            async def wrapper_callback(message: Any, metadata: MessageMetadata) -> Any:
                try:
                    # Parse the message into the output type
                    parsed_message = output_type.model_validate(message) # Use model_validate
                    
                    # Call the original callback with the parsed message
                    return await callback(parsed_message, metadata)
                    
                except Exception as e:
                    logger.error(f"Failed to parse message as {output_type.__name__}: {str(e)}")
                    # Handle parsing errors according to implementation
                    
            # Start consuming with the wrapper callback
            return await self.consume(queue, wrapper_callback, consumer_tag)
            
        except Exception as e:
            # Wrap and re-raise errors with context
            raise ProviderError(
                message=f"Failed to create structured consumer: {str(e)}",
                context=ErrorContext.create(
                    flow_name=f"mq_consume_{queue}",
                    error_type="consume_error",
                    error_location=f"{self.__class__.__name__}.consume_structured",
                    component=self.name,
                    operation=f"consume_from_{queue}"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation=f"consume_structured({queue})",
                    retry_count=0
                ),
                cause=e
            ) 