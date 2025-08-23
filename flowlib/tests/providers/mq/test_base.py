"""Tests for MQ provider base functionality."""

import asyncio
import pytest
import pytest_asyncio
from typing import Optional, Any, Callable
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from pydantic import BaseModel

from flowlib.providers.mq.base import (
    MQProvider,
    MQProviderSettings,
    MessageMetadata
)
from flowlib.core.errors.errors import ProviderError


class TestMQProviderSettings:
    """Test MQProviderSettings model."""
    
    def test_settings_creation_minimal(self):
        """Test settings creation with minimal required data."""
        settings = MQProviderSettings(
            host="localhost",
            port=5672,
            username="guest",
            password="guest"
        )
        
        assert settings.host == "localhost"
        assert settings.port == 5672
        assert settings.username == "guest"
        assert settings.password == "guest"
        assert settings.virtual_host == "/"
        assert settings.timeout == 30.0
        assert settings.heartbeat == 60
        assert settings.ssl_enabled is False
        assert settings.max_retries == 3
        assert settings.retry_delay == 1.0
        assert settings.prefetch_count == 10
        assert settings.auto_ack is False
        # Inherited fields
        assert settings.timeout_seconds == 60.0
    
    def test_settings_creation_full(self):
        """Test settings creation with all fields."""
        settings = MQProviderSettings(
            host="rabbitmq.example.com",
            port=5671,
            username="admin",
            password="secret123",
            virtual_host="/production",
            timeout=60.0,
            heartbeat=120,
            ssl_enabled=True,
            max_retries=5,
            retry_delay=2.0,
            prefetch_count=20,
            auto_ack=True,
            timeout_seconds=90.0,
            api_key="test-key"
        )
        
        assert settings.host == "rabbitmq.example.com"
        assert settings.port == 5671
        assert settings.username == "admin"
        assert settings.password == "secret123"
        assert settings.virtual_host == "/production"
        assert settings.timeout == 60.0
        assert settings.heartbeat == 120
        assert settings.ssl_enabled is True
        assert settings.max_retries == 5
        assert settings.retry_delay == 2.0
        assert settings.prefetch_count == 20
        assert settings.auto_ack is True
        assert settings.timeout_seconds == 90.0
        assert settings.api_key == "test-key"
    
    def test_settings_inheritance(self):
        """Test that MQProviderSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        settings = MQProviderSettings(
            host="localhost",
            port=5672,
            username="guest",
            password="guest"
        )
        assert isinstance(settings, ProviderSettings)
    
    def test_settings_merge(self):
        """Test settings merge functionality."""
        base_settings = MQProviderSettings(
            host="localhost",
            port=5672,
            username="guest",
            password="guest",
            api_key="base-key"
        )
        
        override_settings = MQProviderSettings(
            host="remote.example.com",
            port=5671,
            username="admin",
            password="admin",
            ssl_enabled=True
        )
        
        merged = base_settings.merge(override_settings)
        
        # Should have overridden values
        assert merged.host == "remote.example.com"
        assert merged.port == 5671
        assert merged.username == "admin"
        assert merged.password == "admin"
        assert merged.ssl_enabled is True
        # Should preserve original api_key since override didn't have it
        assert merged.api_key == "base-key"
    
    def test_settings_with_overrides(self):
        """Test with_overrides method."""
        settings = MQProviderSettings(
            host="localhost",
            port=5672,
            username="guest",
            password="guest"
        )
        
        overridden = settings.with_overrides(
            host="production.example.com",
            ssl_enabled=True,
            max_retries=10
        )
        
        assert overridden.host == "production.example.com"
        assert overridden.ssl_enabled is True
        assert overridden.max_retries == 10
        # Original values preserved
        assert overridden.port == 5672
        assert overridden.username == "guest"


class TestMessageMetadata:
    """Test MessageMetadata model."""
    
    def test_metadata_creation_minimal(self):
        """Test metadata creation with minimal data."""
        metadata = MessageMetadata()
        
        assert metadata.message_id is None
        assert metadata.correlation_id is None
        assert metadata.timestamp is None
        assert metadata.expiration is None
        assert metadata.priority is None
        assert metadata.content_type is None
        assert metadata.headers == {}
    
    def test_metadata_creation_full(self):
        """Test metadata creation with all fields."""
        headers = {"custom_header": "value", "app_id": "test_app"}
        timestamp = int(datetime.now().timestamp())
        
        metadata = MessageMetadata(
            message_id="msg_123",
            correlation_id="corr_456",
            timestamp=timestamp,
            expiration=3600,
            priority=5,
            content_type="application/json",
            headers=headers
        )
        
        assert metadata.message_id == "msg_123"
        assert metadata.correlation_id == "corr_456"
        assert metadata.timestamp == timestamp
        assert metadata.expiration == 3600
        assert metadata.priority == 5
        assert metadata.content_type == "application/json"
        assert metadata.headers == headers
    
    def test_metadata_validation(self):
        """Test metadata field validation."""
        # Valid priority range (typically 0-9)
        metadata = MessageMetadata(priority=9)
        assert metadata.priority == 9
        
        # Headers should be a dictionary
        metadata = MessageMetadata(headers={"key": "value"})
        assert metadata.headers == {"key": "value"}
    
    def test_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = MessageMetadata(
            message_id="test_123",
            content_type="text/plain",
            headers={"source": "test"}
        )
        
        data = metadata.model_dump()
        assert data["message_id"] == "test_123"
        assert data["content_type"] == "text/plain"
        assert data["headers"] == {"source": "test"}
        
        # Should include None values
        assert "correlation_id" in data
        assert data["correlation_id"] is None


class TestMQProvider:
    """Test MQProvider base class."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return MQProviderSettings(
            host="localhost",
            port=5672,
            username="test",
            password="test",
            virtual_host="/test"
        )
    
    @pytest_asyncio.fixture
    def provider(self, provider_settings):
        """Create test provider instance."""
        return MockMQProvider(settings=provider_settings)
    
    def test_provider_initialization(self, provider, provider_settings):
        """Test provider initialization."""
        assert provider.name == "mock-mq"
        assert provider.provider_type == "message_queue"
        assert provider.settings == provider_settings
        assert not provider._initialized
    
    def test_provider_inheritance(self, provider):
        """Test provider inheritance from base Provider class."""
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
        assert isinstance(provider, MQProvider)
    
    def test_provider_default_settings(self):
        """Test provider with default settings."""
        # Should work with default name
        provider = MockMQProvider()
        assert provider.name == "mock-mq"
        assert provider.provider_type == "message_queue"
    
    @pytest.mark.asyncio
    async def test_provider_lifecycle(self, provider):
        """Test provider initialization and shutdown lifecycle."""
        # Initially not initialized
        assert not provider._initialized
        
        # Initialize
        await provider.initialize()
        assert provider._initialized
        assert provider.initialize_called
        
        # Shutdown
        await provider.shutdown()
        assert not provider._initialized
        assert provider.shutdown_called
    
    @pytest.mark.asyncio
    async def test_abstract_methods_not_implemented(self, provider_settings):
        """Test that abstract methods raise NotImplementedError."""
        provider = MQProvider(settings=provider_settings)
        
        with pytest.raises(NotImplementedError):
            await provider.publish("exchange", "routing_key", "message")
        
        with pytest.raises(NotImplementedError):
            await provider.consume("queue", lambda x, y: None)
        
        with pytest.raises(NotImplementedError):
            await provider.acknowledge("delivery_tag")
        
        with pytest.raises(NotImplementedError):
            await provider.reject("delivery_tag")
        
        with pytest.raises(NotImplementedError):
            await provider.check_connection()


class MockMQProvider(MQProvider):
    """Mock MQ provider for testing."""
    
    def __init__(self, name: str = "mock-mq", settings: Optional[MQProviderSettings] = None):
        if settings is None:
            settings = MQProviderSettings(
                host="localhost",
                port=5672,
                username="guest",
                password="guest"
            )
        super().__init__(name=name, settings=settings)
        
        # Store tracking attributes in private attributes to avoid Pydantic validation
        object.__setattr__(self, '_initialize_called', False)
        object.__setattr__(self, '_shutdown_called', False)
        object.__setattr__(self, '_publish_calls', [])
        object.__setattr__(self, '_consume_calls', [])
        object.__setattr__(self, '_acknowledge_calls', [])
        object.__setattr__(self, '_reject_calls', [])
        object.__setattr__(self, '_connection_status', True)
    
    @property
    def initialize_called(self):
        return self._initialize_called
    
    @property
    def shutdown_called(self):
        return self._shutdown_called
    
    @property
    def publish_calls(self):
        return self._publish_calls
    
    @property
    def consume_calls(self):
        return self._consume_calls
    
    @property
    def acknowledge_calls(self):
        return self._acknowledge_calls
    
    @property
    def reject_calls(self):
        return self._reject_calls
    
    @property
    def connection_status(self):
        return self._connection_status
    
    @connection_status.setter
    def connection_status(self, value):
        object.__setattr__(self, '_connection_status', value)
    
    async def _initialize(self):
        """Mock initialization."""
        object.__setattr__(self, '_initialize_called', True)
    
    async def _shutdown(self):
        """Mock shutdown."""
        object.__setattr__(self, '_shutdown_called', True)
    
    async def publish(self, exchange: str, routing_key: str, message: Any, metadata: Optional[MessageMetadata] = None) -> bool:
        """Mock publish method."""
        self._publish_calls.append({
            "exchange": exchange,
            "routing_key": routing_key,
            "message": message,
            "metadata": metadata
        })
        return True
    
    async def consume(self, queue: str, callback: Callable[[Any, MessageMetadata], Any], consumer_tag: Optional[str] = None) -> Any:
        """Mock consume method."""
        self._consume_calls.append({
            "queue": queue,
            "callback": callback,
            "consumer_tag": consumer_tag
        })
        return f"consumer_{queue}"
    
    async def acknowledge(self, delivery_tag: Any):
        """Mock acknowledge method."""
        self._acknowledge_calls.append(delivery_tag)
    
    async def reject(self, delivery_tag: Any, requeue: bool = True):
        """Mock reject method."""
        self._reject_calls.append({
            "delivery_tag": delivery_tag,
            "requeue": requeue
        })
    
    async def check_connection(self) -> bool:
        """Mock connection check."""
        return self._connection_status


class TestMQProviderStructuredMessaging:
    """Test structured messaging functionality."""
    
    @pytest_asyncio.fixture
    def provider(self):
        """Create mock provider."""
        return MockMQProvider()
    
    @pytest.mark.asyncio
    async def test_publish_structured_success(self, provider):
        """Test successful structured message publishing."""
        # Define a test model
        class TestMessage(BaseModel):
            id: int
            name: str
            active: bool = True
        
        message = TestMessage(id=123, name="test_item")
        metadata = MessageMetadata(
            message_id="msg_123",
            content_type="application/json"
        )
        
        result = await provider.publish_structured("test_exchange", "test.route", message, metadata)
        
        assert result is True
        assert len(provider.publish_calls) == 1
        
        call = provider.publish_calls[0]
        assert call["exchange"] == "test_exchange"
        assert call["routing_key"] == "test.route"
        assert call["message"] == {"id": 123, "name": "test_item", "active": True}
        assert call["metadata"].content_type == "application/json"
        assert call["metadata"].message_id == "msg_123"
    
    @pytest.mark.asyncio
    async def test_publish_structured_auto_content_type(self, provider):
        """Test structured publishing with automatic content type."""
        class TestMessage(BaseModel):
            value: str
        
        message = TestMessage(value="test")
        
        result = await provider.publish_structured("exchange", "route", message)
        
        assert result is True
        call = provider.publish_calls[0]
        assert call["metadata"].content_type == "application/json"
    
    @pytest.mark.asyncio
    async def test_publish_structured_preserves_content_type(self, provider):
        """Test that explicit content type is preserved."""
        class TestMessage(BaseModel):
            data: str
        
        message = TestMessage(data="xml_data")
        metadata = MessageMetadata(content_type="application/xml")
        
        await provider.publish_structured("exchange", "route", message, metadata)
        
        call = provider.publish_calls[0]
        assert call["metadata"].content_type == "application/xml"
    
    @pytest.mark.asyncio
    async def test_publish_structured_error_handling(self, provider):
        """Test error handling in structured publishing."""
        # Make publish fail
        async def failing_publish(*args, **kwargs):
            raise Exception("Publish failed")
        
        # Store original publish method and replace it using object.__setattr__
        original_publish = provider.publish
        object.__setattr__(provider, 'publish', failing_publish)
        
        try:
            class TestMessage(BaseModel):
                id: int
            
            message = TestMessage(id=1)
            
            with pytest.raises(ProviderError, match="Failed to publish structured message"):
                await provider.publish_structured("exchange", "route", message)
        finally:
            # Restore original method
            object.__setattr__(provider, 'publish', original_publish)
    
    @pytest.mark.asyncio
    async def test_consume_structured_success(self, provider):
        """Test successful structured message consumption."""
        class TestMessage(BaseModel):
            id: int
            name: str
        
        callback_calls = []
        
        async def test_callback(message: TestMessage, metadata: MessageMetadata):
            callback_calls.append({
                "message": message,
                "metadata": metadata
            })
            return "processed"
        
        result = await provider.consume_structured("test_queue", TestMessage, test_callback)
        
        assert result == "consumer_test_queue"
        assert len(provider.consume_calls) == 1
        
        call = provider.consume_calls[0]
        assert call["queue"] == "test_queue"
        assert call["consumer_tag"] is None
        
        # Test the wrapper callback
        wrapper_callback = call["callback"]
        test_data = {"id": 123, "name": "test"}
        test_metadata = MessageMetadata(message_id="test_msg")
        
        await wrapper_callback(test_data, test_metadata)
        
        assert len(callback_calls) == 1
        callback_call = callback_calls[0]
        assert isinstance(callback_call["message"], TestMessage)
        assert callback_call["message"].id == 123
        assert callback_call["message"].name == "test"
        assert callback_call["metadata"] == test_metadata
    
    @pytest.mark.asyncio
    async def test_consume_structured_with_consumer_tag(self, provider):
        """Test structured consumption with consumer tag."""
        class TestMessage(BaseModel):
            value: str
        
        async def test_callback(message: TestMessage, metadata: MessageMetadata):
            pass
        
        await provider.consume_structured("queue", TestMessage, test_callback, "custom_tag")
        
        call = provider.consume_calls[0]
        assert call["consumer_tag"] == "custom_tag"
    
    @pytest.mark.asyncio
    async def test_consume_structured_parsing_error(self, provider):
        """Test handling of message parsing errors in structured consumption."""
        class TestMessage(BaseModel):
            required_field: str
        
        callback_calls = []
        
        async def test_callback(message: TestMessage, metadata: MessageMetadata):
            callback_calls.append(message)
        
        await provider.consume_structured("queue", TestMessage, test_callback)
        
        # Get the wrapper callback
        wrapper_callback = provider.consume_calls[0]["callback"]
        
        # Test with invalid data (missing required field)
        invalid_data = {"wrong_field": "value"}
        test_metadata = MessageMetadata()
        
        # Should not raise exception, but should log error
        await wrapper_callback(invalid_data, test_metadata)
        
        # Original callback should not have been called
        assert len(callback_calls) == 0
    
    @pytest.mark.asyncio
    async def test_consume_structured_error_handling(self, provider):
        """Test error handling in structured consumption setup."""
        # Make consume fail
        async def failing_consume(*args, **kwargs):
            raise Exception("Consume failed")
        
        # Store original consume method and replace it using object.__setattr__
        original_consume = provider.consume
        object.__setattr__(provider, 'consume', failing_consume)
        
        try:
            class TestMessage(BaseModel):
                id: int
            
            async def test_callback(message: TestMessage, metadata: MessageMetadata):
                pass
            
            with pytest.raises(ProviderError, match="Failed to create structured consumer"):
                await provider.consume_structured("queue", TestMessage, test_callback)
        finally:
            # Restore original method
            object.__setattr__(provider, 'consume', original_consume)


class TestMQProviderAdvancedFeatures:
    """Test advanced MQ provider features."""
    
    @pytest_asyncio.fixture
    def provider(self):
        """Create mock provider."""
        return MockMQProvider()
    
    @pytest.mark.asyncio
    async def test_basic_operations(self, provider):
        """Test basic MQ operations."""
        # Initialize provider
        await provider.initialize()
        
        # Test publish
        metadata = MessageMetadata(
            message_id="test_123",
            priority=5,
            headers={"source": "test"}
        )
        result = await provider.publish("test_exchange", "test.key", "test message", metadata)
        assert result is True
        
        # Test consume
        async def message_handler(message, meta):
            return f"handled: {message}"
        
        consumer = await provider.consume("test_queue", message_handler, "consumer_1")
        assert consumer == "consumer_test_queue"
        
        # Test acknowledge
        await provider.acknowledge("delivery_123")
        assert "delivery_123" in provider.acknowledge_calls
        
        # Test reject
        await provider.reject("delivery_456", requeue=False)
        reject_call = provider.reject_calls[0]
        assert reject_call["delivery_tag"] == "delivery_456"
        assert reject_call["requeue"] is False
        
        # Test connection check
        status = await provider.check_connection()
        assert status is True
    
    @pytest.mark.asyncio
    async def test_connection_failure_simulation(self, provider):
        """Test connection failure handling."""
        # Simulate connection failure
        object.__setattr__(provider, 'connection_status', False)
        
        status = await provider.check_connection()
        assert status is False
    
    @pytest.mark.asyncio
    async def test_metadata_handling(self, provider):
        """Test comprehensive metadata handling."""
        current_time = int(datetime.now().timestamp())
        
        metadata = MessageMetadata(
            message_id="comprehensive_test",
            correlation_id="corr_123",
            timestamp=current_time,
            expiration=3600,
            priority=8,
            content_type="application/json",
            headers={
                "source_service": "test_service",
                "version": "1.0",
                "retry_count": "0"
            }
        )
        
        await provider.publish("exchange", "routing.key", {"data": "test"}, metadata)
        
        call = provider.publish_calls[0]
        published_metadata = call["metadata"]
        
        assert published_metadata.message_id == "comprehensive_test"
        assert published_metadata.correlation_id == "corr_123"
        assert published_metadata.timestamp == current_time
        assert published_metadata.expiration == 3600
        assert published_metadata.priority == 8
        assert published_metadata.content_type == "application/json"
        assert published_metadata.headers["source_service"] == "test_service"
        assert published_metadata.headers["version"] == "1.0"
        assert published_metadata.headers["retry_count"] == "0"
    
    @pytest.mark.asyncio
    async def test_complex_message_types(self, provider):
        """Test handling of complex message types."""
        # Test dictionary message
        dict_message = {"key": "value", "nested": {"inner": "data"}}
        await provider.publish("exchange", "dict.route", dict_message)
        
        # Test list message
        list_message = [1, 2, {"item": "data"}]
        await provider.publish("exchange", "list.route", list_message)
        
        # Test string message
        string_message = "simple string message"
        await provider.publish("exchange", "string.route", string_message)
        
        # Test bytes message (simulated)
        bytes_message = b"binary data"
        await provider.publish("exchange", "bytes.route", bytes_message)
        
        assert len(provider.publish_calls) == 4
        
        # Verify each message type was handled
        calls = provider.publish_calls
        assert calls[0]["message"] == dict_message
        assert calls[1]["message"] == list_message
        assert calls[2]["message"] == string_message
        assert calls[3]["message"] == bytes_message


class TestMQProviderErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_settings(self):
        """Test provider with invalid settings."""
        # Missing required fields should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            MQProviderSettings()
    
    @pytest.mark.asyncio
    async def test_operations_without_initialization(self):
        """Test operations work after explicit initialization."""
        provider = MockMQProvider()
        
        # Initialize the provider first
        await provider.initialize()
        
        # Operations should work after initialization
        result = await provider.publish("exchange", "key", "message")
        assert result is True
        assert provider.initialize_called
    
    @pytest.mark.asyncio
    async def test_concurrent_initialization(self):
        """Test concurrent initialization is handled safely."""
        provider = MockMQProvider()
        
        # Start multiple initialization tasks concurrently
        tasks = [provider.initialize() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Should only initialize once
        assert provider._initialized
        assert provider.initialize_called
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown handling."""
        provider = MockMQProvider()
        await provider.initialize()
        
        # Should shutdown cleanly
        await provider.shutdown()
        assert not provider._initialized
        assert provider.shutdown_called
        
        # Multiple shutdowns should be safe
        await provider.shutdown()
        assert not provider._initialized
    
    @pytest.mark.asyncio
    async def test_provider_type_validation(self):
        """Test provider type is correctly set."""
        provider = MockMQProvider()
        assert provider.provider_type == "message_queue"
    
    @pytest.mark.asyncio
    async def test_settings_defaults(self):
        """Test that provider uses sensible defaults."""
        settings = MQProviderSettings(
            host="localhost",
            port=5672,
            username="guest",
            password="guest"
        )
        
        assert settings.virtual_host == "/"
        assert settings.timeout == 30.0
        assert settings.heartbeat == 60
        assert settings.ssl_enabled is False
        assert settings.max_retries == 3
        assert settings.retry_delay == 1.0
        assert settings.prefetch_count == 10
        assert settings.auto_ack is False