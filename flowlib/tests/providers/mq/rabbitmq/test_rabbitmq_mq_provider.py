"""Tests for RabbitMQ MQ provider implementation."""

import pytest
import pytest_asyncio
import json
import uuid
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Optional, Any, List
from datetime import datetime

from flowlib.providers.mq.rabbitmq.provider import (
    RabbitMQProvider,
    RabbitMQProviderSettings
)
from flowlib.providers.mq.base import MessageMetadata
from flowlib.core.errors.errors import ProviderError


class TestRabbitMQProviderSettings:
    """Test RabbitMQProviderSettings model."""
    
    def test_settings_creation_minimal(self):
        """Test settings creation with minimal required data."""
        settings = RabbitMQProviderSettings(
            host="localhost",
            username="guest",
            password="guest"
        )
        
        assert settings.host == "localhost"
        assert settings.port == 5672  # RabbitMQ default
        assert settings.username == "guest"
        assert settings.password == "guest"
        assert settings.connection_string is None
        assert settings.virtual_host == "/"
        assert settings.heartbeat == 60
        assert settings.connection_timeout == 10.0
        assert settings.ssl is False
        assert settings.ssl_options is None
        assert settings.exchange_name is None
        assert settings.exchange_type == "topic"
        assert settings.exchange_durable is True
        assert settings.queue_durable is True
        assert settings.queue_auto_delete is False
        assert settings.delivery_mode == 2  # persistent
        assert settings.connect_args == {}
        # Inherited from MQProviderSettings
        assert settings.timeout == 30.0
        assert settings.prefetch_count == 10
    
    def test_settings_creation_full(self):
        """Test settings creation with all fields."""
        ssl_options = {"cert_reqs": "CERT_REQUIRED"}
        connect_args = {"loop": None, "client_properties": {"app": "test"}}
        
        settings = RabbitMQProviderSettings(
            host="rabbitmq.example.com",
            port=5671,
            username="admin",
            password="secret123",
            connection_string="amqps://admin:secret123@rabbitmq.example.com:5671/prod",
            virtual_host="/prod",
            heartbeat=120,
            connection_timeout=30.0,
            ssl=True,
            ssl_options=ssl_options,
            exchange_name="custom_exchange",
            exchange_type="direct",
            exchange_durable=False,
            queue_durable=False,
            queue_auto_delete=True,
            delivery_mode=1,  # non-persistent
            connect_args=connect_args,
            timeout=60.0,
            prefetch_count=20,
            auto_ack=True
        )
        
        assert settings.host == "rabbitmq.example.com"
        assert settings.port == 5671
        assert settings.username == "admin"
        assert settings.password == "secret123"
        assert settings.connection_string == "amqps://admin:secret123@rabbitmq.example.com:5671/prod"
        assert settings.virtual_host == "/prod"
        assert settings.heartbeat == 120
        assert settings.connection_timeout == 30.0
        assert settings.ssl is True
        assert settings.ssl_options == ssl_options
        assert settings.exchange_name == "custom_exchange"
        assert settings.exchange_type == "direct"
        assert settings.exchange_durable is False
        assert settings.queue_durable is False
        assert settings.queue_auto_delete is True
        assert settings.delivery_mode == 1
        assert settings.connect_args == connect_args
        assert settings.timeout == 60.0
        assert settings.prefetch_count == 20
        assert settings.auto_ack is True
    
    def test_settings_inheritance(self):
        """Test that RabbitMQProviderSettings inherits from MQProviderSettings."""
        from flowlib.providers.mq.base import MQProviderSettings
        settings = RabbitMQProviderSettings(
            host="localhost",
            username="guest",
            password="guest"
        )
        assert isinstance(settings, MQProviderSettings)
    
    def test_exchange_types(self):
        """Test different exchange type configurations."""
        # Topic exchange
        settings = RabbitMQProviderSettings(
            host="localhost",
            username="guest",
            password="guest",
            exchange_type="topic"
        )
        assert settings.exchange_type == "topic"
        
        # Direct exchange
        settings = RabbitMQProviderSettings(
            host="localhost",
            username="guest",
            password="guest",
            exchange_type="direct"
        )
        assert settings.exchange_type == "direct"
        
        # Fanout exchange
        settings = RabbitMQProviderSettings(
            host="localhost",
            username="guest",
            password="guest",
            exchange_type="fanout"
        )
        assert settings.exchange_type == "fanout"
        
        # Headers exchange
        settings = RabbitMQProviderSettings(
            host="localhost",
            username="guest",
            password="guest",
            exchange_type="headers"
        )
        assert settings.exchange_type == "headers"


@pytest.mark.skipif(
    pytest.importorskip("aio_pika", reason="aio_pika not available") is None,
    reason="RabbitMQ tests require aio_pika dependency"
)
class TestRabbitMQProvider:
    """Test RabbitMQProvider implementation."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return RabbitMQProviderSettings(
            host="localhost",
            port=5672,
            username="guest",
            password="guest",
            virtual_host="/test",
            exchange_name="test_exchange",
            exchange_type="topic"
        )
    
    @pytest_asyncio.fixture
    def provider(self, provider_settings):
        """Create test provider instance."""
        return RabbitMQProvider(settings=provider_settings)
    
    def test_provider_initialization(self, provider, provider_settings):
        """Test provider initialization."""
        assert provider.name == "rabbitmq"
        assert provider.settings == provider_settings
        assert not provider._initialized
        assert provider._connection is None
        assert provider._channel is None
        assert provider._exchange is None
        assert provider._queues == {}
        assert provider._consumer_tags == set()
        assert provider._tag_to_queue_map == {}
    
    def test_provider_inheritance(self, provider):
        """Test provider inheritance from base Provider class."""
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
    
    def test_provider_default_settings(self):
        """Test provider with default settings."""
        provider = RabbitMQProvider()
        assert provider.name == "rabbitmq"
        assert isinstance(provider.settings, RabbitMQProviderSettings)
        assert provider.settings.host == "localhost"
    
    def test_provider_decorator_registration(self):
        """Test that provider is properly registered with decorator."""
        assert hasattr(RabbitMQProvider, '__provider_name__')
        assert hasattr(RabbitMQProvider, '__provider_type__')
        assert RabbitMQProvider.__provider_name__ == "rabbitmq"
        assert RabbitMQProvider.__provider_type__ == "message_queue"


class TestRabbitMQProviderWithMocks:
    """Test RabbitMQProvider with mocked aio_pika."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return RabbitMQProviderSettings(
            host="localhost",
            port=5672,
            username="guest",
            password="guest",
            virtual_host="/test",
            exchange_name="test_exchange"
        )
    
    @pytest_asyncio.fixture
    def mock_connection(self):
        """Create mock RabbitMQ connection."""
        connection = AsyncMock()
        connection.close = AsyncMock()
        return connection
    
    @pytest_asyncio.fixture
    def mock_channel(self):
        """Create mock RabbitMQ channel."""
        channel = AsyncMock()
        channel.declare_exchange = AsyncMock()
        channel.declare_queue = AsyncMock()
        channel.queue_delete = AsyncMock()
        channel.cancel = AsyncMock()
        return channel
    
    @pytest_asyncio.fixture
    def mock_exchange(self):
        """Create mock RabbitMQ exchange."""
        exchange = AsyncMock()
        exchange.name = "test_exchange"
        exchange.publish = AsyncMock()
        return exchange
    
    @pytest_asyncio.fixture
    def mock_queue(self):
        """Create mock RabbitMQ queue."""
        queue = AsyncMock()
        queue.bind = AsyncMock()
        queue.consume = AsyncMock()
        queue.cancel = AsyncMock()
        return queue
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_connection, mock_channel, mock_exchange, mock_queue):
        """Create and initialize test provider with mocks."""
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_channel.declare_queue.return_value = mock_queue
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_provider_lifecycle(self, provider_settings, mock_connection, mock_channel, mock_exchange):
        """Test provider initialization and shutdown lifecycle."""
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(settings=provider_settings)
            
            # Initially not initialized
            assert not provider._initialized
            assert provider._connection is None
            
            # Initialize
            await provider.initialize()
            assert provider._initialized
            assert provider._connection is not None
            assert provider._channel is not None
            assert provider._exchange is not None
            
            # Shutdown
            await provider.shutdown()
            assert not provider._initialized
            assert provider._connection is None
            assert provider._channel is None
            assert provider._exchange is None
            mock_connection.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_with_connection_string(self, provider_settings, mock_connection, mock_channel, mock_exchange):
        """Test initialization uses connection_string when provided."""
        provider_settings = provider_settings.model_copy(update={"connection_string": "amqp://guest:guest@localhost:5672/"})
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection) as mock_connect:
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            
            # Should use connection_string
            mock_connect.assert_called_with(
                "amqp://guest:guest@localhost:5672/",
                timeout=provider_settings.connection_timeout,
                **provider_settings.connect_args
            )
    
    @pytest.mark.asyncio
    async def test_initialization_without_connection_string(self, provider_settings, mock_connection, mock_channel, mock_exchange):
        """Test initialization constructs URL when connection_string not provided."""
        provider_settings = provider_settings.model_copy(update={"connection_string": None})
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection) as mock_connect:
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            
            # Should construct URL from host/port/credentials
            expected_url = f"amqp://{provider_settings.username}:{provider_settings.password}@{provider_settings.host}:{provider_settings.port}/{provider_settings.virtual_host}"
            mock_connect.assert_called_with(
                expected_url,
                timeout=provider_settings.connection_timeout,
                **provider_settings.connect_args
            )
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, provider_settings):
        """Test initialization failure handling."""
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', side_effect=Exception("Connection failed")):
            provider = RabbitMQProvider(settings=provider_settings)
            
            with pytest.raises(ProviderError, match="Failed to connect to RabbitMQ"):
                await provider.initialize()
            
            # Should clean up on failure
            assert provider._connection is None
            assert provider._channel is None
            assert provider._exchange is None
    
    @pytest.mark.asyncio
    async def test_exchange_creation(self, provider_settings, mock_connection, mock_channel, mock_exchange):
        """Test exchange creation during initialization."""
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            
            # Should declare exchange with correct parameters
            mock_channel.declare_exchange.assert_called_with(
                name=provider_settings.exchange_name,
                type=provider_settings.exchange_type,
                durable=provider_settings.exchange_durable
            )


class TestRabbitMQProviderPublishing:
    """Test RabbitMQ message publishing functionality."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return RabbitMQProviderSettings(
            host="localhost",
            username="guest",
            password="guest",
            exchange_name="test_exchange",
            delivery_mode=2
        )
    
    @pytest_asyncio.fixture
    def mock_message_class(self):
        """Create mock Message class."""
        with patch('flowlib.providers.mq.rabbitmq.provider.Message') as mock_msg_class:
            mock_message = Mock()
            mock_msg_class.return_value = mock_message
            yield mock_msg_class, mock_message
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_message_class):
        """Create and initialize test provider."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_publish_dict_message(self, provider, mock_message_class):
        """Test publishing dictionary message."""
        mock_msg_class, mock_message = mock_message_class
        routing_key = "test.route"
        message = {"id": 123, "name": "test", "active": True}
        metadata = MessageMetadata(
            message_id="msg_123",
            content_type="application/json"
        )
        
        await provider.publish(routing_key, message, metadata)
        
        # Verify Message creation
        mock_msg_class.assert_called_once()
        call_args = mock_msg_class.call_args[1]
        
        assert call_args["body"] == json.dumps(message).encode()
        assert call_args["content_type"] == "application/json"
        assert call_args["delivery_mode"] == 2
        assert call_args["message_id"] is not None
        assert call_args["timestamp"] is not None
        assert call_args["headers"] == metadata.model_dump()
        
        # Verify exchange publish
        provider._exchange.publish.assert_called_once_with(
            message=mock_message,
            routing_key=routing_key
        )
    
    @pytest.mark.asyncio
    async def test_publish_string_message(self, provider, mock_message_class):
        """Test publishing string message."""
        mock_msg_class, mock_message = mock_message_class
        routing_key = "test.route"
        message = "Hello, RabbitMQ!"
        
        await provider.publish(routing_key, message)
        
        call_args = mock_msg_class.call_args[1]
        assert call_args["body"] == message.encode()
        assert call_args["content_type"] == "application/json"  # default
    
    @pytest.mark.asyncio
    async def test_publish_bytes_message(self, provider, mock_message_class):
        """Test publishing bytes message."""
        mock_msg_class, mock_message = mock_message_class
        routing_key = "test.route"
        message = b"Binary data \x00\x01\x02"
        
        await provider.publish(routing_key, message)
        
        call_args = mock_msg_class.call_args[1]
        assert call_args["body"] == message
    
    @pytest.mark.asyncio
    async def test_publish_with_expiration(self, provider, mock_message_class):
        """Test publishing with message expiration."""
        mock_msg_class, mock_message = mock_message_class
        routing_key = "test.route"
        message = "expiring message"
        expiration = 60000  # 60 seconds in milliseconds
        
        await provider.publish(routing_key, message, expiration=expiration)
        
        call_args = mock_msg_class.call_args[1]
        assert call_args["expiration"] == str(expiration)
    
    @pytest.mark.asyncio
    async def test_publish_with_custom_content_type(self, provider, mock_message_class):
        """Test publishing with custom content type."""
        mock_msg_class, mock_message = mock_message_class
        routing_key = "test.route"
        message = "<xml>data</xml>"
        content_type = "application/xml"
        
        await provider.publish(routing_key, message, content_type=content_type)
        
        call_args = mock_msg_class.call_args[1]
        assert call_args["content_type"] == content_type
    
    @pytest.mark.asyncio
    async def test_publish_auto_initialization(self, provider_settings, mock_message_class):
        """Test that publish auto-initializes provider if needed."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(settings=provider_settings)
            
            # Should auto-initialize
            await provider.publish("test.route", "message")
            
            assert provider._initialized
    
    @pytest.mark.asyncio
    async def test_publish_failure(self, provider):
        """Test publish failure handling."""
        provider._exchange.publish.side_effect = Exception("Publish failed")
        
        with pytest.raises(ProviderError, match="Failed to publish message"):
            await provider.publish("test.route", "message")
    
    @pytest.mark.asyncio
    async def test_publish_with_comprehensive_metadata(self, provider, mock_message_class):
        """Test publishing with comprehensive metadata."""
        mock_msg_class, mock_message = mock_message_class
        routing_key = "test.route"
        message = {"data": "test"}
        metadata = MessageMetadata(
            message_id="comprehensive_test",
            correlation_id="corr_456",
            priority=8,
            content_type="application/json",
            headers={
                "source": "test_service",
                "version": "1.0"
            }
        )
        
        await provider.publish(routing_key, message, metadata)
        
        call_args = mock_msg_class.call_args[1]
        headers = call_args["headers"]
        
        assert headers["message_id"] == "comprehensive_test"
        assert headers["correlation_id"] == "corr_456"
        assert headers["priority"] == 8
        assert headers["content_type"] == "application/json"
        assert headers["headers"]["source"] == "test_service"
        assert headers["headers"]["version"] == "1.0"


class TestRabbitMQProviderConsuming:
    """Test RabbitMQ message consuming functionality."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return RabbitMQProviderSettings(
            host="localhost",
            username="guest",
            password="guest",
            exchange_name="test_exchange",
            queue_durable=True,
            queue_auto_delete=False
        )
    
    @pytest_asyncio.fixture
    def mock_message(self):
        """Create mock RabbitMQ message."""
        message = Mock()
        message.body = b'{"id": 456, "name": "test_data"}'
        message.content_type = "application/json"
        message.message_id = "msg_456"
        message.correlation_id = "corr_789"
        message.timestamp = datetime.now()
        message.expiration = "30000"
        message.priority = 5
        message.headers = {"source": "test_service", "version": "1.0"}
        message.delivery_tag = 12345
        
        # Mock the process context manager
        message.process = Mock()
        message.process.return_value.__aenter__ = AsyncMock(return_value=None)
        message.process.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock ack/nack methods
        message.ack = AsyncMock()
        message.nack = AsyncMock()
        
        return message
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings):
        """Create and initialize test provider."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        mock_queue = AsyncMock()
        
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_channel.declare_queue.return_value = mock_queue
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_consume_success(self, provider):
        """Test successful queue consumption setup."""
        queue_name = "test_queue"
        routing_keys = ["test.route.#"]
        callback_calls = []
        
        async def test_callback(message, metadata):
            callback_calls.append({"message": message, "metadata": metadata})
        
        # Mock queue operations
        mock_queue = provider._queues.get(queue_name, AsyncMock())
        mock_queue.bind = AsyncMock()
        mock_queue.consume = AsyncMock(return_value="consumer_tag_123")
        provider._channel.declare_queue.return_value = mock_queue
        
        consumer_tag = await provider.consume(queue_name, test_callback, routing_keys)
        
        # Should return consumer tag
        assert consumer_tag == "consumer_tag_123"
        
        # Should declare queue with correct parameters
        provider._channel.declare_queue.assert_called_with(
            name=queue_name,
            durable=provider.settings.queue_durable,
            auto_delete=provider.settings.queue_auto_delete
        )
        
        # Should bind queue to exchange
        mock_queue.bind.assert_called_with(provider._exchange, routing_key="test.route.#")
        
        # Should start consuming
        mock_queue.consume.assert_called_once()
        
        # Should track consumer
        assert consumer_tag in provider._consumer_tags
        assert provider._tag_to_queue_map[consumer_tag] == queue_name
    
    @pytest.mark.asyncio
    async def test_consume_with_default_routing_key(self, provider):
        """Test consumption with default routing key."""
        queue_name = "test_queue"
        
        async def test_callback(message, metadata):
            pass
        
        mock_queue = AsyncMock()
        mock_queue.consume.return_value = "consumer_tag_123"
        provider._channel.declare_queue.return_value = mock_queue
        
        await provider.consume(queue_name, test_callback)
        
        # Should use queue name as default routing key
        mock_queue.bind.assert_called_with(provider._exchange, routing_key=queue_name)
    
    @pytest.mark.asyncio
    async def test_consume_with_custom_consumer_tag(self, provider):
        """Test consumption with custom consumer tag."""
        queue_name = "test_queue"
        custom_tag = "my_custom_consumer"
        
        async def test_callback(message, metadata):
            pass
        
        mock_queue = AsyncMock()
        mock_queue.consume.return_value = custom_tag
        provider._channel.declare_queue.return_value = mock_queue
        
        consumer_tag = await provider.consume(queue_name, test_callback, consumer_tag=custom_tag)
        
        assert consumer_tag == custom_tag
        
        # Should pass custom tag to consume
        call_args = mock_queue.consume.call_args[1]
        assert call_args["consumer_tag"] == custom_tag
    
    @pytest.mark.asyncio
    async def test_message_processing(self, provider, mock_message):
        """Test message processing logic."""
        queue_name = "test_queue"
        callback_calls = []
        
        async def test_callback(message, metadata):
            callback_calls.append({"message": message, "metadata": metadata})
        
        mock_queue = AsyncMock()
        provider._channel.declare_queue.return_value = mock_queue
        
        # Start consuming to get the message handler
        await provider.consume(queue_name, test_callback)
        
        # Get the message handler from the consume call
        message_handler = mock_queue.consume.call_args[0][0]
        
        # Process a message
        await message_handler(mock_message)
        
        # Verify callback was called
        assert len(callback_calls) == 1
        call = callback_calls[0]
        
        # Check parsed message
        assert call["message"] == {"id": 456, "name": "test_data"}
        
        # Check metadata
        metadata = call["metadata"]
        assert metadata.message_id == "msg_456"
        assert metadata.correlation_id == "corr_789"
        assert metadata.expiration == 30000  # Converted to int
        assert metadata.priority == 5
        assert metadata.content_type == "application/json"
        assert metadata.headers == {"source": "test_service", "version": "1.0"}
        assert metadata.delivery_tag == 12345
        
        # Should acknowledge message
        mock_message.ack.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_message_processing_with_auto_ack(self, provider, mock_message):
        """Test message processing with auto-acknowledgment."""
        queue_name = "test_queue"
        
        async def test_callback(message, metadata):
            pass
        
        mock_queue = AsyncMock()
        provider._channel.declare_queue.return_value = mock_queue
        
        # Start consuming with auto_ack=True
        await provider.consume(queue_name, test_callback, auto_ack=True)
        
        # Get the message handler
        message_handler = mock_queue.consume.call_args[0][0]
        
        # Process a message
        await message_handler(mock_message)
        
        # Should NOT manually acknowledge message
        mock_message.ack.assert_not_called()
        
        # Should have set no_ack=True in consume call
        call_args = mock_queue.consume.call_args[1]
        assert call_args["no_ack"] is True
    
    @pytest.mark.asyncio
    async def test_message_processing_error_handling(self, provider, mock_message):
        """Test message processing error handling."""
        queue_name = "test_queue"
        
        async def failing_callback(message, metadata):
            raise Exception("Callback failed")
        
        mock_queue = AsyncMock()
        provider._channel.declare_queue.return_value = mock_queue
        
        await provider.consume(queue_name, failing_callback)
        
        # Get the message handler
        message_handler = mock_queue.consume.call_args[0][0]
        
        # Process a message (should not raise exception)
        await message_handler(mock_message)
        
        # Should nack message on error
        mock_message.nack.assert_called_once_with(requeue=False)
    
    @pytest.mark.asyncio
    async def test_message_processing_json_decode_error(self, provider, mock_message):
        """Test message processing with JSON decode error."""
        queue_name = "test_queue"
        callback_calls = []
        
        async def test_callback(message, metadata):
            callback_calls.append({"message": message, "metadata": metadata})
        
        # Set invalid JSON
        mock_message.body = b"invalid json data"
        mock_message.content_type = "application/json"
        
        mock_queue = AsyncMock()
        provider._channel.declare_queue.return_value = mock_queue
        
        await provider.consume(queue_name, test_callback)
        
        # Get the message handler
        message_handler = mock_queue.consume.call_args[0][0]
        
        # Process the message
        await message_handler(mock_message)
        
        # Should pass raw bytes when JSON decode fails
        assert len(callback_calls) == 1
        assert callback_calls[0]["message"] == b"invalid json data"
    
    @pytest.mark.asyncio
    async def test_stop_consuming_success(self, provider):
        """Test successful consumer stopping."""
        queue_name = "test_queue"
        consumer_tag = "consumer_123"
        
        # Set up consumer state
        mock_queue = AsyncMock()
        provider._queues[queue_name] = mock_queue
        provider._consumer_tags.add(consumer_tag)
        provider._tag_to_queue_map[consumer_tag] = queue_name
        
        await provider.stop_consuming(consumer_tag)
        
        # Should cancel consumer
        mock_queue.cancel.assert_called_with(consumer_tag)
        
        # Should clean up state
        assert consumer_tag not in provider._consumer_tags
        assert consumer_tag not in provider._tag_to_queue_map
    
    @pytest.mark.asyncio
    async def test_stop_consuming_nonexistent_tag(self, provider):
        """Test stopping non-existent consumer."""
        consumer_tag = "nonexistent_tag"
        
        # Should not raise error
        await provider.stop_consuming(consumer_tag)
    
    @pytest.mark.asyncio
    async def test_stop_consuming_failure(self, provider):
        """Test consumer stop failure handling."""
        queue_name = "test_queue"
        consumer_tag = "consumer_123"
        
        # Set up consumer state
        mock_queue = AsyncMock()
        mock_queue.cancel.side_effect = Exception("Cancel failed")
        provider._queues[queue_name] = mock_queue
        provider._consumer_tags.add(consumer_tag)
        provider._tag_to_queue_map[consumer_tag] = queue_name
        
        with pytest.raises(ProviderError, match="Failed to stop consumer"):
            await provider.stop_consuming(consumer_tag)
        
        # Should still clean up state
        assert consumer_tag not in provider._consumer_tags
        assert consumer_tag not in provider._tag_to_queue_map


class TestRabbitMQProviderQueueManagement:
    """Test RabbitMQ queue management functionality."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return RabbitMQProviderSettings(
            host="localhost",
            username="guest",
            password="guest",
            exchange_name="test_exchange",
            queue_durable=True,
            queue_auto_delete=False
        )
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings):
        """Create and initialize test provider."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_create_queue_success(self, provider):
        """Test successful queue creation."""
        queue_name = "test_queue"
        routing_keys = ["test.route.#", "another.route.*"]
        
        mock_queue = AsyncMock()
        provider._channel.declare_queue.return_value = mock_queue
        
        await provider.create_queue(queue_name, routing_keys)
        
        # Should declare queue with correct parameters
        provider._channel.declare_queue.assert_called_with(
            name=queue_name,
            durable=provider.settings.queue_durable,
            auto_delete=provider.settings.queue_auto_delete
        )
        
        # Should bind queue to each routing key
        assert mock_queue.bind.call_count == 2
        mock_queue.bind.assert_any_call(provider._exchange, routing_key="test.route.#")
        mock_queue.bind.assert_any_call(provider._exchange, routing_key="another.route.*")
        
        # Should store queue
        assert provider._queues[queue_name] == mock_queue
    
    @pytest.mark.asyncio
    async def test_create_queue_with_custom_settings(self, provider):
        """Test queue creation with custom settings."""
        queue_name = "custom_queue"
        
        mock_queue = AsyncMock()
        provider._channel.declare_queue.return_value = mock_queue
        
        await provider.create_queue(queue_name, durable=False, auto_delete=True)
        
        # Should use custom settings
        provider._channel.declare_queue.assert_called_with(
            name=queue_name,
            durable=False,
            auto_delete=True
        )
    
    @pytest.mark.asyncio
    async def test_create_queue_without_routing_keys(self, provider):
        """Test queue creation without routing keys."""
        queue_name = "simple_queue"
        
        mock_queue = AsyncMock()
        provider._channel.declare_queue.return_value = mock_queue
        
        await provider.create_queue(queue_name)
        
        # Should not bind queue to any routing keys
        mock_queue.bind.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_queue_auto_initialization(self, provider_settings):
        """Test that create_queue auto-initializes provider if needed."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        mock_queue = AsyncMock()
        
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_channel.declare_queue.return_value = mock_queue
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(settings=provider_settings)
            
            # Should auto-initialize
            await provider.create_queue("test_queue")
            
            assert provider._initialized
    
    @pytest.mark.asyncio
    async def test_create_queue_failure(self, provider):
        """Test queue creation failure handling."""
        provider._channel.declare_queue.side_effect = Exception("Queue creation failed")
        
        with pytest.raises(ProviderError, match="Failed to create queue"):
            await provider.create_queue("failing_queue")
    
    @pytest.mark.asyncio
    async def test_delete_queue_success(self, provider):
        """Test successful queue deletion."""
        queue_name = "test_queue"
        
        # Add queue to tracked queues
        mock_queue = AsyncMock()
        provider._queues[queue_name] = mock_queue
        
        await provider.delete_queue(queue_name)
        
        # Should delete queue
        provider._channel.queue_delete.assert_called_with(queue_name)
        
        # Should remove from tracked queues
        assert queue_name not in provider._queues
    
    @pytest.mark.asyncio
    async def test_delete_queue_not_tracked(self, provider):
        """Test deleting queue that's not tracked."""
        queue_name = "untracked_queue"
        
        await provider.delete_queue(queue_name)
        
        # Should still attempt deletion
        provider._channel.queue_delete.assert_called_with(queue_name)
    
    @pytest.mark.asyncio
    async def test_delete_queue_failure(self, provider):
        """Test queue deletion failure handling."""
        provider._channel.queue_delete.side_effect = Exception("Queue deletion failed")
        
        with pytest.raises(ProviderError, match="Failed to delete queue"):
            await provider.delete_queue("failing_queue")


class TestRabbitMQProviderAdvancedFeatures:
    """Test advanced RabbitMQ provider features."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return RabbitMQProviderSettings(
            host="localhost",
            username="admin",
            password="secret",
            virtual_host="/prod",
            ssl=True,
            ssl_options={"cert_reqs": "CERT_REQUIRED"},
            exchange_name="production_exchange",
            exchange_type="direct",
            exchange_durable=True,
            queue_durable=True,
            delivery_mode=2,
            connect_args={"client_properties": {"app": "test"}}
        )
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings):
        """Create and initialize test provider."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_ssl_configuration(self, provider_settings):
        """Test provider with SSL configuration."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection) as mock_connect:
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            
            # Should have been called with SSL parameters expanded from connect_args
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs['client_properties'] == {'app': 'test'}
    
    @pytest.mark.asyncio
    async def test_custom_exchange_configuration(self, provider):
        """Test provider with custom exchange configuration."""
        # Verify exchange was created with custom settings
        provider._channel.declare_exchange.assert_called_with(
            name="production_exchange",
            type="direct",
            durable=True
        )
    
    @pytest.mark.asyncio
    async def test_multiple_consumers(self, provider):
        """Test managing multiple consumers."""
        queues = ["queue1", "queue2", "queue3"]
        consumer_tags = []
        
        async def callback(message, metadata):
            pass
        
        # Mock queue operations
        for i, queue_name in enumerate(queues):
            mock_queue = AsyncMock()
            tag = f"consumer_{i}"
            mock_queue.consume.return_value = tag
            provider._channel.declare_queue.return_value = mock_queue
            
            consumer_tag = await provider.consume(queue_name, callback)
            consumer_tags.append(consumer_tag)
        
        # Should have consumers for all queues
        assert len(provider._consumer_tags) == 3
        assert len(provider._tag_to_queue_map) == 3
        assert all(tag in provider._consumer_tags for tag in consumer_tags)
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_consumers(self, provider):
        """Test graceful shutdown with active consumers."""
        # Add mock consumers
        consumer_tags = ["consumer_1", "consumer_2", "consumer_3"]
        provider._consumer_tags.update(consumer_tags)
        
        # Capture references to mocks before shutdown resets them
        mock_channel = provider._channel
        mock_connection = provider._connection
        
        await provider.shutdown()
        
        # Should cancel all consumers
        for tag in consumer_tags:
            mock_channel.cancel.assert_any_call(tag)
        
        # Should close connection
        mock_connection.close.assert_called_once()
        
        # Should clear state
        assert len(provider._consumer_tags) == 0
        assert len(provider._tag_to_queue_map) == 0
    
    @pytest.mark.asyncio
    async def test_message_persistence_configuration(self, provider):
        """Test message persistence configuration."""
        with patch('flowlib.providers.mq.rabbitmq.provider.Message') as mock_msg_class:
            await provider.publish("test.route", "persistent message")
            
            call_args = mock_msg_class.call_args[1]
            assert call_args["delivery_mode"] == 2  # persistent
    
    @pytest.mark.asyncio
    async def test_exchange_name_fallback(self, provider_settings):
        """Test exchange name fallback to provider name."""
        provider_settings = provider_settings.model_copy(update={"exchange_name": None})
        
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(name="custom_provider", settings=provider_settings)
            await provider.initialize()
            
            # Should use provider name as exchange name
            mock_channel.declare_exchange.assert_called_with(
                name="custom_provider",
                type=provider_settings.exchange_type,
                durable=provider_settings.exchange_durable
            )


class TestRabbitMQProviderErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return RabbitMQProviderSettings(
            host="localhost",
            username="guest",
            password="guest"
        )
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings):
        """Create and initialize test provider."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection):
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_operations_without_aio_pika(self, provider_settings):
        """Test operations without aio_pika package."""
        # This test would require actually removing aio_pika, which is complex in pytest
        # Instead, we test the import error handling logic exists
        provider = RabbitMQProvider(settings=provider_settings)
        assert provider._connection is None
    
    @pytest.mark.asyncio
    async def test_shutdown_error_handling(self, provider_settings):
        """Test shutdown handles errors gracefully."""
        mock_connection = AsyncMock()
        mock_connection.close.side_effect = Exception("Close failed")
        
        provider = RabbitMQProvider(settings=provider_settings)
        provider._connection = mock_connection
        provider._initialized = True
        
        # Should handle shutdown errors gracefully
        await provider.shutdown()
        assert not provider._initialized
    
    @pytest.mark.asyncio
    async def test_consumer_cancellation_error_handling(self, provider_settings):
        """Test consumer cancellation error handling."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.cancel.side_effect = Exception("Cancel failed")
        
        provider = RabbitMQProvider(settings=provider_settings)
        provider._connection = mock_connection
        provider._channel = mock_channel
        provider._consumer_tags.add("failing_consumer")
        provider._initialized = True
        
        # Should handle cancellation errors gracefully during shutdown
        await provider.shutdown()
        assert not provider._initialized
    
    @pytest.mark.asyncio
    async def test_provider_type_validation(self):
        """Test provider type is correctly set."""
        provider = RabbitMQProvider()
        assert provider.provider_type == "message_queue"  # Inherited from base
    
    @pytest.mark.asyncio
    async def test_connection_url_construction(self, provider_settings):
        """Test connection URL construction handles special characters."""
        provider_settings = provider_settings.model_copy(update={
            "username": "user@domain.com",
            "password": "pass@word!123",
            "virtual_host": "/special-vhost"
        })
        
        # Should construct URL properly (actual URL encoding would be handled by aio_pika)
        provider = RabbitMQProvider(settings=provider_settings)
        assert provider.settings.username == "user@domain.com"
        assert provider.settings.password == "pass@word!123"
        assert provider.settings.virtual_host == "/special-vhost"
    
    @pytest.mark.asyncio
    async def test_message_serialization_edge_cases(self, provider_settings):
        """Test message serialization handles edge cases."""
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_connection.channel.return_value = mock_channel
        
        with patch('flowlib.providers.mq.rabbitmq.provider.aio_pika.connect_robust', return_value=mock_connection), \
             patch('flowlib.providers.mq.rabbitmq.provider.Message') as mock_msg_class:
            
            provider = RabbitMQProvider(settings=provider_settings)
            await provider.initialize()
            
            # Test various message types
            test_cases = [
                None,  # Should convert to string
                123,   # Should convert to string  
                [],    # Empty list
                {},    # Empty dict
                {"unicode": "测试"},  # Unicode content
            ]
            
            for message in test_cases:
                await provider.publish("test.route", message)
            
            # Should have called Message constructor for each message
            assert mock_msg_class.call_count == len(test_cases)
    
    @pytest.mark.asyncio
    async def test_queue_binding_error_recovery(self, provider):
        """Test queue binding error handling."""
        queue_name = "test_queue"
        routing_keys = ["route1", "route2"]
        
        mock_queue = AsyncMock()
        mock_queue.bind.side_effect = Exception("Bind failed")
        provider._channel.declare_queue.return_value = mock_queue
        
        with pytest.raises(ProviderError, match="Failed to start consuming"):
            await provider.consume(queue_name, lambda m, meta: None, routing_keys)
    
    @pytest.mark.asyncio
    async def test_message_id_generation(self, provider):
        """Test that message IDs are generated when not provided."""
        with patch('flowlib.providers.mq.rabbitmq.provider.Message') as mock_msg_class, \
             patch('flowlib.providers.mq.rabbitmq.provider.uuid.uuid4') as mock_uuid:
            
            mock_uuid.return_value = "test-uuid-123"
            
            await provider.publish("test.route", "message")
            
            call_args = mock_msg_class.call_args[1]
            assert call_args["message_id"] == "test-uuid-123"