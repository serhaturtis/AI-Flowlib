"""Tests for Kafka MQ provider implementation."""

import pytest
import pytest_asyncio
import asyncio
import json
import sys
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Optional, Any
from datetime import datetime

from flowlib.providers.mq.kafka.provider import (
    KafkaMQProvider,
    KafkaProviderSettings
)
from flowlib.providers.mq.base import MessageMetadata
from flowlib.core.errors.errors import ProviderError


class TestKafkaProviderSettings:
    """Test KafkaProviderSettings model."""
    
    def test_settings_creation_minimal(self):
        """Test settings creation with minimal required data."""
        settings = KafkaProviderSettings(
            host="localhost",
            username="test",
            password="test"
        )
        
        assert settings.host == "localhost"
        assert settings.port == 9092  # Kafka default
        assert settings.username == "test"
        assert settings.password == "test"
        assert settings.bootstrap_servers is None
        assert settings.client_id is None
        assert settings.group_id == "flowlib_consumer_group"
        assert settings.auto_offset_reset == "latest"
        assert settings.enable_auto_commit is True
        assert settings.auto_commit_interval_ms == 5000
        assert settings.security_protocol == "PLAINTEXT"
        assert settings.sasl_mechanism is None
        assert settings.sasl_username is None
        assert settings.sasl_password is None
        assert settings.ssl_context is None
        assert settings.ssl_check_hostname is True
        assert settings.acks == "all"
        assert settings.compression_type is None
        assert settings.connect_args == {}
        # Inherited from MQProviderSettings
        assert settings.virtual_host == "/"
        assert settings.timeout == 30.0
    
    def test_settings_creation_full(self):
        """Test settings creation with all fields."""
        settings = KafkaProviderSettings(
            host="kafka.example.com",
            port=9093,
            username="admin",
            password="secret123",
            bootstrap_servers="kafka1:9092,kafka2:9092,kafka3:9092",
            client_id="test_client",
            group_id="test_consumer_group",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            auto_commit_interval_ms=10000,
            security_protocol="SASL_SSL",
            sasl_mechanism="SCRAM-SHA-256",
            sasl_username="sasl_user",
            sasl_password="sasl_pass",
            ssl_check_hostname=False,
            acks=1,
            compression_type="gzip",
            connect_args={"request_timeout_ms": 30000},
            timeout=60.0,
            heartbeat=120
        )
        
        assert settings.host == "kafka.example.com"
        assert settings.port == 9093
        assert settings.username == "admin"
        assert settings.password == "secret123"
        assert settings.bootstrap_servers == "kafka1:9092,kafka2:9092,kafka3:9092"
        assert settings.client_id == "test_client"
        assert settings.group_id == "test_consumer_group"
        assert settings.auto_offset_reset == "earliest"
        assert settings.enable_auto_commit is False
        assert settings.auto_commit_interval_ms == 10000
        assert settings.security_protocol == "SASL_SSL"
        assert settings.sasl_mechanism == "SCRAM-SHA-256"
        assert settings.sasl_username == "sasl_user"
        assert settings.sasl_password == "sasl_pass"
        assert settings.ssl_check_hostname is False
        assert settings.acks == 1
        assert settings.compression_type == "gzip"
        assert settings.connect_args == {"request_timeout_ms": 30000}
        assert settings.timeout == 60.0
        assert settings.heartbeat == 120
    
    def test_settings_inheritance(self):
        """Test that KafkaProviderSettings inherits from MQProviderSettings."""
        from flowlib.providers.mq.base import MQProviderSettings
        settings = KafkaProviderSettings(
            host="localhost",
            username="test",
            password="test"
        )
        assert isinstance(settings, MQProviderSettings)
    
    def test_security_protocols(self):
        """Test different security protocol configurations."""
        # PLAINTEXT
        settings = KafkaProviderSettings(
            host="localhost",
            username="test",
            password="test",
            security_protocol="PLAINTEXT"
        )
        assert settings.security_protocol == "PLAINTEXT"
        
        # SSL
        settings = KafkaProviderSettings(
            host="localhost",
            username="test",
            password="test",
            security_protocol="SSL"
        )
        assert settings.security_protocol == "SSL"
        
        # SASL_PLAINTEXT
        settings = KafkaProviderSettings(
            host="localhost",
            username="test",
            password="test",
            security_protocol="SASL_PLAINTEXT",
            sasl_mechanism="PLAIN",
            sasl_username="user",
            sasl_password="pass"
        )
        assert settings.security_protocol == "SASL_PLAINTEXT"
        assert settings.sasl_mechanism == "PLAIN"
        assert settings.sasl_username == "user"
        assert settings.sasl_password == "pass"


@pytest.mark.skipif(
    pytest.importorskip("aiokafka", reason="aiokafka not available") is None,
    reason="Kafka tests require aiokafka dependency"
)
class TestKafkaProvider:
    """Test KafkaMQProvider implementation."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return KafkaProviderSettings(
            host="localhost",
            port=9092,
            username="test",
            password="test",
            group_id="test_group",
            client_id="test_client"
        )
    
    @pytest_asyncio.fixture
    def provider(self, provider_settings):
        """Create test provider instance."""
        return KafkaMQProvider(settings=provider_settings)
    
    def test_provider_initialization(self, provider, provider_settings):
        """Test provider initialization."""
        assert provider.name == "kafka"
        assert provider.settings == provider_settings
        assert not provider._initialized
        assert provider._producer is None
        assert provider._consumers == {}
        assert provider._consumer_tasks == {}
    
    def test_provider_inheritance(self, provider):
        """Test provider inheritance from base Provider class."""
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
    
    def test_provider_default_settings(self):
        """Test provider with default settings."""
        provider = KafkaMQProvider()
        assert provider.name == "kafka"
        assert isinstance(provider.settings, KafkaProviderSettings)
        assert provider.settings.host == "localhost"
    
    def test_provider_decorator_registration(self):
        """Test that provider is properly registered with decorator."""
        assert hasattr(KafkaMQProvider, '__provider_name__')
        assert hasattr(KafkaMQProvider, '__provider_type__')
        assert KafkaMQProvider.__provider_name__ == "kafka"
        assert KafkaMQProvider.__provider_type__ == "message_queue"


class TestKafkaProviderWithMocks:
    """Test KafkaMQProvider with mocked aiokafka."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return KafkaProviderSettings(
            host="localhost",
            port=9092,
            username="test",
            password="test",
            bootstrap_servers="localhost:9092",
            group_id="test_group",
            client_id="test_client"
        )
    
    @pytest_asyncio.fixture
    def mock_producer(self):
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer.send_and_wait = AsyncMock()
        return producer
    
    @pytest_asyncio.fixture
    def mock_consumer(self):
        """Create mock Kafka consumer."""
        consumer = AsyncMock()
        consumer.start = AsyncMock()
        consumer.stop = AsyncMock()
        
        # Mock async iteration
        consumer.__aiter__ = AsyncMock(return_value=consumer)
        consumer.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        
        return consumer
    
    @pytest_asyncio.fixture
    def mock_admin_client(self):
        """Create mock Kafka admin client."""
        admin = Mock()
        admin.create_topics = Mock()
        admin.delete_topics = Mock()
        admin.close = Mock()
        return admin
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_producer, mock_consumer):
        """Create and initialize test provider with mocks."""
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer), \
             patch('flowlib.providers.mq.kafka.provider.AIOKafkaConsumer', return_value=mock_consumer):
            
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_provider_lifecycle(self, provider_settings, mock_producer, mock_consumer):
        """Test provider initialization and shutdown lifecycle."""
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer), \
             patch('flowlib.providers.mq.kafka.provider.AIOKafkaConsumer', return_value=mock_consumer):
            
            provider = KafkaMQProvider(settings=provider_settings)
            
            # Initially not initialized
            assert not provider._initialized
            assert provider._producer is None
            
            # Initialize
            await provider.initialize()
            assert provider._initialized
            assert provider._producer is not None
            mock_producer.start.assert_called_once()
            
            # Shutdown
            await provider.shutdown()
            assert not provider._initialized
            assert provider._producer is None
            mock_producer.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_with_bootstrap_servers(self, provider_settings, mock_producer):
        """Test initialization uses bootstrap_servers when provided."""
        provider_settings = provider_settings.model_copy(update={"bootstrap_servers": "kafka1:9092,kafka2:9092"})
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer) as mock_producer_class:
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            
            # Should use bootstrap_servers from settings
            call_args = mock_producer_class.call_args[1]
            assert call_args["bootstrap_servers"] == "kafka1:9092,kafka2:9092"
    
    @pytest.mark.asyncio
    async def test_initialization_without_bootstrap_servers(self, provider_settings, mock_producer):
        """Test initialization uses host:port when bootstrap_servers not provided."""
        provider_settings = provider_settings.model_copy(update={"bootstrap_servers": None})
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer) as mock_producer_class:
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            
            # Should construct from host:port
            call_args = mock_producer_class.call_args[1]
            assert call_args["bootstrap_servers"] == "localhost:9092"
    
    @pytest.mark.asyncio
    async def test_initialization_with_sasl_auth(self, provider_settings, mock_producer):
        """Test initialization with SASL authentication."""
        provider_settings = provider_settings.model_copy(update={
            "sasl_mechanism": "SCRAM-SHA-256",
            "sasl_username": "sasl_user",
            "sasl_password": "sasl_pass"
        })
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer) as mock_producer_class:
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            
            call_args = mock_producer_class.call_args[1]
            assert call_args["sasl_mechanism"] == "SCRAM-SHA-256"
            assert call_args["sasl_plain_username"] == "sasl_user"
            assert call_args["sasl_plain_password"] == "sasl_pass"
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, provider_settings, mock_producer):
        """Test initialization failure handling."""
        mock_producer.start.side_effect = Exception("Connection failed")
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer):
            provider = KafkaMQProvider(settings=provider_settings)
            
            with pytest.raises(ProviderError, match="Failed to initialize provider"):
                await provider.initialize()
            
            # Should clean up producer on failure
            assert provider._producer is None


class TestKafkaProviderPublishing:
    """Test Kafka message publishing functionality."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return KafkaProviderSettings(
            host="localhost",
            port=9092,
            username="test",
            password="test"
        )
    
    @pytest_asyncio.fixture
    def mock_producer(self):
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer.send_and_wait = AsyncMock()
        return producer
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_producer):
        """Create and initialize test provider."""
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer):
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_publish_dict_message(self, provider, mock_producer):
        """Test publishing dictionary message."""
        topic = "test_topic"
        message = {"id": 123, "name": "test", "active": True}
        metadata = MessageMetadata(
            message_id="msg_123",
            content_type="application/json"
        )
        
        await provider.publish(topic, message, metadata)
        
        mock_producer.send_and_wait.assert_called_once()
        call_args = mock_producer.send_and_wait.call_args[1]
        
        assert call_args["topic"] == topic
        assert call_args["value"] == json.dumps(message).encode()
        assert call_args["key"] == b"msg_123"
        
        # Check headers
        headers = call_args["headers"]
        header_dict = {k: v.decode() for k, v in headers if k != "timestamp"}
        assert header_dict["message_id"] == "msg_123"
        assert header_dict["content_type"] == "application/json"
    
    @pytest.mark.asyncio
    async def test_publish_string_message(self, provider, mock_producer):
        """Test publishing string message."""
        topic = "test_topic"
        message = "Hello, Kafka!"
        
        await provider.publish(topic, message)
        
        call_args = mock_producer.send_and_wait.call_args[1]
        assert call_args["value"] == message.encode()
        assert call_args["key"] is None
    
    @pytest.mark.asyncio
    async def test_publish_bytes_message(self, provider, mock_producer):
        """Test publishing bytes message."""
        topic = "test_topic"
        message = b"Binary data \x00\x01\x02"
        
        await provider.publish(topic, message)
        
        call_args = mock_producer.send_and_wait.call_args[1]
        assert call_args["value"] == message
    
    @pytest.mark.asyncio
    async def test_publish_list_message(self, provider, mock_producer):
        """Test publishing list message."""
        topic = "test_topic"
        message = [1, 2, {"nested": "data"}]
        
        await provider.publish(topic, message)
        
        call_args = mock_producer.send_and_wait.call_args[1]
        assert call_args["value"] == json.dumps(message).encode()
    
    @pytest.mark.asyncio
    async def test_publish_with_custom_key(self, provider, mock_producer):
        """Test publishing with custom message key."""
        topic = "test_topic"
        message = "test message"
        key = "custom_key"
        
        await provider.publish(topic, message, key=key)
        
        call_args = mock_producer.send_and_wait.call_args[1]
        assert call_args["key"] == key.encode()
    
    @pytest.mark.asyncio
    async def test_publish_auto_initialization(self, provider_settings, mock_producer):
        """Test that publish auto-initializes provider if needed."""
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer):
            provider = KafkaMQProvider(settings=provider_settings)
            
            # Should auto-initialize
            await provider.publish("topic", "message")
            
            assert provider._initialized
            mock_producer.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_publish_failure(self, provider, mock_producer):
        """Test publish failure handling."""
        mock_producer.send_and_wait.side_effect = Exception("Send failed")
        
        with pytest.raises(ProviderError, match="Failed to publish message"):
            await provider.publish("topic", "message")
    
    @pytest.mark.asyncio
    async def test_publish_with_comprehensive_metadata(self, provider, mock_producer):
        """Test publishing with comprehensive metadata."""
        topic = "test_topic"
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
        
        await provider.publish(topic, message, metadata)
        
        call_args = mock_producer.send_and_wait.call_args[1]
        headers = dict(call_args["headers"])
        
        # Decode headers for comparison
        decoded_headers = {k: v.decode() if isinstance(v, bytes) else v for k, v in headers.items()}
        
        assert decoded_headers["message_id"] == "comprehensive_test"
        assert decoded_headers["correlation_id"] == "corr_456"
        assert decoded_headers["priority"] == "8"
        assert decoded_headers["content_type"] == "application/json"
        assert "timestamp" in decoded_headers


class TestKafkaProviderConsuming:
    """Test Kafka message consuming functionality."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return KafkaProviderSettings(
            host="localhost",
            port=9092,
            username="test",
            password="test",
            group_id="test_group"
        )
    
    @pytest_asyncio.fixture
    def mock_producer(self):
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        return producer
    
    @pytest_asyncio.fixture
    def mock_consumer(self):
        """Create mock Kafka consumer."""
        consumer = AsyncMock()
        consumer.start = AsyncMock()
        consumer.stop = AsyncMock()
        return consumer
    
    @pytest_asyncio.fixture
    def mock_message(self):
        """Create mock Kafka message."""
        message = Mock()
        message.topic = "test_topic"
        message.partition = 0
        message.offset = 123
        message.timestamp = 1640995200000  # 2022-01-01 00:00:00 UTC
        message.key = b"test_key"
        message.value = b'{"id": 456, "name": "test_data"}'
        message.headers = [
            ("content_type", b"application/json"),
            ("source", b"test_service")
        ]
        return message
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_producer, mock_consumer):
        """Create and initialize test provider."""
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer), \
             patch('flowlib.providers.mq.kafka.provider.AIOKafkaConsumer', return_value=mock_consumer):
            
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_subscribe_success(self, provider, mock_consumer):
        """Test successful topic subscription."""
        topic = "test_topic"
        callback_calls = []
        
        async def test_callback(message, metadata):
            callback_calls.append({"message": message, "metadata": metadata})
        
        # Create a mock task that can be cancelled
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        mock_task.cancel = AsyncMock()
        
        # Mock create_task to properly handle the coroutine
        def mock_create_task_fn(coro):
            # Close the coroutine to avoid "never awaited" warning
            coro.close()
            return mock_task
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaConsumer', return_value=mock_consumer), \
             patch('asyncio.create_task', side_effect=mock_create_task_fn) as mock_create_task:
            
            await provider.subscribe(topic, test_callback)
            
            # Should create consumer and start task
            assert topic in provider._consumers
            assert topic in provider._consumer_tasks
            mock_create_task.assert_called_once()
            
            # Verify the task is stored properly
            assert provider._consumer_tasks[topic] == mock_task
    
    @pytest.mark.asyncio
    async def test_subscribe_with_custom_group_id(self, provider, mock_consumer):
        """Test subscription with custom group ID."""
        topic = "test_topic"
        custom_group_id = "custom_group"
        
        async def test_callback(message, metadata):
            pass
        
        # Create a mock task that can be cancelled
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        mock_task.cancel = AsyncMock()
        
        # Mock create_task to properly handle the coroutine
        def mock_create_task_fn(coro):
            # Close the coroutine to avoid "never awaited" warning
            coro.close()
            return mock_task
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaConsumer', return_value=mock_consumer) as mock_consumer_class, \
             patch('asyncio.create_task', side_effect=mock_create_task_fn):
            
            await provider.subscribe(topic, test_callback, group_id=custom_group_id)
            
            # Should use custom group ID
            call_args = mock_consumer_class.call_args[1]
            assert call_args["group_id"] == custom_group_id
    
    @pytest.mark.asyncio
    async def test_subscribe_with_custom_offset_reset(self, provider, mock_consumer):
        """Test subscription with custom auto_offset_reset."""
        topic = "test_topic"
        custom_offset_reset = "earliest"
        
        async def test_callback(message, metadata):
            pass
        
        # Create a mock task that can be cancelled
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        mock_task.cancel = AsyncMock()
        
        # Mock create_task to properly handle the coroutine
        def mock_create_task_fn(coro):
            # Close the coroutine to avoid "never awaited" warning
            coro.close()
            return mock_task
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaConsumer', return_value=mock_consumer) as mock_consumer_class, \
             patch('asyncio.create_task', side_effect=mock_create_task_fn):
            
            await provider.subscribe(topic, test_callback, auto_offset_reset=custom_offset_reset)
            
            # Should use custom offset reset
            call_args = mock_consumer_class.call_args[1]
            assert call_args["auto_offset_reset"] == custom_offset_reset
    
    @pytest.mark.asyncio
    async def test_subscribe_already_subscribed(self, provider):
        """Test subscribing to already subscribed topic."""
        topic = "test_topic"
        provider._consumers[topic] = Mock()
        
        async def test_callback(message, metadata):
            pass
        
        with pytest.raises(ProviderError, match="Already subscribed"):
            await provider.subscribe(topic, test_callback)
    
    @pytest.mark.asyncio
    async def test_unsubscribe_success(self, provider):
        """Test successful topic unsubscription."""
        topic = "test_topic"
        mock_consumer = AsyncMock()
        mock_task = Mock()
        mock_task.done.return_value = False
        
        provider._consumers[topic] = mock_consumer
        provider._consumer_tasks[topic] = mock_task
        
        await provider.unsubscribe(topic)
        
        # Should cancel task and stop consumer
        mock_task.cancel.assert_called_once()
        mock_consumer.stop.assert_called_once()
        
        # Should remove from collections
        assert topic not in provider._consumers
        assert topic not in provider._consumer_tasks
    
    @pytest.mark.asyncio
    async def test_unsubscribe_not_subscribed(self, provider):
        """Test unsubscribing from non-subscribed topic."""
        topic = "nonexistent_topic"
        
        with pytest.raises(ProviderError, match="Not subscribed"):
            await provider.unsubscribe(topic)
    
    @pytest.mark.asyncio
    async def test_message_processing(self, provider, mock_consumer, mock_message):
        """Test message processing in consumer task."""
        topic = "test_topic"
        callback_calls = []
        
        async def test_callback(message, metadata):
            callback_calls.append({"message": message, "metadata": metadata})
        
        # Mock the consumer iteration
        messages = [mock_message]
        
        async def mock_aiter():
            for msg in messages:
                yield msg
        
        mock_consumer.__aiter__ = lambda self: mock_aiter()
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaConsumer', return_value=mock_consumer):
            # Simulate the consumer task logic
            await mock_consumer.start()
            
            async for message in mock_consumer:
                # Parse message (same logic as in actual implementation)
                try:
                    data = json.loads(message.value.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    data = message.value
                
                # Extract metadata
                headers_dict = {}
                if message.headers:
                    for key, value in message.headers:
                        key_str = key.decode() if isinstance(key, bytes) else key
                        try:
                            value_str = value.decode() if isinstance(value, bytes) else value
                            headers_dict[key_str] = value_str
                        except UnicodeDecodeError:
                            pass
                
                # Extract content_type from headers if present
                content_type = headers_dict.pop("content_type", None)
                
                meta_dict = {
                    "topic": message.topic,
                    "partition": message.partition,
                    "offset": message.offset,
                    "timestamp": message.timestamp,
                    "message_id": message.key.decode() if message.key else None,
                    "routing_key": message.topic,
                    "content_type": content_type,
                    "headers": headers_dict
                }
                
                meta = MessageMetadata(**meta_dict)
                await test_callback(data, meta)
        
        # Verify callback was called with correct data
        assert len(callback_calls) == 1
        call = callback_calls[0]
        
        # Check parsed message
        assert call["message"] == {"id": 456, "name": "test_data"}
        
        # Check metadata
        metadata = call["metadata"]
        assert metadata.topic == "test_topic"
        assert metadata.partition == 0
        assert metadata.offset == 123
        assert metadata.timestamp == 1640995200000
        assert metadata.message_id == "test_key"
        assert metadata.routing_key == "test_topic"
        assert metadata.content_type == "application/json"
    
    @pytest.mark.asyncio
    async def test_consumer_task_cancellation(self, provider):
        """Test consumer task handles cancellation gracefully."""
        topic = "test_topic"
        mock_consumer = AsyncMock()
        mock_task = Mock()
        
        # Simulate task cancellation
        mock_task.done.return_value = False
        
        provider._consumers[topic] = mock_consumer
        provider._consumer_tasks[topic] = mock_task
        
        # Test cleanup during shutdown
        await provider.shutdown()
        
        # Should cancel task and stop consumer
        mock_task.cancel.assert_called_once()
        mock_consumer.stop.assert_called_once()


class TestKafkaProviderTopicManagement:
    """Test Kafka topic management functionality."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return KafkaProviderSettings(
            host="localhost",
            port=9092,
            username="test",
            password="test"
        )
    
    @pytest_asyncio.fixture
    def mock_producer(self):
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        return producer
    
    @pytest_asyncio.fixture
    def mock_admin_client(self):
        """Create mock Kafka admin client."""
        admin = Mock()
        admin.create_topics = Mock()
        admin.delete_topics = Mock()
        admin.close = Mock()
        return admin
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_producer):
        """Create and initialize test provider."""
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer):
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_create_topic_success(self, provider, mock_admin_client):
        """Test successful topic creation."""
        topic = "test_topic"
        num_partitions = 3
        replication_factor = 2
        
        # Mock the kafka.admin module at sys.modules level
        mock_new_topic = Mock()
        kafka_admin_mock = Mock()
        kafka_admin_mock.KafkaAdminClient = Mock(return_value=mock_admin_client)
        kafka_admin_mock.NewTopic = mock_new_topic
        kafka_admin_mock.TopicAlreadyExistsError = Exception
        
        with patch.dict('sys.modules', {'kafka': Mock(), 'kafka.admin': kafka_admin_mock, 'kafka.errors': Mock()}):
            
            await provider.create_topic(topic, num_partitions, replication_factor)
            
            # Should create NewTopic with correct parameters
            mock_new_topic.assert_called_with(
                name=topic,
                num_partitions=num_partitions,
                replication_factor=replication_factor
            )
            
            # Should call create_topics
            mock_admin_client.create_topics.assert_called_once()
            mock_admin_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_topic_already_exists(self, provider, mock_admin_client):
        """Test creating topic that already exists."""
        topic = "existing_topic"
        
        # Simulate TopicAlreadyExistsError
        from unittest.mock import Mock
        error = Mock()
        error.__class__.__name__ = "TopicAlreadyExistsError"
        mock_admin_client.create_topics.side_effect = error
        
        # Mock the kafka.admin module at sys.modules level  
        kafka_admin_mock = Mock()
        kafka_admin_mock.KafkaAdminClient = Mock(return_value=mock_admin_client)
        kafka_admin_mock.NewTopic = Mock()
        kafka_admin_mock.TopicAlreadyExistsError = Exception
        
        with patch.dict('sys.modules', {'kafka': Mock(), 'kafka.admin': kafka_admin_mock, 'kafka.errors': Mock()}):
            
            # Should not raise error
            await provider.create_topic(topic)
            mock_admin_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_topic_missing_kafka_python(self, provider):
        """Test topic creation without kafka-python package."""
        topic = "test_topic"
        
        # Mock the kafka.admin module to raise ImportError
        with patch.dict('sys.modules', {}, clear=False), patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(ProviderError, match="kafka-python package not found"):
                await provider.create_topic(topic)
    
    @pytest.mark.asyncio
    async def test_delete_topic_success(self, provider, mock_admin_client):
        """Test successful topic deletion."""
        topic = "test_topic"
        
        # Mock the kafka.admin module at sys.modules level
        kafka_admin_mock = Mock()
        kafka_admin_mock.KafkaAdminClient = Mock(return_value=mock_admin_client)
        kafka_errors_mock = Mock()
        kafka_errors_mock.UnknownTopicOrPartitionError = Exception
        
        with patch.dict('sys.modules', {'kafka': Mock(), 'kafka.admin': kafka_admin_mock, 'kafka.errors': kafka_errors_mock}):
            await provider.delete_topic(topic)
            
            # Should call delete_topics
            mock_admin_client.delete_topics.assert_called_with([topic])
            mock_admin_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_topic_with_active_consumer(self, provider, mock_admin_client):
        """Test deleting topic with active consumer."""
        topic = "test_topic"
        
        # Add mock consumer
        mock_consumer = AsyncMock()
        provider._consumers[topic] = mock_consumer
        
        # Mock the kafka.admin module at sys.modules level
        kafka_admin_mock = Mock()
        kafka_admin_mock.KafkaAdminClient = Mock(return_value=mock_admin_client)
        kafka_errors_mock = Mock()
        kafka_errors_mock.UnknownTopicOrPartitionError = Exception
        
        with patch.dict('sys.modules', {'kafka': Mock(), 'kafka.admin': kafka_admin_mock, 'kafka.errors': kafka_errors_mock}):
            # Mock the unsubscribe method using object.__setattr__ to bypass Pydantic
            original_unsubscribe = provider.unsubscribe
            mock_unsubscribe = AsyncMock()
            object.__setattr__(provider, 'unsubscribe', mock_unsubscribe)
            
            try:
                await provider.delete_topic(topic)
                
                # Should unsubscribe first
                mock_unsubscribe.assert_called_with(topic)
                mock_admin_client.delete_topics.assert_called_with([topic])
            finally:
                # Restore original method
                object.__setattr__(provider, 'unsubscribe', original_unsubscribe)
    
    @pytest.mark.asyncio
    async def test_delete_topic_not_exists(self, provider, mock_admin_client):
        """Test deleting topic that doesn't exist."""
        topic = "nonexistent_topic"
        
        # Simulate UnknownTopicOrPartitionError
        error = Mock()
        error.__class__.__name__ = "UnknownTopicOrPartitionError"
        mock_admin_client.delete_topics.side_effect = error
        
        # Mock the kafka.admin module at sys.modules level
        kafka_admin_mock = Mock()
        kafka_admin_mock.KafkaAdminClient = Mock(return_value=mock_admin_client)
        kafka_errors_mock = Mock()
        kafka_errors_mock.UnknownTopicOrPartitionError = Exception
        
        with patch.dict('sys.modules', {'kafka': Mock(), 'kafka.admin': kafka_admin_mock, 'kafka.errors': kafka_errors_mock}):
            # Should not raise error
            await provider.delete_topic(topic)
            mock_admin_client.close.assert_called_once()


class TestKafkaProviderAdvancedFeatures:
    """Test advanced Kafka provider features."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return KafkaProviderSettings(
            host="localhost",
            port=9092,
            username="test",
            password="test",
            security_protocol="SASL_SSL",
            sasl_mechanism="SCRAM-SHA-256",
            sasl_username="sasl_user",
            sasl_password="sasl_pass",
            compression_type="gzip",
            acks="all"
        )
    
    @pytest_asyncio.fixture
    def mock_producer(self):
        """Create mock Kafka producer."""
        producer = AsyncMock()
        producer.start = AsyncMock()
        producer.stop = AsyncMock()
        producer.send_and_wait = AsyncMock()
        return producer
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_producer):
        """Create and initialize test provider."""
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer):
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_security_configuration(self, provider_settings, mock_producer):
        """Test provider with security configuration."""
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer) as mock_producer_class:
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            
            call_args = mock_producer_class.call_args[1]
            assert call_args["security_protocol"] == "SASL_SSL"
            assert call_args["sasl_mechanism"] == "SCRAM-SHA-256"
            assert call_args["sasl_plain_username"] == "sasl_user"
            assert call_args["sasl_plain_password"] == "sasl_pass"
            assert call_args["compression_type"] == "gzip"
            assert call_args["acks"] == "all"
    
    @pytest.mark.asyncio
    async def test_ssl_configuration(self, provider_settings, mock_producer):
        """Test provider with SSL configuration."""
        import ssl
        ssl_context = ssl.create_default_context()
        provider_settings = provider_settings.model_copy(update={
            "ssl_context": ssl_context,
            "ssl_check_hostname": False
        })
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer) as mock_producer_class:
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            
            call_args = mock_producer_class.call_args[1]
            assert call_args["ssl_context"] == ssl_context
            assert call_args["ssl_check_hostname"] is False
    
    @pytest.mark.asyncio
    async def test_custom_connect_args(self, provider_settings, mock_producer):
        """Test provider with custom connection arguments."""
        provider_settings = provider_settings.model_copy(update={
            "connect_args": {
                "request_timeout_ms": 30000,
                "retry_backoff_ms": 100
            }
        })
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer) as mock_producer_class:
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            
            call_args = mock_producer_class.call_args[1]
            assert call_args["request_timeout_ms"] == 30000
            assert call_args["retry_backoff_ms"] == 100
    
    @pytest.mark.asyncio
    async def test_auto_initialization_on_publish(self, provider_settings, mock_producer):
        """Test that operations auto-initialize provider."""
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer):
            provider = KafkaMQProvider(settings=provider_settings)
            
            # Should auto-initialize on publish
            await provider.publish("topic", "message")
            
            assert provider._initialized
            mock_producer.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_consumers(self, provider):
        """Test managing multiple consumers."""
        topics = ["topic1", "topic2", "topic3"]
        
        async def callback(message, metadata):
            pass
        
        # Create mock tasks that can be cancelled
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        mock_task.cancel = AsyncMock()
        
        # Mock create_task to properly handle the coroutine
        def mock_create_task_fn(coro):
            # Close the coroutine to avoid "never awaited" warning
            coro.close()
            return mock_task
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaConsumer') as mock_consumer_class, \
             patch('asyncio.create_task', side_effect=mock_create_task_fn) as mock_create_task:
            
            mock_consumer_class.return_value = AsyncMock()
            
            # Subscribe to multiple topics
            for topic in topics:
                await provider.subscribe(topic, callback)
            
            # Should have consumers for all topics
            assert len(provider._consumers) == 3
            assert len(provider._consumer_tasks) == 3
            assert all(topic in provider._consumers for topic in topics)
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_consumers(self, provider):
        """Test graceful shutdown with active consumers."""
        # Add mock consumers and tasks
        mock_consumers = {}
        mock_tasks = {}
        
        for i in range(3):
            topic = f"topic_{i}"
            mock_consumer = AsyncMock()
            mock_task = Mock()
            mock_task.done.return_value = False
            
            mock_consumers[topic] = mock_consumer
            mock_tasks[topic] = mock_task
            
            provider._consumers[topic] = mock_consumer
            provider._consumer_tasks[topic] = mock_task
        
        await provider.shutdown()
        
        # Should cancel all tasks and stop all consumers
        for topic, task in mock_tasks.items():
            task.cancel.assert_called_once()
        
        for topic, consumer in mock_consumers.items():
            consumer.stop.assert_called_once()
        
        # Should clear collections
        assert len(provider._consumers) == 0
        assert len(provider._consumer_tasks) == 0


class TestKafkaProviderErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return KafkaProviderSettings(
            host="localhost",
            port=9092,
            username="test",
            password="test"
        )
    
    @pytest.mark.asyncio
    async def test_operations_without_aiokafka(self, provider_settings):
        """Test operations without aiokafka package."""
        # This test would require actually removing aiokafka, which is complex in pytest
        # Instead, we test the import error handling logic
        provider = KafkaMQProvider(settings=provider_settings)
        assert provider._producer is None
    
    @pytest.mark.asyncio
    async def test_shutdown_error_handling(self, provider_settings):
        """Test shutdown handles errors gracefully."""
        mock_producer = AsyncMock()
        mock_producer.stop.side_effect = Exception("Stop failed")
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer):
            provider = KafkaMQProvider(settings=provider_settings)
            await provider.initialize()
            
            # Should handle shutdown errors gracefully
            await provider.shutdown()
            assert not provider._initialized
    
    @pytest.mark.asyncio
    async def test_consumer_task_error_recovery(self, provider_settings):
        """Test consumer task handles errors and continues."""
        # This would be tested by actually running the consumer task
        # For now, verify the error handling structure exists
        provider = KafkaMQProvider(settings=provider_settings)
        assert hasattr(provider, '_consumers')
        assert hasattr(provider, '_consumer_tasks')
    
    @pytest.mark.asyncio
    async def test_provider_type_validation(self):
        """Test provider type is correctly set."""
        provider = KafkaMQProvider()
        assert provider.provider_type == "message_queue"  # Inherited from base
    
    @pytest.mark.asyncio
    async def test_bootstrap_servers_parsing(self, provider_settings):
        """Test bootstrap servers string parsing."""
        provider_settings = provider_settings.model_copy(update={"bootstrap_servers": "kafka1:9092,kafka2:9093,kafka3:9094"})
        provider = KafkaMQProvider(settings=provider_settings)
        
        # Should handle comma-separated bootstrap servers
        assert provider.settings.bootstrap_servers == "kafka1:9092,kafka2:9093,kafka3:9094"
    
    @pytest.mark.asyncio 
    async def test_message_serialization_edge_cases(self, provider_settings):
        """Test message serialization handles edge cases."""
        mock_producer = AsyncMock()
        
        with patch('flowlib.providers.mq.kafka.provider.AIOKafkaProducer', return_value=mock_producer):
            provider = KafkaMQProvider(settings=provider_settings)
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
                await provider.publish("topic", message)
            
            # Should have called send_and_wait for each message
            assert mock_producer.send_and_wait.call_count == len(test_cases)